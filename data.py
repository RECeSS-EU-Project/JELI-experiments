#coding: utf-8

import torch
import random
import os
import numpy as np
import pandas as pd
from qnorm import quantile_normalize
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
from jeli.JELIImplementation import RHOFM, to_cuda, broadcast_cat
from jeli.JELI import JELI

######################################################
## Missing mechanisms                               ##
###################################################### 

## Gaussian self-masking
## mechanism to induce sparsity in the features
## https://arxiv.org/pdf/2007.01627
def GSM(X, sparsity_features=0.3, seed=1234):
	np.random.seed(seed)
	random.seed(seed)
	Ks = np.random.normal(0,0.5,size=X.shape[0])
	Ks -= np.min(Ks)
	Ks /= np.max(Ks)
	def gsm_prob(X, k):
		m, s = np.mean(X[k,:]), np.sqrt(np.var(X[k,:]))
		pks = (1-sparsity_features)*Ks[k]*norm.pdf(X[k,:], loc=m, scale=s)*np.sqrt(2*np.pi*s**2)
		return [(p, 1-p) for p in pks]
	M = np.array([[np.random.choice([0,1], size=1, replace=True, p=ps) for ps in gsm_prob(X, k)] for k in range(X.shape[0])]).reshape(X.shape)
	assert M.shape == X.shape
	return M

## Completely At Random	
def MCAR(X, sparsity_features=0.3, seed=1234):
	np.random.seed(seed)
	random.seed(seed)
	return np.random.choice([0,1], p=[sparsity_features, 1-sparsity_features], size=X.shape, replace=True)

######################################################
## Synthetic interpretable datasets                 ##
###################################################### 

## Logistic regression
def generate_deviated_dataset(npoints, nfeatures, ndim, sparsity, sparsity_features=0.3, binary=False, seed=1234, var=0.1, cuda_on=(torch._C._cuda_getDeviceCount()>0), verbose=False):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True

	assert sparsity>=0. and sparsity<1
	assert int(np.sqrt(npoints))==np.sqrt(npoints), int(np.sqrt(npoints))*int(np.sqrt(npoints))
	feature_embeddings = torch.Tensor(
		#np.random.normal(0,var,(2*nfeatures,ndim))
		np.random.normal(0,var,(nfeatures,ndim))
	)
	if (binary):
		feature_embeddings = normalize(feature_embeddings.numpy())
		feature_embeddings = (feature_embeddings>0.5).astype(int)-(feature_embeddings<-0.5).astype(int)
		feature_embeddings = torch.Tensor(feature_embeddings)		
	nentity = int(np.sqrt(npoints))
	lst1, lst2 = [list(map(str, range(x))) for x in [nentity, nfeatures]]
	P_, S_ = [np.random.normal(0,var,(nfeatures, nentity)) for _ in range(2)]
	P, S = [pd.DataFrame(normalize( 
			np.multiply( X , 
				    #MCAR(X, sparsity_features, seed) 
				    GSM(X, sparsity_features, seed)
			)
		), index=lst2, columns=lst1) for X in [P_,S_]]
	def gen0_model(inp):
		item, user = inp
		x = to_cuda(broadcast_cat([item, user], dim=-1), cuda_on)
		w = to_cuda(broadcast_cat([feature_embeddings]*2, dim=0), cuda_on)
		return torch.sigmoid(torch.matmul(x, w).sum(1, keepdim=True))
	SS = pd.concat(tuple([S for j in lst1]), axis=1)
	SS.columns = range(npoints)
	PP = pd.concat(tuple([P[[j]] for j in lst1 for _ in range(nentity)]), axis=1)
	PP.index = PP.index.astype(int)+nfeatures
	PP.columns = SS.columns
	SSPP = pd.concat((SS, PP), axis=0)
	S_ , P_ = torch.Tensor(SSPP.iloc[:nfeatures].values).T, torch.Tensor(SSPP.iloc[nfeatures:].values).T
	A_ = torch.sigmoid(gen0_model(tuple([to_cuda(X, cuda_on) for X in [S_, P_]])).reshape(nentity, nentity).T)
	A = A_.clone()
	xp = (1-sparsity)/2
	sxp = sparsity+xp
	thres, _thres = torch.quantile(A_, sxp), torch.quantile(A_, xp)
	A[A_>=thres] = 1
	A[A_<=_thres] = -1
	A[(A_<thres)&(A_>_thres)] = 0
	if (verbose):
		print("Sparsity %.2f perc. (expected %.2f, %d >0)" % ((A==0).sum()/np.prod(A.cpu().numpy().shape), sparsity, (A!=0).sum()))
	A = pd.DataFrame(A.cpu().numpy(), index=S.columns, columns=P.columns)
	data_args = {"ratings": A, "users": P, "items": S}
	return data_args, feature_embeddings, gen0_model, A_

## Factorization Machine    
## 
def generate_interpretable_dataset(npoints, nfeatures, ndim, order, sparsity, structure, sparsity_features=0.3, binary=False, draw_coefs=True, seed=1234, var=0.1, cuda_on=(torch._C._cuda_getDeviceCount()>0), verbose=False):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True

	assert sparsity>=0. and sparsity<1
	assert int(np.sqrt(npoints))==np.sqrt(npoints), int(np.sqrt(npoints))*int(np.sqrt(npoints))
	feature_embeddings = torch.Tensor(
		np.random.normal(0,var,(nfeatures,ndim))
	)
	if (binary):
		feature_embeddings = normalize(feature_embeddings.numpy())
		feature_embeddings = (feature_embeddings>0.5).astype(int)-(feature_embeddings<-0.5).astype(int)
		feature_embeddings = torch.Tensor(feature_embeddings)		
	nentity = int(np.sqrt(npoints))
	lst1, lst2 = [list(map(str, range(x))) for x in [nentity, nfeatures]]
	P, S = [np.random.normal(0,var,(nfeatures, nentity)) for _ in range(2)]
	P, S = [pd.DataFrame(normalize( 
			np.multiply( X , 
				    GSM(X, sparsity_features, seed)
			)
		), index=lst2, columns=lst1) for X in [P,S]]
	gen0_model = RHOFM(ndim, order, structure, True, cuda_on)
	if (draw_coefs):
		w0, w1, w2 = [to_cuda(torch.normal(0,var,shp), cuda_on) for shp in [(1,1),(ndim,1),(order-1,1)]] 
		gen0_model.theta0 = w0
		gen0_model.theta1 = w1
		gen0_model.theta2 = w2
	SS = pd.concat(tuple([S for j in lst1]), axis=1)
	SS.columns = range(npoints)
	PP = pd.concat(tuple([P[[j]] for j in lst1 for _ in range(nentity)]), axis=1)
	PP.index = PP.index.astype(int)+nfeatures
	PP.columns = SS.columns
	SSPP = pd.concat((SS, PP), axis=0)
	S_ , P_ = torch.Tensor(SSPP.iloc[:nfeatures].values).T, torch.Tensor(SSPP.iloc[nfeatures:].values).T
	A_ = torch.sigmoid(gen0_model(tuple([to_cuda(X, cuda_on) for X in [S_, P_, feature_embeddings]])).reshape(nentity, nentity).T)
	A = A_.clone()
	xp = (1-sparsity)/2
	sxp = sparsity+xp
	thres, _thres = torch.quantile(A_, sxp), torch.quantile(A_, xp)
	A[A_>=thres] = 1
	A[A_<=_thres] = -1
	A[(A_<thres)&(A_>_thres)] = 0
	if (verbose):
		print("Sparsity %.2f perc. (expected %.2f, %d >0)" % ((A==0).sum()/np.prod(A.cpu().numpy().shape), sparsity, (A!=0).sum()))
	A = pd.DataFrame(A.cpu().numpy(), index=S.columns, columns=P.columns)
	data_args = {"ratings": A, "users": P, "items": S}
	return data_args, feature_embeddings, gen0_model, A_

if __name__=="__main__":

	from stanscofi.datasets import Dataset
	from scipy.sparse import coo_array

	X = np.random.normal(0, 1, (100,200))
	sparsity_features = 0.3
	seed = 1234

	M = GSM(X, sparsity_features, seed)
	print("%.2f +- %.4f (loose target: %.2f)" % (np.mean(np.mean(1-M, axis=1)), np.var(np.mean(1-M, axis=1)), sparsity_features))
	M = MCAR(X, sparsity_features, seed)
	print("%.2f +- %.4f (true: %.2f)" % (np.mean(np.mean(1-M, axis=1)), np.var(np.mean(1-M, axis=1)), sparsity_features))
	
	npoints, nfeatures = 26896, 10  
	ndim = 5
	order, structure = 2, "linear"
	sparsity = 0.
	binary = True 
	draw_coefs = False
	
	## Interpretable dataset (Factorization Machine)
	data_args, _, _, A_ = generate_interpretable_dataset(npoints, nfeatures, ndim, order, sparsity, structure, sparsity_features, binary, draw_coefs, seed, verbose=True)
	Dataset(**data_args).visualize()
	JELI().print_scores(coo_array(A_.numpy().flatten()))

	## Interpretable dataset (Logistic Regression)	
	data_args, _, _, A_ = generate_deviated_dataset(npoints, nfeatures, ndim, sparsity, sparsity_features, binary, seed, verbose=True)
	Dataset(**data_args).visualize()
	JELI().print_scores(coo_array(A_.numpy().flatten()))
