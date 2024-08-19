#coding: utf-8

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_array
import os
import random

from stanscofi.training_testing import random_cv_split
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import negative_sampling

from jeli.JELIImplementation import to_cuda, broadcast_cat

###################################################################################
## Regular order 2 Factorization Machine                                         ##
###################################################################################
class FM2(nn.Module):
    '''
    Regular order 2 Factorization Machine 

    ...

    Parameters
    ----------
    d : int
        dimension
    itemF : int
        number of item features
    userF : int
        number of user features (default: equal to itemF)
    cuda_on : bool
    	use cuda if True

    Attributes
    ----------
    theta2 : torch.nn.Parameter of size (d, 2F)
    	the coefficients of terms of order 2
    theta1 : torch.nn.Parameter of size (1, 2F)
        the coefficients of term of order 1
    theta0 : torch.nn.Parameter of size (1, 1)
        the coefficient of term of order 0

    Methods
    -------
    __init__(d, order, structure, cuda_on)
        Initializes the FM2
    forward(item, user)
        Computes the FM2 function on pair [item, user] 
    '''
    def __init__(self, d, itemF, userF=None, cuda_on=False, random_state=1234):
        '''
        Creates an instance of order 2 FM

        ...

        Parameters
        ----------
    	d : int
            dimension
    	itemF : int
            number of item features
        userF : int
            number of user features (default: equal to itemF)
    	cuda_on : bool
            use cuda if True
        '''
        super().__init__()
        if (userF is None):
            userF = itemF
        self.theta2 = to_cuda(nn.Parameter(torch.randn(itemF+userF, d), requires_grad=True), cuda_on)
        self.theta1 = to_cuda(nn.Parameter(torch.randn(itemF+userF, 1), requires_grad=True), cuda_on)
        self.theta0 = to_cuda(nn.Parameter(torch.randn(1, 1), requires_grad=True), cuda_on) ##
        self.cuda_on = cuda_on
        self.seed_everything(random_state)
        
    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def forward(self, inp):
        '''
        Outputs properly formatted scores (not necessarily in [0,1]!) from the fitted model on test_dataset. Internally calls model_predict() then reformats the scores

        ...

        Parameters
        ----------
        test_dataset : stanscofi.Dataset
            dataset on which predictions should be made

        Returns
        ----------
        scores : COO-array of shape (n_items, n_users)
            sparse matrix in COOrdinate format, with nonzero values corresponding to predictions on available pairs in the dataset
        '''
        item, user = inp
        x = to_cuda(broadcast_cat([item, user], dim=-1), self.cuda_on) ## concatenate user and item feature vectors
        out = self.theta0 + torch.matmul(x, self.theta1)  
        out_pair1 = torch.matmul(x, self.theta2).pow(2).sum(1, keepdim=True) 
        out_pair2 = torch.matmul(x.pow(2), self.theta2.pow(2)).sum(1, keepdim=True)
        out += 0.5*(out_pair1-out_pair2)
        return out
       
    def sample(self, dataset, batch_size, stype="negative", batch_seed=1234, num_neg_samples=3, method="sparse", force_undirected=False):  
        assert stype in ["uniform", "negative"] 
        if (stype == "uniform"):  
            n = len(dataset.folds.data)
            batch_size = min(batch_size, n)
            nbatches = n//batch_size+int(n%batch_size!=0)
            cv_generator = StratifiedKFold(n_splits=nbatches, shuffle=True, random_state=batch_seed)
            batch_folds, _ = random_cv_split(dataset, cv_generator, metric="cosine")
            return [b for _, b in batch_folds]
        elif (stype == "negative"):  
            self.num_neg_samples = num_neg_samples
            pos_folds = dataset.ratings.toarray()
            pos_folds = torch.as_tensor(np.argwhere(pos_folds).T)
            batch_size = min(batch_size, pos_folds.shape[1])
            nbatches = pos_folds.shape[1]//batch_size+int(pos_folds.shape[1]%batch_size!=0)
            neg_folds = negative_sampling(pos_folds, num_nodes=dataset.ratings.shape, num_neg_samples=num_neg_samples, method=method, force_undirected=force_undirected)
            batch_folds = []
            pos_folds, neg_folds = pos_folds.numpy(), neg_folds.numpy()
            for batch in range(nbatches):
                pfolds = pos_folds[:,batch*batch_size//2:(batch+1)*batch_size//2]
                nfolds = neg_folds[:,batch*batch_size//2:(batch+1)*batch_size//2]
                data = np.array([1]*(pfolds.shape[1]+nfolds.shape[1]))
                row = np.concatenate((pfolds[0,:], nfolds[0,:]), axis=0)
                col = np.concatenate((pfolds[1,:], nfolds[1,:]), axis=0)
                batch_folds.append(coo_array((data, (row, col)), shape=dataset.ratings.shape))
            return batch_folds ## outputs *ONE* batch if too few positive
       
     ## SGD and CD pipelines in https://arxiv.org/pdf/1607.07195.pdf
    def fit(self, dataset, n_epochs=25, batch_size=100, optimizer_class=torch.optim.AdamW, loss="MarginRankingLoss", opt_params={'lr': 0.01, "weight_decay": 0.02}, early_stop=0, random_seed=1234):
        assert np.isfinite(dataset.items.toarray()).all()
        assert np.isfinite(dataset.users.toarray()).all()
        params = [x for x in [self.theta0, self.theta1, self.theta2] if ('torch.nn.parameter.Parameter' in str(type(x)))]
        optimizer = optimizer_class(tuple(params), **opt_params)
        train_losses = []
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [cyclical_lr(10000)])
        old_epoch_loss, early_stop_counter = float("inf"), 0
        with tqdm(total=n_epochs * len(dataset.folds.data)) as pbar:
            for epoch in range(n_epochs):
                epoch_loss, epoch_train_losses, n_epoch = 0, [], 0
                #for batch_id, batch_fold in (pbar := tqdm(enumerate(self.sample(dataset, batch_size, batch_seed=random_seed+epoch)))):
                for batch_id, batch_fold in enumerate(self.sample(dataset, batch_size, batch_seed=random_seed+epoch)):
                    batch_y = to_cuda(torch.LongTensor(dataset.ratings.toarray()[batch_fold.row,batch_fold.col].ravel()), self.cuda_on)
                    batch_y[batch_y<0] = 0
                    items = dataset.items.toarray().T[batch_fold.row,:]
                    users = dataset.users.toarray().T[batch_fold.col]
                    batch_item, batch_user = [to_cuda(torch.Tensor(X), self.cuda_on) for X in [items, users]]
                    optimizer.zero_grad()
                    out = self((batch_item, batch_user))
                    try:
                    	out_cross = torch.cat((out, -out), dim=1) ## cross-enthropy
                    	loss_epoch = eval("nn."+loss)()(out_cross, batch_y)
                    except:
                    	assert self.num_neg_samples>0
                    	out_neg = out[batch_y!=1]  ## pairwise
                    	out_pos = out[batch_y==1].repeat((out_neg.size(dim=0)//out[batch_y==1].size(dim=0)+1,1))[:out_neg.size(dim=0)]
                    	target = torch.ones(out_neg.size())
                    	#print((out[batch_y==1].size(), self.num_neg_samples, out_pos.size(), target.size(), out_neg.size()))
                    	loss_epoch = eval("nn."+loss)()(out_pos, out_neg, target)
                    with torch.set_grad_enabled(True):
                    	loss_epoch.backward()
                    	#scheduler.step()
                    	optimizer.step()
                    n_epoch += batch_item.size(0)
                    epoch_loss += loss_epoch.item()*batch_item.size(0)
                    pbar.set_description("Batch #%d (epoch %d): loss %f (prev %f)" % (batch_id+1, epoch+1, epoch_loss/n_epoch, np.nan if (len(epoch_train_losses)==0) else epoch_train_losses[-1]))
                    epoch_train_losses.append(epoch_loss/n_epoch)
                train_losses.append(epoch_train_losses)
                if (old_epoch_loss<epoch_loss):
                    early_stop_counter += 1
                    old_epoch_loss = epoch_loss
                if ((early_stop>0) and (early_stop_counter>early_stop)):
                    break
        return train_losses
        
    def predict_proba(self, dts, default_zero_val=1e-31):
        items = torch.Tensor(dts.items.toarray().T[dts.folds.row,:])
        users = torch.Tensor(dts.users.toarray().T[dts.folds.col])
        scores_data = torch.sigmoid(self((items, users))).detach().cpu().numpy().flatten()
        scores = coo_array((scores_data, (dts.folds.row, dts.folds.col)), shape=dts.ratings.shape)
        scores = scores.toarray()
        default_val = min(default_zero_val, np.min(scores[scores!=0])/2 if ((scores!=0).any()) else default_zero_val)
        scores[(scores==0)&(dts.folds.toarray()==1)] = default_val
        scores = coo_array(coo_array(scores)*dts.folds)
        return scores
        
    def predict(self, scores, threshold=0.5):
        preds = coo_array((scores.toarray()!=0).astype(int)*((-1)**(scores.toarray()<=0.5)))
        return preds
        
###################################################################################
## Cross item/user terms order 2 Factorization Machine                           ##
###################################################################################
        
class CrossFM2(FM2):
    '''
    Custom order 2 Factorization Machine 
    
    Assumes the same number of features for items and users
    Pairwise interaction terms only use cross products between items and users

    ...

    Parameters
    ----------
    d : int
        dimension
    F : int
        number of shared item and user features
    cuda_on : bool
    	use cuda if True

    Attributes
    ----------
    theta2 : torch.nn.Parameter of size (d, F)
    	the coefficients of terms of order 2
    theta1 : torch.nn.Parameter of size (1, F)
        the coefficients of term of order 1
    theta0 : torch.nn.Parameter of size (1, 1)
        the coefficient of term of order 0

    Methods
    -------
    __init__(d, order, structure, cuda_on)
        Initializes the FM2
    forward(item, user)
        Computes the FM2 function on pair [item, user] 
    '''
    def __init__(self, d, F, cuda_on, random_state=1234):
        '''
        Creates an instance of order 2 FM

        ...

        Parameters
        ----------
    	d : int
            dimension
    	F : int
            number of shared item and user features
    	cuda_on : bool
            use cuda if True
        '''
        super().__init__(d, F, 0, cuda_on, random_state)
        
    def forward(self, inp):
        '''
        Outputs properly formatted scores (not necessarily in [0,1]!) from the fitted model on test_dataset. Internally calls model_predict() then reformats the scores

        ...

        Parameters
        ----------
        test_dataset : stanscofi.Dataset
            dataset on which predictions should be made

        Returns
        ----------
        scores : COO-array of shape (n_items, n_users)
            sparse matrix in COOrdinate format, with nonzero values corresponding to predictions on available pairs in the dataset
        '''
        item, user = inp
        x = to_cuda(broadcast_cat([item, user], dim=-1), self.cuda_on) ## concatenate user and item feature vectors
        
        out = self.theta0 + torch.matmul(item+user, self.theta1)  
        out += (torch.matmul(item, self.theta2) * torch.matmul(user, self.theta2)).sum(1, keepdim=True)
        return out
        
###################################################################################
## Test cross item/user term computation                                         ##
###################################################################################
        
if __name__=="__main__":

    #from time import time

    #def sample(d, F, n=1):
    #    i = torch.randn(n, F)
    #    u = torch.randn(n, F)
    #    v = torch.randn(d, F)
    #    return i, u, v

    #def f1(i, u, v):
    #    F = i.size(dim=1)
    #    out = torch.sum(torch.Tensor([torch.matmul(v[:,k].T, v[:,l])*i[:,k]*u[:,l] for k in range(F) for l in range(F)]))
    #    return out
    
    #def f2(i, u, v):
    #    F = i.size(dim=1)
    #    out = torch.sum(torch.matmul(v, i.T) * torch.matmul(v, u.T))
    #    return out

    #def run(d, F, n=1):
    #    i, u, v = sample(d, F, n)
    #    t=time()
    #    print(f1(i, u, v))
    #    print(time()-t)
    #    t=time()
    #    print(f2(i, u, v))
    #    print(time()-t)
       
	import random
	from stanscofi.training_testing import random_simple_split
	from stanscofi.utils import load_dataset
	from stanscofi.datasets import generate_dummy_dataset, Dataset
	from stanscofi.validation import AUC, NDCGk, compute_metrics

	SEED = 1245
	TEST_SIZE = 0.2
	DATA_NAME = "Synthetic"
	DFOLDER="../datasets/"
	
	if (DATA_NAME=="Synthetic"):
		npositive, nnegative, nfeatures, mean, std = 200, 200, 100, 2, 0.01
		data_args = generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std, random_state=SEED)
	else:
		data_args = load_dataset(DATA_NAME, save_folder=DFOLDER)
		data_args["name"] = DATA_NAME
	
	N_DIMENSIONS = 20
	N_EPOCHS=25
	BATCH_SIZE=1024
	
	np.random.seed(SEED)
	random.seed(SEED)

	## Import dataset
	dataset = Dataset(**data_args)
	
	(train_folds, test_folds), _ = random_simple_split(dataset, TEST_SIZE, metric="cosine", random_state=SEED)
	train = dataset.subset(train_folds)
	test = dataset.subset(test_folds)

	print("\n----------- FM2")
	FM2model = FM2(N_DIMENSIONS, dataset.nitem_features, dataset.nuser_features, False)
	FM2model.fit(train, n_epochs=N_EPOCHS, loss="CrossEntropyLoss", batch_size=BATCH_SIZE, random_seed=SEED, opt_params={'lr': 0.01, "weight_decay": 0.02})
	scores = FM2model.predict_proba(test)
	predictions = FM2model.predict(scores, threshold=0.)
	metrics, _ = compute_metrics(scores, predictions, test, metrics=["AUC", "NDCGk"], k=dataset.nitems, beta=1, verbose=False)
	print(metrics)
	y_test = (test.folds.toarray()*test.ratings.toarray()).ravel()
	y_test[y_test<1] = 0
	print("(global) AUC = %.3f" % AUC(y_test, scores.toarray().ravel(), 1, 1))
	print("(global) NDCG@%d = %.3f" % (test.nitems, NDCGk(y_test, scores.toarray().ravel(), test.nitems, 1)))

	print("\n----------- CrossFM")
	CrossFM2model = CrossFM2(N_DIMENSIONS, dataset.nitem_features, False)
	CrossFM2model.fit(train, n_epochs=N_EPOCHS, loss="CrossEntropyLoss", batch_size=BATCH_SIZE, random_seed=SEED, opt_params={'lr': 0.01, "weight_decay": 0.02})
	scores = CrossFM2model.predict_proba(test)
	predictions = CrossFM2model.predict(scores, threshold=0.)
	metrics, _ = compute_metrics(scores, predictions, test, metrics=["AUC", "NDCGk"], k=dataset.nitems, beta=1, verbose=False)
	print(metrics)
	y_test = (test.folds.toarray()*test.ratings.toarray()).ravel()
	y_test[y_test<1] = 0
	print("(global) AUC = %.3f" % AUC(y_test, scores.toarray().ravel(), 1, 1))
	print("(global) NDCG@%d = %.3f" % (test.nitems, NDCGk(y_test, scores.toarray().ravel(), test.nitems, 1)))

