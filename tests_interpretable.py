#coding: utf-8

import torch
import numpy as np
import random
import pandas as pd

from data import generate_interpretable_dataset, generate_deviated_dataset
from stanscofi.datasets import Dataset
from stanscofi.training_testing import random_simple_split
from stanscofi.validation import AUC, NDCGk, compute_metrics
from benchscofi.utils import rowwise_metrics
from jeli.JELI import JELI

import matplotlib.pyplot as plt

## Parameters (instance)
cuda_on=(torch._C._cuda_getDeviceCount()>0)
if (cuda_on):
	folder, kge_folder=["/storage/store3/work/creda/"]*2
else:
	folder, kge_folder = "./results/", "files/"
seed = 1234
npoints, nfeatures = 29929, 10 
ndim = 2 
order, structure = 2, "linear"
sparsity = 0.
sparsity_features = 0.8
binary = True
draw_coefs = False
dvar = 1.

dataset_types = ["interpretable", "deviated"]

## Parameters (JELI)
nepochs, sim_thres, lrate = 60, 0.75, 1e-3
frozen = not draw_coefs
ndim_ = ndim
order_ = order
structure_ = structure

np.random.seed(seed)
random.seed(seed)

if __name__=="__main__":

	plot_it = False
	Ndata=10   ## iterate over several randomly generated instances
	Niter=100  ## iterate over several random seeds for the same instance

	import pickle
	import os
	if (os.path.exists("%s/results_interpretable.pck" % folder)):
		with open("%s/results_interpretable.pck" % folder, "rb") as f:
			results = pickle.load(f)	
	else:
		results = {}

	for idd, dataset_type in enumerate(dataset_types):
		print("* Dataset type %s (%d/%d)" % (dataset_type, idd+1, len(dataset_types)))
		if (dataset_type in results):
			continue
			
		data_seeds = np.random.choice(range(int(1e8)), size=Ndata, replace=False)
		
		if (os.path.exists("%s/results_interpretable_%s.pck" % (folder, dataset_type))):
			with open("%s/results_interpretable_%s.pck" % (folder, dataset_type), "rb") as f:
				results_data = pickle.load(f)	
		else:
			results_data = {}
	
		for iss, data_seed in enumerate(data_seeds):
			print("** Data seed %d (%d/%d) -- Dataset type %s (%d/%d)" % (data_seed, iss+1, len(data_seeds), dataset_type, idd+1, len(dataset_types)))
			
			if (data_seed in results_data):
				continue
				
			###################################################
			## Dataset                                       ##
			###################################################
			if (dataset_type == "interpretable"):
				## Interpretable dataset (Factorization Machine)
				data_args, feature_embeddings, gen0_model, A_ = generate_interpretable_dataset(npoints, nfeatures, ndim, order, sparsity, structure, sparsity_features=sparsity_features, binary=binary, draw_coefs=draw_coefs, seed=data_seed, var=dvar, cuda_on=cuda_on, verbose=True)
			elif (dataset_type == "deviated"):
				## Interpretable dataset (Logistic Regression)	
				data_args, feature_embeddings, gen0_model, A_ = generate_deviated_dataset(npoints, nfeatures, ndim, sparsity, sparsity_features=sparsity_features, binary=binary, seed=data_seed, var=dvar, cuda_on=cuda_on, verbose=True)
			else:
				raise ValueError
			dataset = Dataset(**data_args)
		
			###################################################
			## Split                                         ##
			###################################################
			(train_folds, test_folds), _ = random_simple_split(dataset, 0.2, random_state=data_seed)
			train = dataset.subset(train_folds)
			test = dataset.subset(test_folds)
			## TODO add cross validation
			
			iter_seeds = np.random.choice(range(int(1e8)), size=Niter, replace=False)
			
			if (os.path.exists("%s/results_interpretable_%s_%d.pck" % (folder, dataset_type, data_seed))):
				with open("%s/results_interpretable_%s_%d.pck" % (folder, dataset_type, data_seed), "rb") as f:
					results_iter_all = pickle.load(f)	
			else:
				results_iter_all = {}

			for itt, iter_seed in enumerate(iter_seeds):
				print("*** Iter seed %d (%d/%d) -- Data seed %d (%d/%d) -- Dataset type %s (%d/%d)" % (iter_seed, itt+1, len(iter_seeds), data_seed, iss+1, len(data_seeds), dataset_type, idd+1, len(dataset_types)))
			
				if (iter_seed in results_iter_all):
					continue
					
				results_iter = {}

				###################################################
				## JELI training                                 ##
				###################################################
				model = JELI({"cuda_on": cuda_on, "lr": lrate, "epochs": nepochs, "loss": "SoftMarginRankingLoss",  "n_dimensions": ndim_, "partial_kge": None, "sim_thres": sim_thres, "structure": structure_, "order": order_, "frozen": frozen, "random_seed": iter_seed})
				model.fit(train)
				embs = model.model["feature_embeddings"]
				print("")

				## Training report
				if (plot_it):
					plt.plot(range(len(model.model["losses"])), model.model["losses"], "b-")
					plt.plot([np.argmin(model.model["losses"])]*2, [model.model["losses"][np.argmin(model.model["losses"])]]*2, "r--")
					plt.savefig("training_loss_%s_%d_%d.png" % (dataset_type, data_seed, iter_seed), bbox_inches="tight")
					plt.close()
				#print((np.argmin(model.model["losses"]), model.model["losses"][np.argmin(model.model["losses"])]))
				#print([(-1)**(model.model["losses"][i]-model.model["losses"][i+1]>0) for i in range(len(model.model["losses"])-1)])
				print([np.round(model.model["losses"][i]-model.model["losses"][i+1],3) for i in range(len(model.model["losses"])-1)])
				if (feature_embeddings.shape == embs.numpy().shape):
					print("- features %.3f %.3f" % (np.linalg.norm(feature_embeddings-embs)**2/np.prod(embs.shape), 1-np.linalg.norm((feature_embeddings>0)^(embs>0))**2/np.prod(embs.shape)))	
				#print(feature_embeddings-embs)
				#print(embs)
				if (not frozen):
					print("- w0 %.3f %.3f" % (float(gen0_model.theta0), float(model.model["model"].FM.theta0)))
					print("- w1 %.3f" % (1-np.linalg.norm((gen0_model.theta1.numpy()>0)^(model.model["model"].FM.theta1.detach().numpy()>0))**2/np.prod(gen0_model.theta1.numpy().shape)))
					print("- w2 %.3f %.3f" % (float(gen0_model.theta2), float(model.model["model"].FM.theta2)))
				print("")

				###################################################
				## JELI evaluation                               ##
				###################################################
				scores = model.predict_proba(test)
				model.print_scores(scores)
				predictions = model.predict(scores, threshold=0.5)
				model.print_classification(predictions)

				## Validation metrics
				metrics, _ = compute_metrics(scores, predictions, test, metrics=["AUC", "NDCGk"], k=dataset.nitems, beta=1, verbose=False)
				y_test = (test.folds.toarray()*test.ratings.toarray()).ravel()
				y_test[y_test<1] = 0
				auc = AUC(y_test, scores.toarray().ravel(), 1, 1)
				nsauc = np.mean(rowwise_metrics.calc_auc(scores, test))
				print(metrics)
				print("(global) AUC = %.3f" % auc)
				print("(global) NDCG@%d = %.3f" % (test.nitems, NDCGk(y_test, scores.toarray().ravel(), test.nitems, 1)))
				print("(average) NS-AUC = %.3f" % nsauc)
				print("")
				
				results_iter.update({"AUC": auc, "NS-AUC": nsauc})

				N=min(npoints, 10)
				if (False):
					y_data = (dataset.folds.toarray()*dataset.ratings.toarray()).ravel()
					print(pd.DataFrame( np.vstack(( A_.numpy().flatten(), y_data, y_test, scores.toarray().ravel() )) , index=["true", "full", "bin", "pred"], columns=range(npoints) ).round(2).T.sort_values(by="pred", ascending=False).T.iloc[:,list(range(N+1))+[npoints-i for i in range(N, 0, -1)]] )
					print("")

				###################################################
				## Interpretability                              ##
				###################################################
				f_lst = ["F%d" % (i+1) for i in range(embs.shape[0])]

				## JELI feature importance scores
				feature_scores = embs.sum(axis=1)
				## "True" feature importance scores
				feature_scores_true = feature_embeddings.sum(axis=1)
				
				results_iter.update({"feature_scores_true": feature_scores_true, "feature_scores": feature_scores})

				imp_scores = pd.DataFrame([np.round(feature_scores.numpy(),2).flatten().tolist(), feature_scores_true.numpy().flatten().tolist()], index=["jeli","true"],columns=f_lst)
				print(imp_scores.T.sort_values(by="true", ascending=False).T)
				print("")

				########### 1. Test the significance of each feature (not using the predicted importance scores)
				import statsmodels.api as sm
				import statsmodels.stats.multitest as mt
				import statsmodels.formula.api as smf
					
				if (dataset.nitem_features<30):
					## https://stackoverflow.com/questions/50117157/how-can-i-do-test-wald-in-python
					## https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.t_test.html
					## test for each feature the null hypothesis "coefficient for feature in linear regression = 0" on the testing set
					labels = test.ratings.toarray()[test.folds.row,test.folds.col].ravel().reshape(-1,1)
					X = np.array([(test.users.toarray()[:,test.folds.col[i]]+test.items.toarray()[:,test.folds.row[i]]).flatten() for i in range(len(test.folds.data))])
					data = pd.DataFrame(np.concatenate((X, labels), axis=1), index=range(len(test.folds.data)), columns=["label"]+f_lst)
					mlr = smf.ols(formula="label ~ "+(" + ".join(f_lst)), data=data).fit()
					res_di = {}
					for iff, f in enumerate(f_lst):
						waldtest = mlr.wald_test("(Intercept = 0, %s = 1)" % f, scalar=True) #(r_matrix="(Intercept = 0, %s = 1)" %f, use_f=True)
						res_di.setdefault(f, {"F": np.round(float(waldtest.fvalue)), "p": float(waldtest.pvalue), "pred_score": np.round(float(feature_scores[iff]),2), "true_score": np.nan if (feature_embeddings is None) else float(feature_embeddings.sum(axis=1)[iff])})
					res_df = pd.DataFrame(res_di)
					reject, pvals_corrected, _, _ = mt.multipletests(res_df.loc["p"].values.ravel(), alpha=0.01, method="fdr_bh", maxiter=1)
					res_df.loc["corr(p)"] = pvals_corrected
					res_df.loc["reject"] = ["***"*int(t) for t in reject]
					print(res_df.T.sort_values(by="F", ascending=False).T)
				print("")

				########### 2. Compute the Jensen Shannon Distance
				from scipy.spatial.distance import jensenshannon
				fp, ft = [(f.abs()/f.abs().sum()).reshape(-1,1) for f in [feature_scores, feature_scores_true]]
				jsdist = float(jensenshannon(fp, ft))
				print("Jensen Shannon Distance = %.4f" % jsdist)
				print("")
				
				results_iter.update({"jsdist": jsdist})

				########### 3. Compute the Spearman’s rank-order correlation
				if (plot_it):
					plotdata = pd.DataFrame({'predicted': feature_scores, "true": feature_scores_true}, index=f_lst)
					plotdata.plot(kind="bar")
					plt.plot([0, embs.shape[0]], [0]*2, "k--")
					plt.plot(range(embs.shape[0]), feature_scores, "b-")
					plt.plot(range(embs.shape[0]), feature_scores_true, "r-")
					plt.title("Spearman's correlation %.2f" % plotdata.corr(method="spearman").values[0,1])
					plt.savefig("barplot_features_%s_%d_%d.png" % (dataset_type, data_seed, iter_seed), bbox_inches="tight")
					plt.close()

				from scipy.stats import spearmanr
				res = spearmanr(feature_scores, feature_scores_true, alternative="two-sided")
				print("Spearman's correlation = %.4f (p=%f)" % (res.correlation, res.pvalue))
				print("")
				
				results_iter.update({"spearman-p": res.pvalue, "spearman-rho": res.correlation})

				########### 4. Run a proportion statistical test
				from scipy.stats import chisquare, ks_2samp
				from copy import deepcopy
				## 4.a Kolmogorov-Smirnov test
				res_ks = ks_2samp(feature_scores, feature_scores_true, alternative="two-sided")
				## 4.b Empirical test
				p_ = []
				r1, r2 = deepcopy(feature_scores), deepcopy(feature_scores_true)
				for _ in range(10000):
					np.random.seed(int(1e6))
					np.random.shuffle(r1)
					np.random.shuffle(r2)
					res_ks_ = ks_2samp(r1, r2, alternative="two-sided")
					p_.append(res_ks_.pvalue)
				#print((np.mean(p_),np.var(p_)))
				res_di = {"statistic": {"KSstat": res_ks.statistic, "Emp": "--"}, "pvalue": {"KSstat": res_ks.pvalue, "Emp": np.mean(p_)}}
				print("The null hypothesis is that the proportions of importance of each feature are similar in the true and predicted model")
				print(pd.DataFrame(res_di))
				print("")
				
				results_iter.update({'Kstat': res_ks.statistic, "Kpval": res_ks.pvalue, "pEmp": np.mean(p_)})

				########### 5. Plot the Principal Component Analysis
				from sklearn.decomposition import PCA
				from sklearn.preprocessing import StandardScaler

				pca_true = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(feature_embeddings))
				pca_pred = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(embs))
				if (plot_it):
					for txt in range(1,embs.shape[0]+1):
						if (txt==1):
							plt.scatter(pca_true[txt-1,0], pca_true[txt-1,1], s=100, c="b", marker=f'${txt}$', label="true", alpha=0.2)
							plt.scatter(pca_pred[txt-1,0], pca_pred[txt-1,1], s=100, c="r", marker=f'${txt}$', label="pred", alpha=0.2)
						else:
							plt.scatter(pca_true[txt-1,0], pca_true[txt-1,1], s=100, c="b", marker=f'${txt}$', alpha=0.2)
							plt.scatter(pca_pred[txt-1,0], pca_pred[txt-1,1], s=100, c="r", marker=f'${txt}$', alpha=0.2)
					plt.legend()
					plt.savefig("PCA_features_%s_%d_%d.png" % (dataset_type, data_seed, iter_seed), bbox_inches="tight")
					plt.close()
				PCAdist = np.linalg.norm(pca_true-pca_pred,2)
				print("||PCA_t - PCA_p ||_2 = %f" % PCAdist)
				print("")
				
				results_iter.update({'PCAdist': PCAdist})

				results_iter_all.update({iter_seed: results_iter})
				with open("%s/results_interpretable_%s_%d.pck" % (folder, dataset_type, data_seed), "wb") as f:
					pickle.dump(results_iter_all, f)
				
			results_data.update({data_seed: results_iter_all})
			with open("%s/results_interpretable_%s.pck" % (folder, dataset_type), "wb") as f:
				pickle.dump(results_data, f)
			
		results.update({dataset_type: results_data})
		import pickle
		with open("results_interpretable.pck", "wb") as f:
			pickle.dump(results, f)
