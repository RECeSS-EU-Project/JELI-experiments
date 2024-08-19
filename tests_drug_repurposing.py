#coding:utf-8

import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd

from stanscofi.datasets import Dataset
from stanscofi.training_testing import random_simple_split
from stanscofi.utils import load_dataset
from stanscofi.validation import AUC, NDCGk, compute_metrics
from benchscofi.utils import rowwise_metrics
from jeli.JELI import JELI
import benchscofi
import matplotlib.pyplot as plt	

import io
from contextlib import redirect_stdout

## Parameters (instance)
cuda_on=(torch._C._cuda_getDeviceCount()>0)
random_seed = 1234
if (cuda_on):
	folder, kge_folder=["/storage/store3/work/creda/"]*2
else:
	folder, kge_folder = "./results/", "files/"

## Models
baselines = [
	"HAN", "FastaiCollabWrapper", "NIMCGCN"
]
## shared parameters
nepochs = 25
batch_size = 1000
lrate = 1e-3
wd = 0.02
plotit = False

dataset_names = ["Gottlieb","LRSSL","PREDICT_Gottlieb","TRANSCRIPT"] 
ndim_ = 15

## specific parameters
algorithm_params = {
        "HAN": { "k": ndim_, "learning_rate": lrate, "epoch": nepochs, "weight_decay": wd, "seed": random_seed },
        "FastaiCollabWrapper": { "n_iterations": nepochs, "n_factors" : ndim_, "weight_decay" : wd, "learning_rate" : lrate, "random_state": random_seed },
        "NIMCGCN": { "epoch" : nepochs, "alpha" : 0.2, "fg" : 256, "fd" : 256, "k" : ndim_, "display_epoch": nepochs, "random_state": random_seed,  "learning_rate": lrate },
        "JELI": { "epochs": nepochs, "batch_size": batch_size, "lr": lrate, "n_dimensions": ndim_, "cuda_on": cuda_on, "order": 2, "structure": "linear", 
            "sim_thres": 0.75, "use_ratings": False, "random_seed": random_seed, "partial_kge": None, "frozen": False
        },
}

def train_test(train, test, model_template, model_params, train_params={}, type_=1):
	print("\n----------- %s" % model_template)
	if (type_):
		model = eval(model_template)(**model_params)
	else:
		model = eval(model_template)(model_params)
	with redirect_stdout(io.StringIO()):
		model.fit(train, **train_params)
	scores = model.predict_proba(test)
	predictions = model.predict(scores, threshold=0.5)
	JELI().print_scores(scores)
	
	if ((plotit) and (model_template=="JELI")):
	
		print([np.round(model.model["losses"][i]-model.model["losses"][i+1],3) for i in range(len(model.model["losses"])-1)])
		plt.plot(range(len(model.model["losses"])), model.model["losses"], "b-")
		plt.plot([np.argmin(model.model["losses"])]*2, [model.model["losses"][np.argmin(model.model["losses"])]]*2, "r--")
		plt.savefig("training_loss_%s_%d_JELI.png" % (dataset_name, iter_seed), bbox_inches="tight")
		plt.close()

	metrics, _ = compute_metrics(scores, predictions, test, metrics=["AUC", "NDCGk"], k=dataset.nitems, beta=1, verbose=False)
	y_test = (test.folds.toarray()*test.ratings.toarray()).ravel()
	y_test[y_test<1] = 0
	auc, ndcg = AUC(y_test, scores.toarray().ravel(), 1, 1), NDCGk(y_test, scores.toarray().ravel(), test.nitems, 1)
	nsauc = np.mean(rowwise_metrics.calc_auc(scores, test))
	print(metrics)
	print("(global) AUC = %.3f" % auc)
	print("(global) NDCG@%d = %.3f" % (test.nitems, ndcg))
	print("(average) NS-AUC = %.3f" % nsauc)
	return dict(auc=auc, ndcg=ndcg, nsauc=nsauc)
	
if __name__=="__main__":
	np.random.seed(random_seed)
	random.seed(random_seed)
	
	Niter = 100
	
	import pickle
	import os
	if (os.path.exists("%s/results_drug_repurposing.pck" % folder)):
		with open("%s/results_drug_repurposing.pck" % folder, "rb") as f:
			results = pickle.load(f)	
	else:
		results = {}
	
	for idd, dataset_name in enumerate(dataset_names):
		print("* Dataset %s (%d/%d)" % (dataset_name, idd+1, len(dataset_names)))
	
		if (dataset_name in results):
			continue

		data_args = load_dataset(dataset_name, save_folder="%s/datasets/" % folder)
		data_args["name"] = dataset_name
		dataset = Dataset(**data_args)
		#dataset.visualize()
			
		if (os.path.exists("%s/results_drug_repurposing-%s.pck" % (folder,dataset_name))):
			with open("%s/results_drug_repurposing-%s.pck" % (folder,dataset_name), "rb") as f:
				results_iter = pickle.load(f)	
		else:
			results_iter = {}
			
		iter_seeds = np.random.choice(range(int(1e8)), size=Niter, replace=False)
		
		for iss, iter_seed in enumerate(iter_seeds):
			print("** Seed %d (%d/%d) -- Dataset %s (%d/%d)" % (iter_seed, iss+1, len(iter_seeds), dataset_name, idd+1, len(dataset_names)))
		
			if (iter_seed in results_iter):
				continue
		
			results_algo = {}

			(train_folds, test_folds), _ = random_simple_split(dataset, 0.2, random_state=iter_seed)
			train = dataset.subset(train_folds)
			test = dataset.subset(test_folds)

			print("*** Algorithm JELI -- Seed %d (%d/%d) -- Dataset %s (%d/%d)" % (iter_seed, iss+1, len(iter_seeds), dataset_name, idd+1, len(dataset_names)))			
			res_metrics = train_test(train, test, "JELI", algorithm_params["JELI"], type_=0)
					
			results_algo.update({"JELI": res_metrics})
			for imm, model_name in enumerate(baselines):
				print("*** Baseline %s (%d/%d) -- Seed %d (%d/%d) -- Dataset %s (%d/%d)" % (model_name, imm+1, len(baselines), iter_seed, iss+1, len(iter_seeds), dataset_name, idd+1, len(dataset_names)))	
				__import__("benchscofi."+model_name)
				res_metrics = train_test(train, test, "benchscofi."+model_name+"."+model_name, algorithm_params[model_name], type_=0)
				results_algo.update({model_name: res_metrics})
				
			print("\n---------- %s (seed=%d)" % (dataset_name, iter_seed))
			print(pd.DataFrame(results_algo))

			results_iter.update({iter_seed: results_algo})
			with open("%s/results_drug_repurposing-%s.pck" % (folder,dataset_name), "wb") as f:
				pickle.dump(results_iter, f)
			
		results.update({dataset_name: results_iter})
		with open("%s/results_drug_repurposing.pck" % folder, "wb") as f:
			pickle.dump(results, f)
