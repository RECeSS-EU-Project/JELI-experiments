#coding:utf-8

import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd

from data import generate_deviated_dataset
from stanscofi.datasets import Dataset, generate_dummy_dataset
from stanscofi.training_testing import random_simple_split
from stanscofi.validation import AUC, NDCGk, compute_metrics
from benchscofi.utils import rowwise_metrics
from jeli.JELI import JELI
from jeli.JELIImplementation import RHOFM
import benchscofi

import sys
sys.path.insert(0,"./baselines/")
from SELT_pca_f import SELT_pca_f
from SELT_pca_iu import SELT_pca_iu
from SELT_kge import SELT_kge
from FM_baselines import FM2, CrossFM2

import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout

## Parameters (instance)
cuda_on=(torch._C._cuda_getDeviceCount()>0)
random_seed = 1234
folder, kge_folder = "./results/", "files/"
dataset_params = dict(npoints=int(np.sqrt(29929)), nfeatures=10, ndim=2, sparsity_features=0.8, dvar=0.01, dmean=2)

## Models
models = [
	"HAN", "FastaiCollabWrapper", "NIMCGCN", 
	"SELT_pca_f", "SELT_pca_iu", "SELT_kge", 
	"FM2", "CrossFM2", 
	"JELI"
]
## shared parameters
nepochs = 60 #25
ndim_ = dataset_params["ndim"]
batch_size = 1000
lrate = 1e-3
wd = 0.02
## specific parameters
algorithm_params = {
        "HAN": { "k": ndim_, "learning_rate": lrate, "epoch": nepochs, "weight_decay": wd, "seed": random_seed },
        "FastaiCollabWrapper": { "n_iterations": nepochs, "n_factors" : ndim_, "weight_decay" : wd, "learning_rate" : lrate, "random_state": random_seed },
        "NIMCGCN": { "epoch" : nepochs, "alpha" : 0.2, "fg" : 256, "fd" : 256, "k" : ndim_, "display_epoch": nepochs, "random_state": random_seed,  "learning_rate": lrate },
        "JELI": { "epochs": nepochs, "batch_size": batch_size, "lr": lrate, "n_dimensions": ndim_, "cuda_on": cuda_on, "order": 2, "structure": "linear", 
            "sim_thres": 0.75, "use_ratings": False, "random_seed": random_seed, "partial_kge": None, "frozen": False
        },
        "SELT": { "n_dimensions": ndim_, "random_state": random_seed,
            "fm_params": {"batch_size": batch_size, "n_epochs": nepochs//2+int(nepochs%2!=0), "loss": "MarginRankingLoss",
            "opt_params":{'lr': lrate, "weight_decay": wd}, "early_stop":0, "random_seed":random_seed},
            "kge_params" : {"n_epochs": nepochs//2, "opt_params":{'lr': lrate, "weight_decay": wd}, "batch_size": batch_size, "loss": "CrossEntropyLoss", "interaction": {"MuRE": {"p": 1}}}
        },     
        "FM2": { "d": ndim_, "itemF": dataset_params["nfeatures"], "userF": dataset_params["nfeatures"], "cuda_on": cuda_on, "random_state": random_seed},
        "CrossFM2" : {"d": ndim_, "F": dataset_params["nfeatures"], "cuda_on": cuda_on, "random_state": random_seed},
        "RHOFM": {"d": ndim_, "order": 2, "structure": "linear", "frozen": False, "cuda_on": cuda_on, "random_state": random_seed}, 
}
FM_train_params = {"n_epochs": nepochs, "loss": "MarginRankingLoss", "batch_size": batch_size, "random_seed": random_seed}

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
	
if __name__ == "__main__":
	
	np.random.seed(random_seed)
	random.seed(random_seed)
	
	Ndata=5      ## iterate over several randomly generated instances
	Niter=100    ## iterate over several random seeds for the same instance
	sparsities = [i*0.01 for i in range(50, 85, 15)]+[0.99]
	dataset_types = ["synthetic", "deviated"]
	
	import pickle
	import os
	if (os.path.exists("%s/results_sparsity.pck" % folder)):
		with open("%s/results_sparsity.pck" % folder, "rb") as f:
			results = pickle.load(f)	
	else:
		results = {}
	
	for idd, dataset_type in enumerate(dataset_types):
		print("* Dataset type %s (%d/%d)" % (dataset_type, idd+1, len(dataset_types)))
		if (dataset_type in results):
			continue
	
		data_seeds = np.random.choice(range(int(1e8)), size=Ndata, replace=False)
		if (os.path.exists("%s/results_sparsity_%s.pck" % (folder, dataset_type))):
			with open("%s/results_sparsity_%s.pck" % (folder, dataset_type), "rb") as f:
				results_data = pickle.load(f)	
		else:
			results_data = {}
		
		for iss, data_seed in enumerate(data_seeds):
			print("** Data seed %d (%d/%d) -- Dataset type %s (%d/%d)" % (data_seed, iss+1, len(data_seeds), dataset_type, idd+1, len(dataset_types)))
			
			if (data_seed in results_data):
				continue
			dataset_params.update({"data_seed": data_seed})
			
			if (os.path.exists("%s/results_sparsity_%s_%d.pck" % (folder, dataset_type, data_seed))):
				with open("%s/results_sparsity_%s_%d.pck" % (folder, dataset_type, data_seed), "rb") as f:
					results_sparsity = pickle.load(f)	
			else:
				results_sparsity = {}
			
			for isp, sparsity in enumerate(sparsities):
				print("*** Sparsity %.2f (%d/%d) -- Data seed %d (%d/%d) -- Dataset type %s (%d/%d)" % (sparsity, isp+1, len(sparsities), data_seed, iss+1, len(data_seeds), dataset_type, idd+1, len(dataset_types)))
				if (sparsity in results_sparsity):
					continue
				dataset_params.update({"sparsity": sparsity})
				iter_seeds = np.random.choice(range(int(1e8)), size=Niter, replace=False)
				
				if (os.path.exists("%s/results_sparsity_%s_%d_%d.pck" % (folder, dataset_type, data_seed, sparsity*100))):
					with open("%s/results_sparsity_%s_%d_%d.pck" % (folder, dataset_type, data_seed, sparsity*100), "rb") as f:
						results_sparsity_iter = pickle.load(f)	
				else:
					results_sparsity_iter = {}
				
				for its, iter_seed in enumerate(iter_seeds):
					print("**** Iter seed %d (%d/%d) -- Sparsity %.2f (%d/%d) -- Data seed %d (%d/%d) -- Dataset type %s (%d/%d)" % (iter_seed, its+1, len(iter_seeds), sparsity, isp+1, len(sparsities), data_seed, iss+1, len(data_seeds), dataset_type, idd+1, len(dataset_types)))
					if (iter_seed in results_sparsity_iter):
						continue
				
					if (os.path.exists("%s/results_sparsity_%s_%d_%d_%d.pck" % (folder, dataset_type, data_seed, sparsity*100, iter_seed))):
						with open("%s/results_sparsity_%s_%d_%d_%d.pck" % (folder, dataset_type, data_seed, sparsity*100, iter_seed), "rb") as f:
							results_algo = pickle.load(f)	
					else:
						results_algo = {}

					## Synthetic
					if (dataset_type == "synthetic"):
						npositive, nnegative, nfeatures, mean, std = dataset_params["npoints"]//2, dataset_params["npoints"]//2+int(dataset_params["npoints"]%2!=0), 2*dataset_params["nfeatures"], dataset_params["dmean"], dataset_params["dvar"]
						data_args = generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std, random_state=dataset_params["data_seed"])
					## Interpretable
					else:
						data_args, _, _, _ = generate_deviated_dataset(int(dataset_params["npoints"]**2), dataset_params["nfeatures"], dataset_params["ndim"],  dataset_params["sparsity"], sparsity_features=dataset_params["sparsity_features"], binary=False, seed=dataset_params["data_seed"], var=dataset_params["dvar"], cuda_on=cuda_on)
					
					dataset = Dataset(**data_args)
					#dataset.summary()
					(train_folds, test_folds), _ = random_simple_split(dataset, 0.2, random_state=dataset_params["data_seed"])
					train = dataset.subset(train_folds)
					test = dataset.subset(test_folds)
					
					for imm, model_name in enumerate(models):
						print("***** Algorithm %s (%d/%d) -- Iter seed %d (%d/%d) -- Sparsity %.2f (%d/%d) -- Data seed %d (%d/%d) -- Dataset type %s (%d/%d)" % (model_name, imm+1, len(models), iter_seed, its+1, len(iter_seeds), sparsity, isp+1, len(sparsities), data_seed, iss+1, len(data_seeds), dataset_type, idd+1, len(dataset_types)))
						if (model_name in results_algo):
							continue

						if ("FM" in model_name):
							res_metrics = train_test(train, test, model_name, algorithm_params[model_name], train_params=FM_train_params)
						elif ("SELT" in model_name):
							res_metrics = train_test(train, test, model_name, algorithm_params["SELT"], type_=0)
						elif ("JELI"==model_name):
							res_metrics = train_test(train, test, "JELI", algorithm_params["JELI"], type_=0)
						else:
							__import__("benchscofi."+model_name)
							res_metrics = train_test(train, test, "benchscofi."+model_name+"."+model_name, algorithm_params[model_name], type_=0)
							
						results_algo.update({model_name: res_metrics})
						with open("%s/results_sparsity_%s_%d_%d_%d.pck" % (folder, dataset_type, data_seed, sparsity*100, iter_seed), "wb") as f:
							pickle.dump(results_algo, f)
		
					results_sparsity_iter.update({iter_seed: results_algo})
					with open("%s/results_sparsity_%s_%d_%d.pck" % (folder, dataset_type, data_seed, sparsity*100), "wb") as f:
						pickle.dump(results_sparsity_iter, f)
				results_sparsity.update({dataset_params["sparsity"]: results_sparsity_iter })
				with open("%s/results_sparsity_%s_%d.pck" % (folder, dataset_type, data_seed), "wb") as f:
					pickle.dump(results_sparsity, f)
			results_data.update({dataset_params["data_seed"]: results_sparsity})
			with open("%s/results_sparsity_%s.pck" % (folder, dataset_type), "wb") as f:
				pickle.dump(results_data, f)
		results.update({dataset_type: results_data})
		with open("%s/results_sparsity.pck" % folder, "wb") as f:
			pickle.dump(results, f)
