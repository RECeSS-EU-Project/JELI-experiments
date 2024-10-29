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
dataset_params = dict(npoints=int(np.sqrt(29929)), nfeatures=10, ndim=5, sparsity_features=0.8, dvar=0.01, dmean=2, sparsity=0.9)

## Models
models = [
	"JELI"
]
## shared parameters
nepochs = 60 #25
batch_size = 1000
lrate = 1e-3
wd = 0.02
## specific parameters
algorithm_params = {
        "JELI": { "epochs": nepochs, "batch_size": batch_size, "lr": lrate, "cuda_on": cuda_on, "order": 2, "structure": "linear", 
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
	
	Ndata=100      ## iterate over several randomly generated instances
	Niter=1    ## iterate over several random seeds for the same instance
	ndims = range(1,10) #
	dataset_types = ["deviated"]
	
	import pickle
	import os
	if (os.path.exists("%s/results_parameter2.pck" % folder)):
		with open("%s/results_parameter2.pck" % folder, "rb") as f:
			results = pickle.load(f)	
	else:
		results = {}
	
	for idd, dataset_type in enumerate(dataset_types):
		print("* Dataset type %s (%d/%d)" % (dataset_type, idd+1, len(dataset_types)))
		if (dataset_type in results):
			continue
	
		data_seeds = np.random.choice(range(int(1e8)), size=Ndata, replace=False)
		if (os.path.exists("%s/results_parameter2_%s.pck" % (folder, dataset_type))):
			with open("%s/results_parameter2_%s.pck" % (folder, dataset_type), "rb") as f:
				results_data = pickle.load(f)	
		else:
			results_data = {}
		
		for iss, data_seed in enumerate(data_seeds):
			print("** Data seed %d (%d/%d) -- Dataset type %s (%d/%d)" % (data_seed, iss+1, len(data_seeds), dataset_type, idd+1, len(dataset_types)))
			
			if (data_seed in results_data):
				continue
			dataset_params.update({"data_seed": data_seed})
			
			if (os.path.exists("%s/results_parameter2_%s_%d.pck" % (folder, dataset_type, data_seed))):
				with open("%s/results_parameter2_%s_%d.pck" % (folder, dataset_type, data_seed), "rb") as f:
					results_parameter = pickle.load(f)	
			else:
				results_parameter = {}
			
			for isp, ndim in enumerate(ndims):
				print("*** Dimension %d, true %d (%d/%d) -- Data seed %d (%d/%d) -- Dataset type %s (%d/%d)" % (ndim, dataset_params["ndim"], isp+1, len(ndims), data_seed, iss+1, len(data_seeds), dataset_type, idd+1, len(dataset_types)))
				if (ndim in results_parameter):
					continue
				iter_seeds = np.random.choice(range(int(1e8)), size=Niter, replace=False)
				
				if (os.path.exists("%s/results_parameter2_%s_%d_%d.pck" % (folder, dataset_type, data_seed, ndim))):
					with open("%s/results_parameter2_%s_%d_%d.pck" % (folder, dataset_type, data_seed, ndim), "rb") as f:
						results_parameter_iter = pickle.load(f)	
				else:
					results_parameter_iter = {}
				
				for its, iter_seed in enumerate(iter_seeds):
					print("**** Iter seed %d (%d/%d) -- Dimension %d, true %d (%d/%d) -- Data seed %d (%d/%d) -- Dataset type %s (%d/%d)" % (iter_seed, its+1, len(iter_seeds), ndim, dataset_params["ndim"], isp+1, len(ndims), data_seed, iss+1, len(data_seeds), dataset_type, idd+1, len(dataset_types)))
					if (iter_seed in results_parameter_iter):
						continue
				
					if (os.path.exists("%s/results_parameter2_%s_%d_%d_%d.pck" % (folder, dataset_type, data_seed, ndim, iter_seed))):
						with open("%s/results_parameter2_%s_%d_%d_%d.pck" % (folder, dataset_type, data_seed, ndim, iter_seed), "rb") as f:
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
						print("***** Algorithm %s (%d/%d) -- Iter seed %d (%d/%d) -- Dimension %d, true %d (%d/%d) -- Data seed %d (%d/%d) -- Dataset type %s (%d/%d)" % (model_name, imm+1, len(models), iter_seed, its+1, len(iter_seeds), ndim, dataset_params["ndim"], isp+1, len(ndims), data_seed, iss+1, len(data_seeds), dataset_type, idd+1, len(dataset_types)))
						if (model_name in results_algo):
							continue

						if ("JELI"==model_name):
							algorithm_params["JELI"]["n_dimensions"] = ndim
							res_metrics = train_test(train, test, "JELI", algorithm_params["JELI"], type_=0)
						else:
							raise ValueError
							
						results_algo.update({model_name: res_metrics})
						with open("%s/results_parameter2_%s_%d_%d_%d.pck" % (folder, dataset_type, data_seed, ndim, iter_seed), "wb") as f:
							pickle.dump(results_algo, f)
		
					results_parameter_iter.update({iter_seed: results_algo})
					with open("%s/results_parameter2_%s_%d_%d.pck" % (folder, dataset_type, data_seed, ndim), "wb") as f:
						pickle.dump(results_parameter_iter, f)
				results_parameter.update({ndim: results_parameter_iter })
				with open("%s/results_parameter2_%s_%d.pck" % (folder, dataset_type, data_seed), "wb") as f:
					pickle.dump(results_parameter, f)
			results_data.update({dataset_params["data_seed"]: results_parameter})
			with open("%s/results_parameter2_%s.pck" % (folder, dataset_type), "wb") as f:
				pickle.dump(results_data, f)
		results.update({dataset_type: results_data})
		with open("%s/results_parameter2.pck" % folder, "wb") as f:
			pickle.dump(results, f)