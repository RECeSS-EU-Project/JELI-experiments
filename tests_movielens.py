#coding:utf-8

import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
from subprocess import Popen

from stanscofi.datasets import Dataset
from stanscofi.training_testing import random_simple_split
from stanscofi.utils import load_dataset
from stanscofi.validation import AUC, NDCGk, compute_metrics
from benchscofi.utils import rowwise_metrics
from jeli.JELI import JELI
import benchscofi
import matplotlib.pyplot as plt	

import os
import pickle
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
ndim_ = 50

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
	
	if (os.path.exists("%s/results_movielens.pck" % folder)):
		with open("%s/results_movielens.pck" % folder, "rb") as f:
			results = pickle.load(f)	
	else:
		results = {}
		
	dataset_names = ["MovieLens"]
	idd, dataset_name = 0, dataset_names[0]
	
	print("* Dataset %s (%d/%d)" % (dataset_name, idd+1, len(dataset_names)))

	if (dataset_name in results):
		exit()

	## Create the MovieLens data set
	if (not os.path.exists(f"{kge_folder}ml-latest-small/")):
		proc = Popen("wget -qO - https://files.grouplens.org/datasets/movielens/ml-latest-small.zip |  bsdtar -xvf -".split(" "))
		proc.wait()
		proc = Popen(f"mv ml-latest-small/ {kge_folder}".split(" "))
		proc.wait()
	## Movie feature matrix
	items = pd.read_csv(f"{kge_folder}ml-latest-small/movies.csv", sep=",", index_col=0)
	all_genres = list(set([y for x in items["genres"] for y in x.split("|")]))
	items["Year"] = [x.split(")")[len(x.split(")"))-2**int(len(x.split(")"))>1)].split("(")[-1].split("–")[-1] if (len(x.split("("))>1) else "0" for x in items["title"]]
	for genre in all_genres:
		items[genre] = [int(genre in x) for x in items["genres"]]
	items = items[["Year"]+all_genres].T
	#print((items, np.mean(items.values!=0)))
	## Second example (sparsier) with semantic embeddings
	if (False):
		from sklearn.feature_extraction.text import TfidfVectorizer
		items = pd.read_csv(f"{kge_folder}ml-latest-small/movies.csv", sep=",", index_col=0)
		corpus = [items.loc[idx]["title"]+" "+" ".join(items.loc[idx]["genres"].split("|")) for idx in items.index]
		max_val, max_len = 0, 1
		try:
			for i in range(1,10):
				vectorizer = TfidfVectorizer(analyzer="word",stop_words="english",token_pattern=r"(?u)\b"+r"\w"*i+r"+\b")
				items_mat = vectorizer.fit_transform(corpus).toarray().T
				sparsity = np.sum(items_mat!=0)
				if (max_val < sparsity):
					max_val = val
					max_len = i
		except:
			pass
		vectorizer = TfidfVectorizer(analyzer="word",stop_words="english",token_pattern=r"(?u)\b"+r"\w"*max_len+r"+\b")
		items_mat = vectorizer.fit_transform(corpus).toarray().T
		items = pd.DataFrame(items_mat, columns=items.index, index=vectorizer.get_feature_names_out())
		print((items, np.mean(items.values!=0)))
	items = items.astype(float)
	items.index = items.index.astype(str)
	items.columns = items.columns.astype(str)
	users = pd.read_csv(f"{kge_folder}ml-latest-small/tags.csv", sep=",")
	users["count"] = 1
	users = pd.pivot_table(users, columns=["userId"], values=["count"], index=["tag"], aggfunc="sum", fill_value=0)
	#users.reset_index(level=[0,0])
	users = users.astype(float)
	users.index = users.index.astype(str)
	users.columns = users.columns.get_level_values(1).astype(str)
	ratings = pd.read_csv(f"{kge_folder}ml-latest-small/ratings.csv", sep=",")
	ratings = pd.pivot_table(ratings, columns=["userId"], values=["rating"], index=["movieId"], aggfunc="mean", fill_value=0)
	ratings = ratings.astype(float)
	ratings.index = ratings.index.astype(str)
	ratings.columns = ratings.columns.get_level_values(1).astype(str)
	col_idx, row_idx = [x for x in list(ratings.columns) if (x in users.columns)], [x for x in list(ratings.columns) if (x in items.columns)]
	users = users[col_idx]
	items = items[row_idx]
	ratings = ratings.loc[row_idx][col_idx]
	threshold = int(np.max(ratings.values)/2)+1
	ratings[(ratings!=0)&(ratings<threshold)] = -1
	ratings[(ratings!=0)&(ratings>=threshold)] = 1
	data_args = {"ratings": ratings, "users": users, "items": items, "name": "MovieLens"}
	dataset = Dataset(**data_args)
	#dataset.visualize()
	dataset.summary()
		
	if (os.path.exists("%s/results_movielens-%s.pck" % (folder,dataset_name))):
		with open("%s/results_movielens-%s.pck" % (folder,dataset_name), "rb") as f:
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

		for imm, model_name in enumerate(baselines):
			print("*** Baseline %s (%d/%d) -- Seed %d (%d/%d) -- Dataset %s (%d/%d)" % (model_name, imm+1, len(baselines), iter_seed, iss+1, len(iter_seeds), dataset_name, idd+1, len(dataset_names)))	
			__import__("benchscofi."+model_name)
			res_metrics = train_test(train, test, "benchscofi."+model_name+"."+model_name, algorithm_params[model_name], type_=0)
			results_algo.update({model_name: res_metrics})
			
		print("*** Algorithm JELI -- Seed %d (%d/%d) -- Dataset %s (%d/%d)" % (iter_seed, iss+1, len(iter_seeds), dataset_name, idd+1, len(dataset_names)))			
		res_metrics = train_test(train, test, "JELI", algorithm_params["JELI"], type_=0)		
		results_algo.update({"JELI": res_metrics})
			
		print("\n---------- %s (seed=%d)" % (dataset_name, iter_seed))
		print(pd.DataFrame(results_algo))

		results_iter.update({iter_seed: results_algo})
		with open("%s/results_movielens-%s.pck" % (folder,dataset_name), "wb") as f:
			pickle.dump(results_iter, f)
		
	results.update({dataset_name: results_iter})
	with open("%s/results_movielens.pck" % folder, "wb") as f:
		pickle.dump(results, f)
