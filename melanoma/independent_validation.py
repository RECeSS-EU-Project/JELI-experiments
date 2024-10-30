#coding:utf-8

from stanscofi.datasets import Dataset
from stanscofi.utils import load_dataset
from stanscofi.validation import compute_metrics, AUC, NDCGk
from stanscofi.models import LogisticRegression as LR
from benchscofi.utils.rowwise_metrics import calc_auc
from benchscofi.Constant import Constant
from subprocess import call
import numpy as np
import pickle
import pandas as pd
import os

folder="./"
dataset_name = ["TRANSCRIPT", "PREDICT"][0]
diseases = pd.read_csv("diseases.csv")
threshold = 0.92

## Data set
data_args = load_dataset(dataset_name, save_folder="../datasets/")
data_args["name"] = dataset_name
data_args["same_item_user_features"] = dataset_name == "TRANSCRIPT"
dataset = Dataset(**data_args)

dataset.summary()

with open("%s/results_JELI_%s.pck" % (folder, dataset_name), "rb") as f: ##
	results = pickle.load(f)
scores = results["scores"]
feature_embeddings = results["feature_embeddings"]

predictions = Constant().predict(scores, threshold=0.5)
Constant().print_scores(scores)
metrics, _ = compute_metrics(scores, predictions, dataset, metrics=["AUC", "NDCGk"], k=dataset.nitems, beta=1, verbose=False)
y_test = (dataset.folds.toarray()*dataset.ratings.toarray()).ravel()
y_test[y_test<1] = 0
auc, ndcg = AUC(y_test, scores.toarray().ravel(), 1, 1), NDCGk(y_test, scores.toarray().ravel(), dataset.nitems, 1)
nsauc = np.mean(calc_auc(scores, dataset))
metrics.loc["global AUC"] = [auc, 0]
metrics.loc["global NDCGk"] = [ndcg, 0]
metrics.loc["avg. NS-AUC"] = [nsauc, 0]
print(metrics)
print("")

with open("../files/pubchemid2drugname.pck", "rb") as f:
	pubchemid2drugname = pickle.load(f)

with open("../files/drugbankid2drugname.pck", "rb") as f:
	drugbankid2drugname = pickle.load(f)

for i in diseases.index:
	print((diseases.loc[i]["Name"], diseases.loc[i]["ConceptID"], "is "+("not"*int(diseases.loc[i]["ConceptID"] not in dataset.user_list))+f"in data set {dataset_name}"))
	disease_idx = dataset.user_list.index(diseases.loc[i]["ConceptID"])
	feature_vector = dataset.users.toarray()[:,disease_idx]
	print(np.mean(feature_vector!=0)*100)
	recs = pd.DataFrame([[x] for x in scores.toarray()[:,disease_idx].flatten()], index=dataset.item_list, columns=["drug score"])
	recs["drug annotation"] = [dataset.ratings.toarray()[dataset.item_list.index(j),disease_idx] for j in recs.index]
	recs["drug name"] = [pubchemid2drugname.get(x, drugbankid2drugname.get(x,x)) for x in recs.index]
	recs = recs.sort_values(by="drug score", ascending=False)
	N = min(10, recs.loc[recs["drug score"]>threshold].shape[0])
	print(recs.loc[recs.index[:N]])
	print("")
	recs = recs.sort_values(by="drug annotation", ascending=False)
	N = min(10, recs.loc[recs["drug annotation"]>=1].shape[0])
	print(recs.loc[recs.index[:N]])
	print("")
	N = min(10, recs.loc[recs["drug annotation"]<=-1].shape[0])
	if (N>0):
		print(recs.loc[recs.index[-N:]])
		print("")
	exit()

