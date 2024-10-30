#coding:utf-8

from stanscofi.datasets import Dataset
from stanscofi.utils import load_dataset
from stanscofi.validation import compute_metrics, AUC, NDCGk
from stanscofi.models import LogisticRegression as LR
from benchscofi.utils.rowwise_metrics import calc_auc
from benchscofi.Constant import Constant
import numpy as np
import os
import pickle
import pandas as pd
from subprocess import Popen

folder="./"
score_folder=f"{folder}scores/"
dataset_name = ["TRANSCRIPT", "PREDICT"][0]
threshold=0.92

diseases = pd.read_csv("diseases.csv")

## Data set
data_args = load_dataset(dataset_name, save_folder="../datasets/")
data_args["name"] = dataset_name
data_args["same_item_user_features"] = dataset_name == "TRANSCRIPT"
dataset = Dataset(**data_args)

with open("%s/results_JELI_%s.pck" % (folder, dataset_name), "rb") as f: ##
	results = pickle.load(f)
scores = results["scores"]
feature_embeddings = results["feature_embeddings"].numpy()

with open("../files/pubchemid2drugname.pck", "rb") as f:
	pubchemid2drugname = pickle.load(f)

with open("../files/drugbankid2drugname.pck", "rb") as f:
	drugbankid2drugname = pickle.load(f)
	
with open("../files/medgenid2diseasename.pck", "rb") as f:
	medgenid2diseasename = pickle.load(f)

with open("../files/omimid2diseasename.pck", "rb") as f:
	omimid2diseasename = pickle.load(f)
	
## Disease-specific ranking scores (for the enrichment of functional pathways)
for i in diseases.index:
	print("%s (%s) %s" % (diseases.loc[i]["Name"], diseases.loc[i]["ConceptID"], "is "+("not"*int(diseases.loc[i]["ConceptID"] not in dataset.user_list))+f"in data set {dataset_name}"))
	disease_idx = dataset.user_list.index(diseases.loc[i]["ConceptID"])
	disease_feature_vector = dataset.users.toarray()[:,disease_idx]
	imp_scores_disease = np.multiply(feature_embeddings, np.tile(disease_feature_vector.reshape(-1,1), (1, feature_embeddings.shape[1]))).sum(axis=1)
	imp_scores = pd.DataFrame(imp_scores_disease.T, index=dataset.item_features, columns=["Score"])
	ranks = imp_scores.sort_values(by="Score", ascending=False)
	ranks.to_csv(f"{diseases.loc[i]['Name']}-scores.rnk", header=False, sep="\t")
	print(ranks)
	print("%s #genes %d, %.2f percent" % (diseases.loc[i]["Name"], np.sum(ranks["Score"]!=0), np.mean(ranks["Score"]!=0)*100))

## use file f"{diseases.loc[i]['Name']}-scores.rnk" in WebGestalt.org
