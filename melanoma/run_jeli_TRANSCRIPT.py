#coding:utf-8

from stanscofi.datasets import Dataset
from stanscofi.utils import load_dataset
from jeli.JELI import JELI

import pickle
import os
import io
import torch
import numpy as np
import random
import pandas as pd
from contextlib import redirect_stdout
from subprocess import Popen

## Parameters (instance)
cuda_on=(torch._C._cuda_getDeviceCount()>0)
random_seed = 1234
folder, kge_folder = "./", "files/"

np.random.seed(random_seed)
random.seed(random_seed)

## Parameters (model)
nepochs = 25
batch_size = 1000
lrate = 1e-3
wd = 0.02
ndim_ = 50 # 15
JELI_params = { "epochs": nepochs, "batch_size": batch_size, "lr": lrate, "n_dimensions": ndim_, "cuda_on": cuda_on, "order": 2, "structure": "linear", 
            "sim_thres": 0.75, "use_ratings": False, "random_seed": random_seed, "partial_kge": None, "frozen": True
}

## Data set
dataset_name = "TRANSCRIPT"
data_args = load_dataset(dataset_name, save_folder=folder+"/datasets/")
data_args["name"] = dataset_name
data_args["same_item_user_features"] = True
dataset = Dataset(**data_args)

## KG from STRING
proc = Popen(("mkdir -p %s/" % kge_folder).split(" "))
proc.wait()
TRANSCRIPT_PATH = "%s/REGULATORY_network_t=%1.f.tsv" % (kge_folder,0.5)
PROT_PATH = TRANSCRIPT_PATH.split(".tsv")[0]+"_prot.tsv"
assert len(dataset.user_features)==len(dataset.item_features)
gene_list = dataset.item_features
prot_list = pd.read_csv(PROT_PATH, sep=",")
prot_list.index = prot_list["stringId"]
network = pd.read_csv(kge_folder+"STRING network (physical-proteins).sif", sep="\t", header=None).astype(str)
network.columns = ["Head", "Relation", "Tail"]
genes_di = prot_list[["queryItem"]].to_dict()["queryItem"]
partial_kge = pd.DataFrame([],index=range(network.shape[0]))
partial_kge["Head"] = [genes_di.get(x,np.nan) for x in network["Head"]]
partial_kge["Relation"] = "ppp"
partial_kge["Tail"] = [genes_di.get(x,np.nan) for x in network["Tail"]]
partial_kge = partial_kge.astype(str).values
kge_name = kge_folder+"kge_STRING"
JELI_params.update({"kge_name": kge_name, "partial_kge": partial_kge})

JELI_model = JELI(JELI_params)
try:
	with redirect_stdout(io.StringIO()):
		JELI_model.fit(dataset)
		scores = JELI_model.predict_proba(dataset)
	feature_embeddings = JELI_model.model.get("feature_embeddings")
	results = dict(dataset_name=dataset_name, prior=kge_name, scores=scores, feature_embeddings=feature_embeddings)
except:
	print("Failed")
	results = {}

with open("%s/results_JELI_%s.pck" % (folder, dataset_name), "wb") as f: ##
	pickle.dump(results, f)
