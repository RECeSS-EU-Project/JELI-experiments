#coding:utf-8

import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
import requests

from stanscofi.datasets import Dataset
from stanscofi.training_testing import random_simple_split
from stanscofi.utils import load_dataset
from stanscofi.validation import AUC, NDCGk, compute_metrics
from benchscofi.utils import rowwise_metrics
from jeli.JELI import JELI

import os
import io
from contextlib import redirect_stdout
from subprocess import Popen

## Parameters (instance)
cuda_on=(torch._C._cuda_getDeviceCount()>0)
random_seed = 1234
folder, kge_folder = "./", "files/"

## Models
## shared parameters
nepochs = 25
batch_size = 1000
lrate = 1e-3
wd = 0.02

ndim_ = 50 # 15

## specific parameters
JELI_params = { "epochs": nepochs, "batch_size": batch_size, "lr": lrate, "n_dimensions": ndim_, "cuda_on": cuda_on, "order": 2, "structure": "linear", 
            "sim_thres": 0.75, "use_ratings": False, "random_seed": random_seed, "partial_kge": None, "frozen": True
        }

def train_test(train, test, model_template, model_params, train_params={}, type_=1):
	print("\n----------- %s" % model_template)
	if (type_):
		model = eval(model_template)(**model_params)
	else:
		model = eval(model_template)(model_params)
	try:
		with redirect_stdout(io.StringIO()):
			model.fit(train, **train_params)
	except:
		return {}
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
	feature_embeddings = model.model.get("feature_embeddings")
	return dict(auc=auc, ndcg=ndcg, nsauc=nsauc, feature_embeddings=feature_embeddings)
	
if __name__=="__main__":
	np.random.seed(random_seed)
	random.seed(random_seed)
	
	dataset_name = "TRANSCRIPT"
	Niter = 10
	kg_types = ["PrimeKG", "Hetionet", "DRKG", "PharmKG8k", "PharmKG"] 
	## https://pykeen.readthedocs.io/en/stable/reference/datasets.html
	import pykeen, pykeen.datasets

	data_args = load_dataset(dataset_name, save_folder=folder+"/datasets/")
	data_args["name"] = dataset_name
	if (dataset_name in ["TRANSCRIPT"]):
		data_args["same_item_user_features"] = True
	dataset = Dataset(**data_args)
	assert dataset.same_item_user_features
	#dataset.visualize()
	
	iter_seeds = np.random.choice(range(int(1e8)), size=Niter, replace=False)
	
	import pickle
	import os
	if (os.path.exists("%s/results_add_prior2.pck" % folder)):
		with open("%s/results_add_prior2.pck" % folder, "rb") as f:
			results = pickle.load(f)	
	else:
		results = {}
	
	for iss, iter_seed in enumerate(iter_seeds):
		print("* Seed %d (%d/%d) -- Dataset TRANSCRIPT" % (iter_seed, iss+1, len(iter_seeds)))
	
		if (iter_seed in results): ##
			continue            ##

		(train_folds, test_folds), _ = random_simple_split(dataset, 0.2, random_state=iter_seed)
		train = dataset.subset(train_folds)
		test = dataset.subset(test_folds)
		
		if (os.path.exists("%s/results_add_prior_%d.pck" % (folder,iter_seed))):
			with open("%s/results_add_prior_%d.pck" % (folder,iter_seed), "rb") as f:
				results_prior = pickle.load(f)	
		else:
			results_prior = {}

		for ikk, kg_type in enumerate(kg_types):
			print("** Seed %d (%d/%d) -- Dataset TRANSCRIPT with prior %s (%d/%d)" % (iter_seed, iss+1, len(iter_seeds), kg_type, ikk+1, len(kg_types)))
		
			if (kg_type in results_prior): ##
				continue                ##
		
			########### No additional KG
			if (kg_type == "None"):
				partial_kge = None
				kge_name = None
				
			########### KG from STRING	
			elif (kg_type == "STRING"):
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
			
			########### KG from PyKEEN
			else:
				## refer to https://github.com/RECeSS-EU-Project/drug-repurposing-datasets/blob/master/notebooks/PREDICT_dataset.ipynb
				with open(kge_folder+"drugbankid2drugname.pck", "rb") as f:
					drugbankid2drugname = pickle.load(f)
					drugname2drugbankid = {v.lower(): k for k,v in drugbankid2drugname.items()}
				with open(kge_folder+"pubchemid2drugname.pck", "rb") as f:
					pubchemid2drugname = pickle.load(f)
					drugname2pubchemid = {v.lower(): k for k,v in pubchemid2drugname.items()}
				with open(kge_folder+"medgenid2diseasename.pck", "rb") as f:
					medgenid2diseasename = pickle.load(f)
					diseasename2medgenid = {v.lower(): k for k,v in medgenid2diseasename.items()}
				with open(kge_folder+"omimid2diseasename.pck", "rb") as f:
					omimid2diseasename = pickle.load(f)
					diseasename2omimid = {v.lower(): k for k,v in omimid2diseasename.items()}
				def concept2mesh(diseases):
					convert_file_folder = "%s/disease_phenotype_MeshTERMS.out" % kge_folder
					CID_MESH_df = pd.read_csv(convert_file_folder, sep=", ", index_col=0, header=None, engine='python')
					CID_MESH_df.index = [d.split("'")[1] for d in CID_MESH_df.index]
					CID_MESH_df[CID_MESH_df.columns[0]] = [np.nan if ("'" not in d) else d.split("'")[1] for d in CID_MESH_df[CID_MESH_df.columns[0]]]
					return list(CID_MESH_df.loc[diseases][CID_MESH_df.columns[0]])
				def concept2hp(diseases):
					converted_diseases = {}
					for di in diseases:
						url = f"https://www.ncbi.nlm.nih.gov/medgen/{di}"
						urlData = requests.get(url).content.decode('utf-8')
						if ("https://hpo.jax.org/app/browse/term/HP:" in urlData):
							hp_id = "HP:"+(urlData.split("https://hpo.jax.org/app/browse/term/HP:")[-1].split("\">")[0])
							converted_diseases.update({hp_id: di})
					return converted_diseases
				## https://www.genenames.org/download/custom/ NCBI ID -> HUGO symbols
				def NCBI2HUGO_f():
					url="https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&col=gd_pub_eg_id&status=Approved&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit"
					urlData = requests.get(url).content
					NCBI2HUGO = pd.read_csv(io.StringIO(urlData.decode('utf-8')), sep="\t").dropna()
					NCBI2HUGO.index = NCBI2HUGO["Approved symbol"].astype(str)
					NCBI2HUGO = NCBI2HUGO.loc[[i for i in dataset.item_features if (i in NCBI2HUGO.index)]]
					NCBI2HUGO.index = NCBI2HUGO["NCBI Gene ID"].astype(int).astype(str)
					NCBI2HUGO = NCBI2HUGO["Approved symbol"].to_dict()
					return NCBI2HUGO
					
				dt = eval(f"pykeen.datasets.{kg_type}")(cache_root=folder)
				kge_name = kge_folder+f"kge_{kg_type}"
				partial_kge = torch.cat([dt.training.mapped_triples, dt.testing.mapped_triples, dt.validation.mapped_triples], dim=0).numpy().astype(str)
				assert partial_kge.shape[1] == 3
				inv_map = {str(dt.entity_to_id[k]):k for k in dt.entity_to_id}
				partial_kge[:,0] = [inv_map[k] for k in partial_kge[:,0]]
				partial_kge[:,2] = [inv_map[k] for k in partial_kge[:,2]]
				inv_map = {str(dt.relation_to_id[k]):k for k in dt.relation_to_id}
				partial_kge[:,1] = [inv_map[k] for k in partial_kge[:,1]]
				
				if (kg_type in ["PrimeKG", "PharmKG8k", "PharmKG"]):
					kg2dataset_genes = { g: g for g in dataset.item_features if (g in dt.entity_to_id)}
					kg2dataset_drugs = { dr: drugname2drugbankid.get(dr.lower(), drugname2pubchemid.get(dr.lower())) for dr in dt.entity_to_id }
					kg2dataset_drugs = { dr: did for dr, did in kg2dataset_drugs.items() if (did in dataset.item_list)}
					kg2dataset_diseases = { di: diseasename2omimid.get(di.lower(), diseasename2medgenid.get(di.lower())) for di in dt.entity_to_id }
					kg2dataset_diseases = { di: did for di, did in kg2dataset_diseases.items() if (did in dataset.user_list)}
			
				elif (kg_type == "Hetionet"):
					kg2dataset_genes = { g.split("Gene::")[-1]: g.split("Gene::")[-1] for g in dt.entity_to_id if (("Gene::" in g) and (g.split("Gene::")[-1] in dataset.item_features)) }
					kg2dataset_drugs = { idd: idd.split("::")[1] for idd in dt.entity_to_id if (("Compound::" in idd) and (idd.split("::")[1] in dataset.item_list)) }
					adis = concept2mesh(dataset.user_list)
					kg2dataset_diseases = { idd: dataset.user_list[adis.index(idd.split("::")[1])] for idd in dt.entity_to_id if (("Symptom::" in idd) and (idd.split("::")[1] in adis)) }
				
				elif (kg_type == "DRKG"):
					adis = concept2mesh(dataset.user_list)
					NCBI2HUGO = NCBI2HUGO_f()
					kg2dataset_diseases = {g: g.split("Disease::MESH:")[-1] for g in dt.entity_to_id if (("Disease::MESH:" in g) and (g.split("Disease::MESH:")[-1] in adis))}							
					kg2dataset_genes = {g: NCBI2HUGO[g.split("Gene::")[-1]] for g in dt.entity_to_id if (("Gene::" in g) and (g.split("Gene::")[-1] in NCBI2HUGO))}
					kg2dataset_drugs = {g: g.split("Compound::")[-1] for g in dt.entity_to_id if (("Compound::" in g) and (g.split("Compound::")[-1] in dataset.item_list))}
					
				else:
					raise ValueError(f"KG type {kg_type} does not exist.")
			
			ent2id = dataset.user_list+dataset.item_list+dataset.item_features
			for i in [0,2]:
				partial_kge = partial_kge[np.vectorize(lambda x : kg2dataset_drugs.get(x, kg2dataset_diseases.get(x, kg2dataset_genes.get(x, x))) in ent2id)(partial_kge[:,i])]
			print(f"Number of additional edges: {partial_kge.shape[0]} featuring {len(kg2dataset_drugs)} items, {len(kg2dataset_diseases)} users and {len(kg2dataset_genes)} features")
			JELI_params.update({"kge_name": kge_name, "partial_kge": partial_kge})
			res = train_test(train, test, "JELI", JELI_params, type_=0)
			results_prior.update({kg_type: res})
			with open("%s/results_add_prior2_%d.pck" % (folder, iter_seed), "wb") as f: ##
				pickle.dump(results_prior, f)
			
		print(pd.DataFrame(results_prior))
			
		results.update({iter_seed: results_prior})		
	
	with open("%s/results_add_prior2.pck" % folder, "wb") as f: ##
		pickle.dump(results, f)
