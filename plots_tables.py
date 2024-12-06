#coding:utf-8

import pickle
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from glob import glob
import seaborn as sns
import scipy.stats as stats

plot_it = True
cuda_on=(torch._C._cuda_getDeviceCount()>0)
random_seed = 1234
folder, kge_folder = "./results/", "files/"
fsize=25
runtests = ["interpretable","sparsity","drug_repurposing","add_prior","parameter_impact","gene_enrichment","order_impact","movielens","scalability"]

##################################################
## Interpretability (synthetic datasets)        ##
##################################################
print("------------------- Interpretability (synthetic datasets)")

def result_interpretable(R, dataset_type=""):
	if (R is None):
		return None
	aucs, nsaucs, spearmans, jsdists, pcadists = [], [], [], [], []
	max_spearman = 0
	max_metrics = {"AUC": -1, "NS-AUC": -1, "spearman-rho": -1, 'PCAdist': -1, "jsdist": -1}
	N = 0
	## Different datasets
	for data_seed in R:
		id_max, max_rho = -1, -1
		## Different trainings
		for iter_seed in R[data_seed]:
			res = R[data_seed][iter_seed]
			aucs.append(res["AUC"])
			nsaucs.append(res["NS-AUC"])
			spearmans.append(res["spearman-rho"]*int(res["spearman-p"]<0.01))
			#if (max_rho<res["spearman-rho"]):
			if (max_rho<res["NS-AUC"]):
				id_max = iter_seed
				max_rho = res["NS-AUC"]
			for val in max_metrics:
				if (max_metrics[val]<res[val]):
					max_metrics[val] = res[val]
			jsdists.append(res["jsdist"])
			pcadists.append(res['PCAdist'])
			N += 1
		max_spearman += R[data_seed][id_max]["spearman-rho"]
		if (plot_it):
			feature_scores_true = R[data_seed][id_max]["feature_scores_true"]
			feature_scores = R[data_seed][id_max]["feature_scores"]
			rho = R[data_seed][id_max]["spearman-rho"]
			Nf = len(feature_scores)
			f_lst = ["F%d" % (i+1) for i in range(Nf)]
			plotdata = pd.DataFrame({'predicted': feature_scores, "true": feature_scores_true}, index=f_lst)
			plotdata.plot(kind="bar")
			plt.plot([0, Nf], [0]*2, "k--")
			plt.plot(range(Nf), feature_scores, "b-")
			plt.plot(range(Nf), feature_scores_true, "r-")
			plt.xticks(rotation=27,fontsize=fsize)
			plt.yticks(fontsize=fsize)
			plt.legend(fontsize=fsize)#fsize*2/3)
			plt.title("Spearman's correlation %.2f" % rho, fontsize=fsize)
			plt.savefig("barplot_features_%s_%d.png" % (dataset_type, data_seed), bbox_inches="tight")
			plt.close()
	print("Mean Spearman's rho of top models %.3f (%d data sets)" % (max_spearman/len(R), len(R)))
			
	for lst in ["aucs", "nsaucs", "spearmans", "pcadists"]:
		print("%s:\t\t%s%.3f (%.3f %.3f %.3f)\t+- %.4f\t(N=%d)" % (lst, "\t"*int("aucs" in lst), np.mean(eval(lst)), np.min(eval(lst)), np.median(eval(lst)), np.max(eval(lst)), np.std(eval(lst)), N))
	print("")
	if (plot_it):
		pal = {"AUCs": "indianred", "NS-AUCs": "goldenrod", r"(0-1) $\rho$": "forestgreen", "(0-1) PCA-dist": "steelblue", "(0-1) JS-dist": "purple"}
		pal = {k: "steelblue" for k in pal}
		pal.update({"AUCs": "indianred", "NS-AUCs": "goldenrod"})
		norm01 = lambda x : ((np.array(x)-np.min(x))/np.max(np.array(x)-np.min(x))).tolist()
		norm_pcadists = norm01(pcadists)
		norm_spearmans = norm01(spearmans)
		norm_jsdists = norm01(jsdists)
		data = pd.DataFrame([aucs, nsaucs, norm_spearmans, norm_pcadists, norm_jsdists], index=["AUCs", "NS-AUCs", r"(0-1) $\rho$", "(0-1) PCA-dist", "(0-1) JS-dist"], columns=range(N)).T
		fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,8))
		sns.boxplot(data, palette=pal, ax=ax)
		for val in max_metrics:
			if ("AUC" in val):
				ax.plot([0,len(max_metrics)], [max_metrics[val]]*2, linestyle="--", color=pal[val+"s"])
				ax.text(len(max_metrics)*3/5, max_metrics[val]+0.03*(-2)**int("NS-AUC" in val), 'average best %s across data sets' % val, c=pal[val+"s"], fontsize=fsize*2/3)
		ax.set_xticklabels(ax.get_xticklabels(), rotation=27,fontsize=fsize)
		ax.set_yticklabels(ax.get_yticklabels(), fontsize=fsize)
		ax.set_ylabel(r"Values (N=%d$\times$%d iterations)" % (N/len(R), len(R)), fontsize=fsize)
		plt.gca().spines[['top','right']].set_visible(False)
		plt.savefig("results_interpretable_%s.png" % dataset_type, bbox_inches="tight")
		plt.close()

if ("interpretable" in runtests):
	if (not os.path.exists("%s/results_interpretable.pck")):
		fnames = ["%s/results_interpretable_%s.pck" % (folder, dtype) for dtype in ["interpretable","deviated"]]
		results_interpretable = {}
		for fname in fnames:
			with open(fname, "rb") as f:
				results_interpretable.setdefault(fname.split(".pck")[0].split("_")[-1], pickle.load(f))
	else:
		with open("%s/results_interpretable.pck" % folder, "rb") as f:
			results_interpretable = pickle.load(f)
		
	for dataset_type in results_interpretable:
		print("* %s" % dataset_type)
		result_interpretable(results_interpretable[dataset_type], dataset_type)

##################################################
## Sparsity (associations) (synthetic datasets) ##
##################################################
print("------------------- Sparsity [associations] (synthetic datasets)")

if (not os.path.exists("%s/results_sparsity.pck")):
	fnames = glob("%s/results_sparsity_*.pck" % folder)
	fnames = [fname for fname in fnames if (len(fname.split("_"))==4)]
	dataset_names = list(set([fname.split("_")[2] for fname in fnames]))
	results_sparsity = {}
	for dataset_name in dataset_names:
		results_iter = {}
		fnames = glob("%s/results_sparsity_%s_*.pck" % (folder, dataset_name))
		fnames = [fname for fname in fnames if (len(fname.split("_"))==4)]
		for fname in fnames:
			seed = int(fname.split("_")[-1].split(".pck")[0])
			with open(fname, "rb") as f:
				results_iter.setdefault(seed, pickle.load(f))
		results_sparsity.setdefault(dataset_name, results_iter)
else:
	with open("%s/results_sparsity.pck" % folder, "rb") as f:
		results_sparsity = pickle.load(f)
		
def boxplot_metric(im, m, ndata, niter, fsize, dataset_type, b_type, var="sparsity", var_name="Sparsity number", ylim=None, no_legend=False):
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
	mm = pd.DataFrame(m)
	mmm = pd.melt(mm)
	mmm[var] = [i for _ in mm.columns for i in mm.index]
	dfs = []
	for i in mmm.index:
		N = len(mmm.loc[i]["value"])
		df = pd.DataFrame([mmm.loc[i]["value"]], index=["value"], columns=range(N)).T
		df[var] = mmm.loc[i][var]
		df["variable"] = mmm.loc[i]["variable"]
		dfs.append(df)
	dfs = pd.concat(tuple(dfs), axis=0)
	pal = {"FastaiCollabWrapper": "lightcoral", "JELI": "sandybrown", "HAN": "forestgreen", "NIMCGCN": "gold", 
		"FM2":"teal", "CrossFM2":"mediumseagreen", "SELT_pca_f": "deepskyblue", "SELT_pca_iu": "steelblue", "SELT_kge":"indigo", 
		"None": "steelblue", "STRING": "goldenrod",
		"PrimeKG": "lightcoral", "Hetionet": "deepskyblue", "DRKG": "mediumseagreen", "PharmKG8k": "teal", "PharmKG": "cyan"}
	#m_name = {0:"AUC", 1:"NDCG", 2:"NS-AUC", 3:"Metric"}[im]
	m_name = {0:"AUC", 1:"NDCG", 2:"NS-AUC", 3:"Inference time", 4: "Training time"}[im]
	dfs.columns = [{var: var_name, "value" : m_name, "variable": "Algorithm"}[c] for c in dfs.columns]
	algo_renames = {
		"FastaiCollabWrapper": "Fast.ai", "SELT_pca_f": "SELT (PCAf)", 
		"SELT_pca_iu": "SELT (PCAiu)", "SELT_kge": "SELT (KGE)",
		"FM2" : "FM", "CrossFM2": "CrossFM"
		}
	fsize=40 ##
	pal = {algo_renames.get(a,a):pal[a] for a in pal}
	dfs["Algorithm"] = [algo_renames.get(a,a) for a in dfs["Algorithm"]]
	if ("Validation metric" in dfs.columns):
		di = {"auc": "AUC", "ndcg": "NDCG", "nsauc": "NS-AUC"}
		col = dfs[["Validation metric"]].values.flatten()
		for k in di:
			col[col==k] = di[k]
		dfs["Validation metric"] = col
	if (b_type != "ablationstudy"):
		sns.boxplot(dfs, x=var_name, y=m_name, hue="Algorithm", ax=ax, palette=pal)
	else:
		sns.boxplot(dfs, y=var_name, x=m_name, hue="Algorithm", orient="h", ax=ax, palette=pal)
	if (len(dfs["Algorithm"].unique()) in [2, 5]):
		for im, m in enumerate(dfs[var_name].unique()):
			aa_lst = {a: dfs[dfs.columns[0]][(dfs[var_name]==m)&(dfs['Algorithm'] == a)] for a in dfs["Algorithm"].unique()}
			mx = np.max([np.max(aa_lst[a])+np.std(aa_lst[a]) for a in dfs["Algorithm"].unique()])
			if (len(dfs["Algorithm"].unique())==2):
				a1, a2 = dfs["Algorithm"].unique()
				res = stats.f_oneway(aa_lst[a1], aa_lst[a2])
				if (res.pvalue<0.05):
					dec = str(res.pvalue)[-2:]
					ax.text(im-0.25, 1.03*mx, ("*"+"**"*int(res.pvalue<0.01))+"    "+(r"p=$10^{-%s}$" % dec), c=pal[a1], fontsize=fsize)
					ax.plot([im-0.5,im+0.5], [mx]*2, c=pal[a1])
	if (b_type != "ablationstudy"):
		ax.set_xticklabels(ax.get_xticklabels(), rotation=27, fontsize=fsize)
		if (ylim is not None):
			ax.set_ylim(ylim)
		ax.set_yticklabels(ax.get_yticklabels(), fontsize=fsize)
		ax.set_ylabel((r"%s"+"\n"*int(len(m_name)>12)+" (N=%d iter)") % (m_name, ndata*niter), fontsize=fsize)
		ax.set_xlabel(var_name, fontsize=fsize)
		if (not no_legend):
			ax.legend(fontsize=fsize, loc='center left', bbox_to_anchor=(1, 0.5))
		else:
			ax.get_legend().remove()
	else:
		ax.set_xticklabels(ax.get_xticklabels(), rotation=27, fontsize=fsize)
		if (ylim is not None):
			ax.set_ylim(ylim)
		ax.set_yticklabels(ax.get_yticklabels(), fontsize=fsize)
		ax.set_ylabel(r"%s (N=%d iter)" % (m_name, ndata*niter), fontsize=fsize)
		ax.set_xlabel(var_name, fontsize=fsize)
		if (not no_legend):
			ax.legend(fontsize=fsize, loc='center left', bbox_to_anchor=(1, 0.5))
		else:
			ax.get_legend().remove()
	plt.gca().spines[['top','right']].set_visible(False)
	plt.savefig("results_%s_%s_%s_%d.png" % (var, dataset_type, b_type, im), bbox_inches="tight")
	plt.close()
	
def result_sparsity(R, baselines, dataset_type, b_type):
	if (R is None):
		return None
	## Different datasets
	ndata = len(R)
	metrics_aucs, metrics_ndcg, metrics_nsauc = [{b: {} for b in baselines} for i in range(3)]
	for data_seed in R:
		iter_sparsity = []
		auc_mean = None
		## Different sparsity numbers
		for sparsity in R[data_seed]:
			if (sparsity in [0.99]):
				continue
			iter_mat = None
			metrics = None
			models = None
			## Different trainings
			niter = len(R[data_seed][sparsity])
			for si, iter_seed in enumerate(R[data_seed][sparsity]):
				res = R[data_seed][sparsity][iter_seed]
				mat = pd.DataFrame(res).values
				if (iter_mat is None):
					d1, d2 = mat.shape 
					iter_mat = np.empty((d1, d2, niter))
					models = pd.DataFrame(res).columns
					metrics = pd.DataFrame(res).index
					if (auc_mean is None):
						auc_mean = np.zeros(len(models))
				iter_mat[:,:,si] = mat
			for ia, metrics_m in enumerate([metrics_aucs, metrics_ndcg, metrics_nsauc]):
				for ic, col in enumerate(models):
					if (col in baselines):
						di = metrics_m[col]
						auc_col_sparsity = di.get(sparsity,[])+iter_mat[ia,ic,:].tolist()
						di.update({sparsity: auc_col_sparsity})
						metrics_m.update({col: di})
			mean_mat, std_mat = iter_mat.mean(axis=2), iter_mat.std(axis=2)
			mat = np.vectorize(lambda x : x + " +-")(mean_mat.round(3).astype(str))
			mat = np.vectorize(lambda x,y : x + y)(mat, std_mat.round(4).astype(str))
			auc_mean += mean_mat[0,:]
			iter_sparsity.append((sparsity, iter_mat))
	
	print("Results")
	for i in ["aucs", "nsauc", "ndcg"]:
		X = pd.DataFrame(eval("metrics_"+i))
		for k in X.index:
			for j in X.columns:
				print((i, k, j, np.round(np.mean(X.loc[k][j]), 2), np.round(np.std(X.loc[k][j]), 3)))
	
	if (plot_it):	
		for im, m in enumerate([metrics_aucs, metrics_ndcg, metrics_nsauc]):
			boxplot_metric(im, m, ndata, niter, fsize, dataset_type, b_type)
	print("")
	
if ("sparsity" in runtests):
	for dataset_name in results_sparsity:
		if (dataset_name in ["synthetic"]):
			continue
		print("* %s" % dataset_name)
		result_sparsity(results_sparsity[dataset_name], ["FastaiCollabWrapper", "JELI", "HAN", "NIMCGCN"], dataset_name, "benchmark") 
		result_sparsity(results_sparsity[dataset_name], ["FM2", "CrossFM2", "JELI", "SELT_pca_f", "SELT_pca_iu", "SELT_kge"], dataset_name, "ablationstudy") 

##################################################
## Performance (drug repurposing datasets)      ##
##################################################
print("------------------- Drug repurposing (drug repurposing datasets)")

if (not os.path.exists("%s/results_drug_repurposing.pck" % folder)):
	fnames = glob("%s/results_drug_repurposing-*.pck" % folder)
	results_drug_repurposing = {}
	for fname in fnames:
		with open(fname, "rb") as f:
			results_drug_repurposing.setdefault(fname.split(".pck")[0].split("-")[-1], pickle.load(f))
else:
	with open("%s/results_drug_repurposing.pck" % folder, "rb") as f:
		results_drug_repurposing = pickle.load(f)
	
def result_drug_repurposing(R, baselines, dataset_type):
	if (R is None):
		return None
	## Different trainings
	iter_mat = None
	metrics = None
	models = None
	niter = len(R)
	metrics_df = {b: {} for b in baselines}
	for si, iter_seed in enumerate(R):
		res = R[iter_seed]
		mat = pd.DataFrame(res).values
		if (iter_mat is None):
			d1, d2 = mat.shape 
			iter_mat = np.empty((d1, d2, niter))
			models = pd.DataFrame(res).columns
			metrics = pd.DataFrame(res).index
			auc_mean = np.zeros(len(models))
		iter_mat[:,:,si] = mat
	for ic, col in enumerate(models):
		if (col in baselines):
			di = metrics_df[col]
			for im, m in enumerate(["auc"]):
				im = 0
				auc_col_metric = di.get(m,[])+iter_mat[im,ic,:].tolist()
				di.update({m: auc_col_metric})
			metrics_df.update({col: di})
	mean_mat, std_mat = iter_mat.mean(axis=2), iter_mat.std(axis=2)
	mat = np.vectorize(lambda x : x + " +-")(mean_mat.round(3).astype(str))
	mat = np.vectorize(lambda x,y : x + y)(mat, std_mat.round(4).astype(str))
	print(pd.DataFrame(mat, index=metrics, columns=models))

	print(pd.DataFrame(np.round(mean_mat[0,:].reshape(1,-1),3), index=["avg. auc"], columns=models).T.sort_values(by="avg. auc", ascending=False).T)
	print("")
	
	return metrics_df, niter
	if (plot_it):
		boxplot_metric(0, metrics_df, 1, niter, fsize, dataset_type, "", var="metric", var_name="Validation metric")
	
if ("drug_repurposing" in runtests):
	metrics_df_ = {b: {} for b in ["JELI","HAN","FastaiCollabWrapper","NIMCGCN"]}
	for dataset_name in results_drug_repurposing:
		print("* %s" % dataset_name)
		metrics_df, niter = result_drug_repurposing(results_drug_repurposing[dataset_name], ["JELI","HAN","FastaiCollabWrapper","NIMCGCN"], dataset_name)
		for a in metrics_df_:
			di = metrics_df_.get(a, {})
			di.update({{"PREDICT_Gottlieb": "PREDICT-G"}.get(dataset_name, dataset_name) : metrics_df.get(a)["auc"]})
			metrics_df_.update({a: di})
	if (plot_it):
		boxplot_metric(0, metrics_df_, 1, niter, fsize, "compare", "", var="metric", var_name="Data set", ylim=(0.850,0.975))
		
##################################################
## Performance (MovieLens dataset)              ##
##################################################
print("------------------- Recommendation (MovieLens dataset)")

if (not os.path.exists("%s/results_movielens.pck" % folder)):
	fnames = glob("%s/results_movielens-*.pck" % folder)
	results_movielens = {}
	for fname in fnames:
		with open(fname, "rb") as f:
			results_movielens.setdefault(fname.split(".pck")[0].split("-")[-1], pickle.load(f))
else:
	with open("%s/results_movielens.pck" % folder, "rb") as f:
		results_movielens = pickle.load(f)
	
def result_movielens(R, baselines, dataset_type):
	if (R is None):
		return None
	## Different trainings
	iter_mat = None
	metrics = None
	models = None
	niter = len(R)
	metrics_df = {b: {} for b in baselines}
	for si, iter_seed in enumerate(R):
		res = R[iter_seed]
		mat = pd.DataFrame(res).values
		if (iter_mat is None):
			d1, d2 = mat.shape 
			iter_mat = np.empty((d1, d2, niter))
			models = pd.DataFrame(res).columns
			metrics = pd.DataFrame(res).index
			auc_mean = np.zeros(len(models))
		iter_mat[:,:,si] = mat
	for ic, col in enumerate(models):
		if (col in baselines):
			di = metrics_df[col]
			for im, m in enumerate(["auc"]):
				im = 0
				auc_col_metric = di.get(m,[])+iter_mat[im,ic,:].tolist()
				di.update({m: auc_col_metric})
			metrics_df.update({col: di})
	mean_mat, std_mat = iter_mat.mean(axis=2), iter_mat.std(axis=2)
	mat = np.vectorize(lambda x : x + " +-")(mean_mat.round(3).astype(str))
	mat = np.vectorize(lambda x,y : x + y)(mat, std_mat.round(4).astype(str))
	print(pd.DataFrame(mat, index=metrics, columns=models))

	print(pd.DataFrame(np.round(mean_mat[0,:].reshape(1,-1),3), index=["avg. auc"], columns=models).T.sort_values(by="avg. auc", ascending=False).T)
	print("")
	
	return metrics_df, niter
	if (plot_it):
		boxplot_metric(0, metrics_df, 1, niter, fsize, dataset_type, "", var="metric", var_name="Validation metric")
	
if ("movielens" in runtests):
	metrics_df_ = {b: {} for b in ["JELI","HAN","FastaiCollabWrapper","NIMCGCN"]}
	for dataset_name in results_movielens:
		print("* %s" % dataset_name)
		metrics_df, niter = result_movielens(results_movielens[dataset_name], ["JELI","HAN","FastaiCollabWrapper","NIMCGCN"], dataset_name)
		for a in metrics_df_:
			di = metrics_df_.get(a, {})
			di.update({{"PREDICT_Gottlieb": "PREDICT-G"}.get(dataset_name, dataset_name) : metrics_df.get(a)["auc"]})
			metrics_df_.update({a: di})
	if (plot_it):
		boxplot_metric(0, metrics_df_, 1, niter, fsize, "movielens", "", var="metric", var_name="Data set")
		
##################################################
## Parameter impact (dimension) on deviated     ##
##################################################
print("------------------- Effect of the dimension (deviated dataset)")

def result_parameter(R, baselines, dataset_type, b_type):
	if (R is None):
		return None
	## Different datasets
	ndata = len(R)
	metrics_aucs, metrics_ndcg, metrics_nsauc = [{b: {} for b in baselines} for i in range(3)]
	for data_seed in R:
		iter_parameter = []
		auc_mean = None
		## Different dimensions numbers
		for ndim in R[data_seed]:
			iter_mat = None
			metrics = None
			models = None
			## Different trainings
			niter = len(R[data_seed][ndim])
			for si, iter_seed in enumerate(R[data_seed][ndim]):
				res = R[data_seed][ndim][iter_seed]
				mat = pd.DataFrame(res).values
				if (iter_mat is None):
					d1, d2 = mat.shape 
					iter_mat = np.empty((d1, d2, niter))
					models = pd.DataFrame(res).columns
					metrics = pd.DataFrame(res).index
					if (auc_mean is None):
						auc_mean = np.zeros(len(models))
				iter_mat[:,:,si] = mat
			for ia, metrics_m in enumerate([metrics_aucs, metrics_ndcg, metrics_nsauc]):
				for ic, col in enumerate(models):
					if (col in baselines):
						di = metrics_m[col]
						auc_col_ndim = di.get(ndim,[])+iter_mat[ia,ic,:].tolist()
						di.update({ndim: auc_col_ndim})
						metrics_m.update({col: di})
			mean_mat, std_mat = iter_mat.mean(axis=2), iter_mat.std(axis=2)
			mat = np.vectorize(lambda x : x + " +-")(mean_mat.round(3).astype(str))
			mat = np.vectorize(lambda x,y : x + y)(mat, std_mat.round(4).astype(str))
			auc_mean += mean_mat[0,:]
			iter_parameter.append((ndim, iter_mat))

	for i in ["aucs", "nsauc", "ndcg"]:
		X = pd.DataFrame(eval("metrics_"+i))
		for k in X.index:
			for j in X.columns:
				pass

	if (plot_it):	
		for im, m in enumerate([metrics_aucs, metrics_ndcg, metrics_nsauc]):
			boxplot_metric(im, m, ndata, niter, fsize, dataset_type, b_type+"-"+["AUC","NDCG",'NSAUC'][im], var="dimension", var_name="Embedding dimension", no_legend=True)
	print("")

for ii in ["", "2"]:

	if (not os.path.exists("%s/results_parameter%s.pck" % (folder, ii))):
		fnames = glob("%s/results_parameter%s_*.pck" % (folder, ii))
		fnames = [fname for fname in fnames if (len(fname.split("_"))==4)]
		dataset_names = list(set([fname.split("_")[2] for fname in fnames]))
		results_parameter = {}
		for dataset_name in dataset_names:
			results_iter = {}
			fnames = glob("%s/results_parameter%s_%s_*.pck" % (folder, ii, dataset_name))
			fnames = [fname for fname in fnames if (len(fname.split("_"))==4)]
			for fname in fnames:
				seed = int(fname.split("_")[-1].split(".pck")[0])
				with open(fname, "rb") as f:
					results_iter.setdefault(seed, pickle.load(f))
			results_parameter.setdefault(dataset_name, results_iter)
	else:
		with open("%s/results_parameter%s.pck" % (folder,ii), "rb") as f:
			results_parameter = pickle.load(f)
		
	if ("parameter_impact" in runtests):
		for dataset_name in results_parameter:
			if (dataset_name in ["synthetic"]):
				continue
			print("* %s" % dataset_name)
			result_parameter(results_parameter[dataset_name], ["JELI"], dataset_name, f"parameter-impact{ii}") 
			
##################################################
## Parameter impact (order) on deviated         ##
##################################################
print("------------------- Effect of the order (deviated dataset)")

def result_parameter_order(R, baselines, dataset_type, b_type):
	if (R is None):
		return None
	## Different datasets
	ndatain = np.max([len(R[r]) for r in R])
	R = {r: R[r] for r in R if (len(R[r])==ndatain)} ## only full runs
	ndata = len(R)
	metrics_aucs, metrics_ndcg, metrics_nsauc, metrics_testtime, metrics_traintime = [{b: {} for b in baselines} for i in range(5)]
	for data_seed in R:
		iter_parameter = []
		auc_mean = None
		## Different dimensions numbers
		for order in R[data_seed]:
			iter_mat = None
			metrics = None
			models = None
			## Different trainings
			niter = len(R[data_seed][order])
			for si, iter_seed in enumerate(R[data_seed][order]):
				res = R[data_seed][order][iter_seed]
				mat = pd.DataFrame(res).values
				if (iter_mat is None):
					d1, d2 = mat.shape 
					iter_mat = np.empty((d1, d2, niter))
					models = pd.DataFrame(res).columns
					metrics = pd.DataFrame(res).index
					if (auc_mean is None):
						auc_mean = np.zeros(len(models))
				iter_mat[:,:,si] = mat
			for ia, metrics_m in enumerate([metrics_aucs, metrics_ndcg, metrics_nsauc, metrics_testtime, metrics_traintime]):
				for ic, col in enumerate(models):
					if (col in baselines):
						di = metrics_m[col]
						auc_col_ndim = di.get(order,[])+iter_mat[ia,ic,:].tolist()
						di.update({order: auc_col_ndim})
						metrics_m.update({col: di})
			mean_mat, std_mat = iter_mat.mean(axis=2), iter_mat.std(axis=2)
			mat = np.vectorize(lambda x : x + " +-")(mean_mat.round(3).astype(str))
			mat = np.vectorize(lambda x,y : x + y)(mat, std_mat.round(4).astype(str))
			auc_mean += mean_mat[0,:]
			iter_parameter.append((order, iter_mat))

	for i in ["aucs", "nsauc", "ndcg", "testtime", "traintime"]:
		X = pd.DataFrame(eval("metrics_"+i))
		for k in X.index:
			for j in X.columns:
				pass

	if (plot_it):	
		for im, m in enumerate([metrics_aucs, metrics_ndcg, metrics_nsauc, metrics_testtime, metrics_traintime]):
			boxplot_metric(im, m, ndata, niter, fsize, dataset_type, b_type+"-"+["AUC","NDCG",'NSAUC','testtime', 'traintime'][im], var="order", var_name="RHOFM order", no_legend=True)
	print("")

if (not os.path.exists("%s/results_order.pck" % (folder))):
	fnames = glob("%s/results_order_*.pck" % (folder))
	fnames = [fname for fname in fnames if (len(fname.split("_"))==4)]
	dataset_names = list(set([fname.split("_")[2] for fname in fnames]))
	results_parameter = {}
	for dataset_name in dataset_names:
		results_iter = {}
		fnames = glob("%s/results_order_%s_*.pck" % (folder, dataset_name))
		fnames = [fname for fname in fnames if (len(fname.split("_"))==4)]
		for fname in fnames:
			seed = int(fname.split("_")[-1].split(".pck")[0])
			with open(fname, "rb") as f:
				results_iter.setdefault(seed, pickle.load(f))
		results_parameter.setdefault(dataset_name, results_iter)
else:
	with open("%s/results_order.pck" % (folder), "rb") as f:
		results_parameter = pickle.load(f)
	
if ("order_impact" in runtests):
	for dataset_name in results_parameter:
		if (dataset_name in ["synthetic"]):
			continue
		print("* %s" % dataset_name)
		result_parameter_order(results_parameter[dataset_name], ["JELI"], dataset_name, "order-impact") 
		
##################################################
## Parameter other impact on deviated           ##
##################################################
print("------------------- Effect of other parameters (deviated dataset)")

def result_scalability(R, baselines, dataset_type, b_type):
	if (R is None):
		return None
	## Different datasets
	#ndatain = np.max([len(R[r]) for r in R])
	#R = {r: R[r] for r in R if (len(R[r])==ndatain)} ## only full runs
	for param in R:
		#print(param)
		metrics_aucs, metrics_ndcg, metrics_nsauc, metrics_testtime, metrics_traintime = [{b: {} for b in baselines} for i in range(5)]
		ndata = len(R[param])
		for data_seed in R[param]:
			iter_parameter = []
			auc_mean = None
			for value in R[param][data_seed]:
				iter_mat = None
				metrics = None
				models = None
				## Different trainings
				niter = len(R[param][data_seed][value])
				for si, iter_seed in enumerate(R[param][data_seed][value]):
					res = R[param][data_seed][value][iter_seed]
					mat = pd.DataFrame(res).values
					if (iter_mat is None):
						d1, d2 = mat.shape 
						iter_mat = np.empty((d1, d2, niter))
						models = pd.DataFrame(res).columns
						metrics = pd.DataFrame(res).index
						if (auc_mean is None):
							auc_mean = np.zeros(len(models))
					iter_mat[:,:,si] = mat
				for ia, metrics_m in enumerate([metrics_aucs, metrics_ndcg, metrics_nsauc, metrics_testtime, metrics_traintime]):
					for ic, col in enumerate(models):
						if (col in baselines):
							di = metrics_m[col]
							auc_col_ndim = di.get(value,[])+iter_mat[ia,ic,:].tolist()
							di.update({value: auc_col_ndim})
							metrics_m.update({col: di})
				mean_mat, std_mat = iter_mat.mean(axis=2), iter_mat.std(axis=2)
				mat = np.vectorize(lambda x : x + " +-")(mean_mat.round(3).astype(str))
				mat = np.vectorize(lambda x,y : x + y)(mat, std_mat.round(4).astype(str))
				auc_mean += mean_mat[0,:]
				iter_parameter.append((value, iter_mat))

		for i in ["aucs", "nsauc", "ndcg", "testtime", "traintime"]:
			X = pd.DataFrame(eval("metrics_"+i))
			print([(i, param, k2,np.mean(X[k1].loc[k2])) for k1 in X.columns for k2 in X.index])
			for k in X.index:
				for j in X.columns:
					pass

		if (plot_it):	
			for im, m in enumerate([metrics_aucs, metrics_ndcg, metrics_nsauc, metrics_testtime, metrics_traintime]):
				metrics_list = ["AUC","NDCG",'NSAUC','testtime', 'traintime']
				boxplot_metric(im, m, ndata, niter, fsize, dataset_type, b_type+"-"+param+"-"+metrics_list[im], var=param, var_name={"npoints": r"$n_i \times n_u$", "ndim": "d", "nfeatures": "F", "thres": r"$\tau$"}.get(param), no_legend=True)
		print("")

if (not os.path.exists("%s/results_scalability.pck" % (folder))):
	fnames = glob("%s/results_scalability_*.pck" % (folder))
	fnames = [fname for fname in fnames if (len(fname.split("_"))>=5)]
	dataset_names = list(set([fname.split("_")[2] for fname in fnames]))
	results_parameter = {}
	for dataset_name in dataset_names:
		results_iter = {}
		fnames = glob("%s/results_scalability_%s_*.pck" % (folder, dataset_name))
		fnames = [fname for fname in fnames if ((len(fname.split("_"))==5) or (len(fname.split("_"))==6 and "sim_thres" in fname))]
		for fname in fnames:
			seed = int(fname.split("_")[3].split(".pck")[0])
			param, value = tuple(fname.split(".pck")[0].split("_")[-1].split("="))
			with open(fname, "rb") as f:
				di = results_iter.get(param, {})
				ddi = di.get(seed, {})
				ddi.update({float(value): pickle.load(f)})
				di.update({seed: ddi})
				results_iter.update({param : di})
		results_parameter.setdefault(dataset_name, results_iter)
else:
	with open("%s/results_scalability.pck" % (folder), "rb") as f:
		results_parameter = pickle.load(f)
	
if ("scalability" in runtests):
	for dataset_name in results_parameter:
		if (dataset_name in ["synthetic"]):
			continue
		print("* %s" % dataset_name)
		result_scalability(results_parameter[dataset_name], ["JELI"], dataset_name, "scalability") 

##################################################
## Graph-Based Prior (TRANSCRIPT dataset)       ##
##################################################
print("------------------- Graph-Based Prior (TRANSCRIPT dataset)")

###################################

if ("add_prior" in runtests):
		
	print("*** Simple priors (Similarity (default), PPI)")
	if (not os.path.exists("%s/results_add_prior.pck")):
		fnames = glob("%s/results_add_prior_*.pck" % folder)
		results_add_prior = {}
		results_embs = {}
		for fname in fnames:
			with open(fname, "rb") as f:
				iter_seed = fname.split("_prior_")[-1].split(".pck")[0]
				res = pickle.load(f)
				res_scores = {k: {u: res[k][u] for u in res[k] if (u in ["auc","ndcg","nsauc"])} for k in res if (len(res[k])>0)}
				res_embs = {k: res[k]["feature_embeddings"] for k in res if (len(res[k])>0)}
				if (len(res_scores)>0):
					results_add_prior.update({ iter_seed: res_scores })
					results_embs.update({ iter_seed: res_embs })
	else:
		with open("%s/results_add_prior.pck" % folder, "rb") as f:
			results_add_prior = pickle.load(f)
	
	print("*** KG priors (KG from the literature)")
	if (not os.path.exists("%s/results_add_prior2.pck")):
		fnames = glob("%s/results_add_prior2_*.pck" % folder)
		results_add_prior2 = {}
		results_embs2 = {}
		for fname in fnames:
			with open(fname, "rb") as f:
				iter_seed = fname.split("_prior2_")[-1].split(".pck")[0]
				res = pickle.load(f)
				res_scores = {k: {u: res[k][u] for u in res[k] if (u in ["auc","ndcg","nsauc"])} for k in res if (len(res[k])>0)}
				res_embs = {k: res[k]["feature_embeddings"] for k in res if (len(res[k])>0)}
				if (len(res_scores)>0):
					results_add_prior2.update({ iter_seed: res_scores })
					results_embs2.update({ iter_seed: res_embs })
	else:
		with open("%s/results_add_prior2.pck" % folder, "rb") as f:
			results_add_prior2 = pickle.load(f)
		
	id_max2k = [(k, len(results_add_prior2[k])) for k in results_add_prior2]
	id_max2 = id_max2k[np.argmax([l for k,l in id_max2k])][0]
	priors2 = list(results_add_prior2[id_max2].keys())
	id_maxk = [(k, len(results_add_prior[k])) for k in results_add_prior if (len(results_add_prior[k])<=2)]
	id_max = id_maxk[np.argmax([l for k,l in id_maxk])][0]
	priors = list(results_add_prior[id_max].keys())
	metrics_prior2 = {
		m: {"auc": [], "ndcg": [], "nsauc": []} for m in priors2+priors
		}
	niter1, niter = 0, 0	
	metrics_priors_lst2 = [m for m in priors2+priors]
	metrics_priors_names2 = [x for x in metrics_priors_lst2]
	for iter_seed in results_add_prior:
		niter1 += 1
		for p, pp in zip(metrics_priors_lst2, metrics_priors_names2):
			if (pp in results_add_prior[iter_seed]):
				for m in metrics_prior2[p]:
					lst = metrics_prior2[p][m]+[results_add_prior[iter_seed][pp][m]]
					metrics_prior2[p][m] = lst
	for iter_seed in results_add_prior2:
		niter += 1
		for p, pp in zip(metrics_priors_lst2, metrics_priors_names2):
			if (pp in results_add_prior2[iter_seed]):
				for m in metrics_prior2[p]:
					lst = metrics_prior2[p][m]+[results_add_prior2[iter_seed][pp][m]]
					metrics_prior2[p][m] = lst
	metrics_prior2 = {k : metrics_prior2[k] for k in metrics_priors_lst2}
	values2 = np.array([[np.mean(metrics_prior2[p][m]) for p in metrics_prior2] for m in ["auc", "ndcg", "nsauc"]]+[[len(metrics_prior2[p]["auc"]) for p in metrics_prior2]])
	print(pd.DataFrame(values2, index=["avg. "+m for m in  ["auc", "ndcg", "nsauc"]]+["nruns"], columns=metrics_priors_lst2))
	print("")
	
	values2_std = np.array([[np.std(metrics_prior2[p][m]) for p in metrics_prior2] for m in ["auc", "ndcg", "nsauc"]]+[[len(metrics_prior2[p]["auc"]) for p in metrics_prior2]])
	print(pd.DataFrame(values2_std, index=["std. "+m for m in  ["auc", "ndcg", "nsauc"]]+["nruns"], columns=metrics_priors_lst2))
	print("")

	if (plot_it):
		boxplot_metric(3, metrics_prior2, 1, niter, fsize, "comparison_all", "", var="metric", var_name="Validation metric")
		
###################################

import math

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv

## from https://stackoverflow.com/questions/42697933/colormap-with-maximum-distinguishable-colours
def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80
    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)
    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)
    arr_by_shade_columns = arr_by_shade_rows.T
    number_of_partitions = arr_by_shade_columns.shape[0]
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)
    initial_cm = hsv(nums_distributed_like_rising_saw)
    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier
    return ListedColormap(initial_cm)
	
if ("gene_enrichment" in runtests):

	if ("add_prior" not in runtests):
		print("Add add_prior test, exiting now")
		exit()

	print("\n*** Variation of importance scores and embeddings across iterations for each prior")
	imp_scores_all, imp_scores, embs = {}, {}, {}
	for p, pp in zip(metrics_priors_lst2, metrics_priors_names2):
		## JELI feature importance scores for each iter seed
		Np=0
		if (pp in results_add_prior2[id_max2]):
			imp_scores_all[p] = np.matrix([results_embs2[iter_seed][pp].numpy().sum(axis=1).flatten().tolist() for iter_seed in results_add_prior2 if (pp in results_embs2[iter_seed])])
			imp_scores[p] = imp_scores_all[p].mean(axis=0)
			F, d = results_embs2[id_max2][pp].numpy().shape
			all_embs = np.zeros((len(results_add_prior2), F, d))
			for it, iter_seed in enumerate(results_add_prior2):
				if (pp not in results_embs2[iter_seed]):
					continue
				Np+=1
				v = results_embs2[iter_seed][pp].numpy()
				if (p not in embs):
					embs[p] = v
				else:
					embs[p] += v
				all_embs[it,:,:] = v
		else:
			imp_scores_all[p] = np.matrix([results_embs[iter_seed][pp].numpy().sum(axis=1).flatten().tolist() for iter_seed in results_add_prior if (pp in results_embs[iter_seed])])
			imp_scores[p] = imp_scores_all[p].mean(axis=0)
			F, d = results_embs[id_max][pp].numpy().shape
			all_embs = np.zeros((len(results_add_prior), F, d))
			for it, iter_seed in enumerate(results_add_prior):
				if (pp not in results_embs[iter_seed]):
					continue
				Np+=1
				v = results_embs[iter_seed][pp].numpy()
				if (p not in embs):
					embs[p] = v
				else:
					embs[p] += v
				all_embs[it,:,:] = v
		embs[p] /= Np
		## Variation in importance scores and embeddings due to randomness (random seeds)
		print(f"{p}\tImportance scores:\tVar={imp_scores_all[p].var(axis=0).mean()}\tN={imp_scores_all[p].shape[0]}\tF={imp_scores_all[p].shape[1]}")
		print(f"\tEmbeddings:\tMin={np.round(all_embs.min(axis=0).mean(axis=0).mean(),3)}\tMean={np.round(all_embs.mean(axis=0).mean(axis=0).mean(),3)}\tMax={np.round(all_embs.max(axis=0).mean(axis=0).mean(),3)}\tVar={np.round(all_embs.var(axis=0).mean(axis=0).mean(),3)}\tF={F}\td={d}")
	print("")
		
	from stanscofi.utils import load_dataset
	from stanscofi.datasets import Dataset
	if (cuda_on):
		ffolder = "/storage/store3/work/creda/"
	else:
		ffolder = "./"
	data_args = load_dataset("TRANSCRIPT", save_folder="%s/datasets/" % ffolder)
	data_args["name"] = "TRANSCRIPT"
	dataset = Dataset(**data_args)
	
	print("*** Gene pathway enrichment based on embeddings and importance scores across priors")
	## a. Get a ground truth 
	## https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp 50 Hallmark gene sets
	import json
	classes = pd.read_csv(kge_folder+"h.all.v2023.2.Hs.symbols.gmt", sep="\t", index_col=0, header=None)
	nclusters = classes.shape[0]
	classes.index = range(nclusters)
	if (not os.path.exists("true_classes.json")):
		true_classes = {g:[c for c in classes.index if (g in classes.loc[c].values.flatten())] for g in dataset.item_features}
		true_classes = {g:true_classes[g] for g in true_classes if (len(true_classes[g])>0)} ## some genes are not annotated
		with open("true_classes.json", "wb") as f:
			json.dump(f, true_classes)
	with open("true_classes.json", "rb") as f:
		true_classes = json.load(f)

	## b. Test interpretability on embeddings: similarity between clusters of embeddings and meaningful families of genes
	genes = [(ig,g) for ig, g in enumerate(dataset.item_features) if (np.min(true_classes.get(g,[-1]))>0)]
	igenes = [ig for ig,_ in genes]
	ngroups = len(np.unique([int(c) for _, g in genes for c in true_classes[g]]))
	print(f"{ngroups} groups for {len(genes)} annotated genes in total")
	true = [true_classes[g] for _, g in genes]
	
	## Like a Rand Index but where the reference clustering is fuzzy
	def fuzzy_Rand_index(true, pred):
		assert len(true)==len(pred)
		n = len(true)
		N = n*(n-1)/2
		Shared = np.zeros((n, n))
		for ig1, t1 in enumerate(true):
			#Shared[ig1, ig1] = 1
			for ig2, t2 in enumerate(true[(ig1+1):]):
				Shared[ig1, ig2+ig1+1] = int(len(set(t1).intersection(set(t2)))>0)
				#Shared[ig2+ig1+1, ig1] = Shared[ig1, ig2+ig1+1]
		Shared += Shared.T
		np.fill_diagonal(Shared, 1)	
		Clusters = np.zeros((n, n))
		for ig1, pg1 in enumerate(pred):
			#Clusters[ig1, ig1] = 1
			for ig2, pg2 in enumerate(pred[(ig1+1):]):
				Clusters[ig1, ig2+ig1+1] = int(pg1==pg2)
				#Clusters[ig2+ig1+1, ig1] = Clusters[ig1, ig2+ig1+1]
		Clusters += Clusters.T
		np.fill_diagonal(Clusters, 1)
		## Agreement type #1: two genes with at least one shared pathway are clustered together
		a = (np.multiply(Shared, Clusters).sum()-n)/2
		## Agreement type #2: two genes without any shared pathway are not clustered together
		b = np.multiply((Shared+1)%2, (Clusters+1)%2).sum()/2
		npaths = (np.sum(Shared)-n)/2
		return a/npaths, b/(N-npaths), (a+b)/N
		
	def fuzzy_Adj_Rand_index(true, pred):
		assert len(true)==len(pred)
		ngroups = len(np.unique([tt for t in true for tt in t]))
		SharedClusters = np.zeros((ngroups, np.max(pred)+1))
		for t in range(ngroups):
			for c in range(np.max(pred)+1):
				SharedClusters[t, c] = np.sum([int((pg==c) and ((t+1) in true[ig])) for ig, pg in enumerate(pred)])
		a = SharedClusters.sum(axis=1)
		b = SharedClusters.sum(axis=0)
		n = len(true)
		N = n*(n-1)/2
		Nnum = (np.multiply(SharedClusters, SharedClusters-1)/2).sum()
		NA = (np.multiply(a, a-1)/2).sum()
		NB = (np.multiply(b, b-1)/2).sum()
		num = Nnum - (NA*NB)/N
		denom = 0.5*(NA+NB)-(NA*NB)/N
		return num/denom

	#print(fuzzy_Rand_index([[1],[1,2]], [1,1]))		
	#print(fuzzy_Rand_index([[1],[1,2],[3],[3,4]], [1,1,2,2]))
	#print(fuzzy_Rand_index([[1],[1],[3],[3]], [1,1,2,2]))
	#print(fuzzy_Rand_index([[1],[2],[3],[3,4]], [1,1,2,2]))
	#print(fuzzy_Rand_index([[1],[2],[3],[4]], [1,2,3,4]))
	#print(fuzzy_Rand_index([[1],[1,2],[3],[1,4]], [1,2,3,2]))
	#print(fuzzy_Rand_index([[1],[1,2],[3],[1,4]], [1,2,3,4]))
	
	#print(fuzzy_Adj_Rand_index([[1],[1,2],[3]], [1,1,0]))		
	#print(fuzzy_Adj_Rand_index([[1],[1,2],[3],[3,4]], [0,0,1,1]))
	#print(fuzzy_Adj_Rand_index([[1],[1],[3],[3]], [0,0,1,1]))
	#print(fuzzy_Adj_Rand_index([[1],[2],[3],[3,4]], [1,1,0,0]))
	#print(fuzzy_Adj_Rand_index([[1],[2],[3],[4]], [0,1,2,3]))
	#print(fuzzy_Adj_Rand_index([[1],[1,2],[3],[1,4]], [0,1,2,3]))
	#print(fuzzy_Adj_Rand_index([[1],[1,2],[3],[1,4]], [0,1,2,3]))

	from sklearn.cluster import KMeans, HDBSCAN
	from sklearn.metrics import pairwise_distances
	from sklearn.preprocessing import StandardScaler
	a_lst, ri_lst, ari_lst = {}, {}, {}
	clust_algo = ["KMeans", "HDBSCAN"][0]
	df_values2 = pd.DataFrame(values2[:-1,:], index=["avg. "+m for m in  ["auc", "ndcg", "nsauc"]], columns=metrics_priors_lst2)
	for prior in embs:
		X = StandardScaler().fit_transform(embs[prior][igenes,:])
		
		if (clust_algo=="HDBSCAN"):
		
			dists = pairwise_distances(X, metric='euclidean')
			#print((np.min(dists[dists!=0]), np.percentile(dists, 0.25), np.mean(dists), np.percentile(dists, 0.75), np.max(dists)))
			eps = int(np.percentile(dists, 0.25))
			#print(f"eps={eps}"
			clust = lambda _ : HDBSCAN(cluster_selection_epsilon=eps, min_samples=2)
		else:
			eps = np.nan
			clust = lambda _ : KMeans(n_clusters=ngroups, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0, random_state=1234, copy_x=True, algorithm='lloyd')
		
		pred = clust(0).fit(X).labels_
		a, b, RI = fuzzy_Rand_index(true,pred)
		ARI = fuzzy_Adj_Rand_index(true,pred)
		print(f"Prior {prior}, a={a}, b={b}, RI={RI}, ARI={ARI}, nclusters={np.max(pred)+1}, eps={eps}")
		a_lst.setdefault(prior, a)
		ri_lst.setdefault(prior, RI)
		ari_lst.setdefault(prior, ARI)
	print("")
	df_values2.loc["a"] = [a_lst[k] for k in df_values2.columns]
	df_values2.loc["RI"] = [ri_lst[k] for k in df_values2.columns]
	df_values2.loc["ARI"] = [ari_lst[k] for k in df_values2.columns]
	print(df_values2)
	print(df_values2.T.corr(method="spearman"))
