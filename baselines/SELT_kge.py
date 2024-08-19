#coding: utf-8

## SELT: "Separate Embedding Learning then Training" with PCA embeddings on users and items
from SELT import SELT
import numpy as np
## https://pykeen.readthedocs.io/en/latest/byo/data.html
from pykeen.triples import TriplesFactory
from pykeen.sampling import PseudoTypedNegativeSampler
import torch

from jeli.JELI import JELI
from jeli.JELIImplementation import pipeline, to_cuda, normalization

class SELT_kge(SELT):
    def __init__(self, params=None):
        params_ = self.default_parameters()
        if (params is not None):
            params_.update(params)
        super(SELT_kge, self).__init__(params_)
        self.name = "SELT_kge"

    def default_parameters(self): 
        params = super(SELT_kge, self).default_parameters()
        p = {
            "kge_params": {"thres_kge": 0.5, "n_epochs": 25, "batch_size": 1032, "loss": "SoftMarginRankingLoss", "interaction": {"MuRE": {"p": 1}}},
            "structure": "linear",
        }
        params.update(p)
        return params
        
    def learn_embeddings(self, dataset):
    	## Create KGE
        kge = JELI({"partial_kge": None, "kge_name": None}).preprocessing(dataset, is_training=True)[0]
        ## Train KGE learning
        tf = TriplesFactory.from_labeled_triples(kge.triplets)
        ## https://pykeen.readthedocs.io/en/stable/api/pykeen.pipeline.pipeline.html
        assert "interaction" in self.kge_params
        interaction = [_ for _ in self.kge_params.get("interaction", {"MuRE": {"p":1}})][0]
        model_args = self.kge_params.get("interaction", {"MuRE": {"p": 1}})[interaction]
        model_args.update({ "embedding_dim": self.n_dimensions })
        results = pipeline(tf=tf, model=interaction, model_kwargs=model_args, optimizer_kwargs=self.kge_params.get("opt_params", {'lr': 1e-3, "weight_decay": 0.01}), 
        	training_kwargs={'batch_size': self.kge_params.get("batch_size", 256), "stopper": None},
                training_loop='sLCWA', negative_sampler=PseudoTypedNegativeSampler, cuda_on=False, loss=self.kge_params.get("loss", "SoftMarginRankingLoss"), epochs=self.kge_params["n_epochs"],
                random_seed=self.random_state) 
        ## https://pykeen.readthedocs.io/en/latest/tutorial/translational_toy_example.html 
        ## Retrieve embeddings for features
        self.feature_embeddings = results["model"].entity_representations[0](indices=to_cuda(torch.LongTensor(range(kge.nfeatures)), kge.cuda_on)).detach().cpu()
        
    def transform(self, entity): ## entity of size F x n
        assert self.feature_embeddings is not None
        entity_ = torch.Tensor(normalization(entity))
        if (self.structure=="linear"):
            X = torch.matmul(self.feature_embeddings.T, entity_)
        else:
            X = structure(entity_, self.feature_embeddings)
        return X
		
if __name__ == "__main__":
	import random
	from stanscofi.training_testing import random_simple_split
	from stanscofi.utils import load_dataset
	from stanscofi.datasets import generate_dummy_dataset, Dataset
	from stanscofi.validation import AUC, NDCGk, compute_metrics
	import random
	import numpy as np

	SEED = 1245
	TEST_SIZE = 0.2
	DATA_NAME = "Synthetic"
	DFOLDER="../datasets/"
	
	if (DATA_NAME=="Synthetic"):
		npositive, nnegative, nfeatures, mean, std = 200, 200, 100, 2, 0.01
		data_args = generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std, random_state=SEED)
	else:
		data_args = load_dataset(DATA_NAME, save_folder=DFOLDER)
		data_args["name"] = DATA_NAME
	
	N_DIMENSIONS = 20
	N_EPOCHS=10
	BATCH_SIZE=1024
	
	np.random.seed(SEED)
	random.seed(SEED)

	## Import dataset
	dataset = Dataset(**data_args)
	
	(train_folds, test_folds), _ = random_simple_split(dataset, TEST_SIZE, metric="cosine", random_state=SEED)
	train = dataset.subset(train_folds)
	test = dataset.subset(test_folds)
	
	model = SELT_kge({"n_dimensions": N_DIMENSIONS, "random_state": SEED, "kge_params": {"thres_kge": 0.5, "n_epochs": N_EPOCHS-N_EPOCHS//2, "opt_params": {"lr":0.001, "weight_decay":0.02}, "batch_size": BATCH_SIZE, "loss": "CrossEntropyLoss", "interaction": {"MuRE": {"p": 1}}}, "fm_params": {"loss": "CrossEntropyLoss", "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS//2, "opt_params": {"lr":0.01, "weight_decay":0.02}}, "structure": "linear"})
	model.fit(train)
	scores = model.predict_proba(test)
	model.print_scores(scores)
	predictions = model.predict(scores, threshold=0)
	model.print_classification(predictions)
	metrics, _ = compute_metrics(scores, predictions, test, metrics=["AUC", "NDCGk"], k=dataset.nitems, beta=1, verbose=False)
	print(metrics)
	y_test = (test.folds.toarray()*test.ratings.toarray()).ravel()
	y_test[y_test<1] = 0
	print("(global) AUC = %.3f" % AUC(y_test, scores.toarray().ravel(), 1, 1))
	print("(global) NDCG@%d = %.3f" % (test.nitems, NDCGk(y_test, scores.toarray().ravel(), test.nitems, 1)))
	

