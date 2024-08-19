#coding: utf-8

## SELT: "Separate Embedding Learning then Training" with PCA embeddings on users and items
from sklearn.decomposition import PCA
import numpy as np
from SELT import SELT

class SELT_pca_iu(SELT):
    def __init__(self, params=None):
        params_ = self.default_parameters()
        if (params is not None):
            params_.update(params)
        super(SELT_pca_iu, self).__init__(params_)
        self.name = "SELT_pca_iu" 

    def default_parameters(self): 
        params = super(SELT_pca_iu, self).default_parameters()
        p = {
            "PCA_params": dict(copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', n_oversamples=10, power_iteration_normalizer='auto', random_state=1234),
        }
        params.update(p)
        return params
        
    def learn_embeddings(self, dataset):
        X = np.concatenate((dataset.items.toarray(), dataset.users.toarray()), axis=1)
        pca_params = dict(n_components=self.n_dimensions)
        pca_params.update(self.PCA_params) 
        with np.errstate(invalid="ignore"): # for NaN or 0 variance matrices
            self.model_embeddings = PCA(**pca_params)
            self.model_embeddings.fit(X.T)
        
    def transform(self, entity): ## entity of size F x n 
        assert self.model_embeddings is not None
        return self.model_embeddings.transform(entity.values.T).T
		
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
	
	model = SELT_pca_iu({"n_dimensions": N_DIMENSIONS, "random_state": SEED, "fm_params": {"loss": "CrossEntropyLoss", "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS, "opt_params": {"lr":0.01, "weight_decay":0.02}}})
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
	

