#coding: utf-8

## SELT: "Separate Embedding Learning then Training"
from stanscofi.models import BasicModel
from stanscofi.datasets import Dataset
from scipy.sparse import coo_array
import pandas as pd

from jeli.JELIImplementation import RHOFM, pipeline, to_cuda

class SELT(BasicModel):
    def __init__(self, params=None):
        params_ = self.default_parameters()
        if (params is not None):
            params_.update(params)
        super(SELT, self).__init__(params_)
        self.scalerS, self.scalerP = None, None
        self.name = "SELT" 
        self.model_embeddings = None
        self.model = RHOFM(self.n_dimensions, 2, "linear", self.frozen, self.cuda_on, self.random_state)

    def default_parameters(self): 
        params = {
            "cuda_on": False,
            "frozen": False,
            "n_dimensions": 50,
            "random_state": 1234,
            "fm_params": {"batch_size": 1032, "n_epochs": 25, "loss": "MarginRankingLoss", 
            "opt_params":{'lr': 0.01, "weight_decay": 0.02}, "early_stop":0},
        }
        return params

    def preprocessing(self, dataset, is_training=True, inf=2):
        assert dataset.nitem_features == dataset.nuser_features
        if (is_training):
            self.learn_embeddings(dataset)
        items = coo_array(self.transform(pd.DataFrame(dataset.items.toarray(), index=dataset.item_features, columns=dataset.item_list)))
        users = coo_array(self.transform(pd.DataFrame(dataset.users.toarray(), index=dataset.user_features, columns=dataset.user_list)))
        self.nfeatures = dataset.nitem_features
        return [Dataset(ratings=pd.DataFrame(dataset.ratings.toarray(), index=dataset.item_list, columns=dataset.user_list), users=users, items=items)]
        
    def model_fit(self, dts, random_seed=None): ## TODO problem of reproducibility with RHOFM?
        if (random_seed is None):
            random_seed = self.random_state
        #from FM_baselines import FM2
        #self.model = FM2(self.n_dimensions, dts.nitem_features)
        self.fm_params.update({"random_seed": random_seed})
        self.model.fit(dts, **self.fm_params)

    def model_predict_proba(self, dts):
    	return self.model.predict_proba(dts).toarray()
    	
    def predict(self, scores, threshold=0.5, default_zero_val=1e-31):
        preds = coo_array((scores.toarray()!=0).astype(int)*((-1)**(scores.toarray()<=0.5)))
        return preds
        
    def learn_embeddings(self, dataset):
        raise NotImplemented
        
    def transform(self, entity): ## entity of size F x n 
        raise NotImplemented

