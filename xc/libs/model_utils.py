import scipy.sparse as sp
import numpy as np
import joblib


class FilterShortlist(object):
    def __init__(self, keep_all=False):
        self.docs = np.asarray([])
        self.lbls = np.asarray([])
        self.keep_all = keep_all
    
    def fit(self, docs, lbls, lbl_mapping=None):
        self.set_docs_lbls(docs, lbls)
        if sp.issparse(lbl_mapping):
            _, self.num_lbls = lbl_mapping.shape
        else:
            self.num_lbls = lbl_mapping
        
    def set_docs_lbls(self, docs, lbls):
        self.docs = docs
        self.lbls = lbls
    
    def __call__(self, shorty):
        if sp.issparse(shorty) and self.docs.size > 0:
            shorty = shorty.tolil()
            shorty[self.docs, self.lbls] = 0
            shorty = shorty.tocsr()
        return shorty
    
    def save(self, model_path):
        joblib.dump({"docs": self.docs, "lbls": self.lbls,
                     "n_lbls": self.num_lbls}, model_path)
        
    def load(self, model_path):
        data = joblib.load(model_path)
        self.docs = data["docs"]
        self.lbls = data["lbls"]
        self.num_lbls = data["n_lbls"]