import scipy.sparse as sp
import numpy as np
import joblib


class FilterShortlist(object):
    def __init__(self, keep_all=False):
        self.docs = np.asarray([])
        self.lbls = np.asarray([])
        self.valid_lbls = np.asarray([])
        self.keep_all = keep_all
    
    def fit(self, docs, lbls, lbl_mapping=None):
        self.set_docs_lbls(docs, lbls)
        if sp.issparse(lbl_mapping):
            _, num_lbls = lbl_mapping.shape
            self.num_lbls = num_lbls
            valid_lbls = np.where(np.ravel(lbl_mapping.getnnz(axis=0))>0)[0]
            ones = np.zeros(num_lbls)
            ones[valid_lbls] = 1
            self.valid_lbls = ones
        else:
            self.num_lbls = lbl_mapping
            self.valid_lbls = np.ones(lbl_mapping)
        if self.keep_all:
            self.valid_lbls = np.ones(self.num_lbls)
        print(f"FILTER::#(VALID LABELS) = {int(np.sum(self.valid_lbls))} / {self.num_lbls}")
    
    def set_docs_lbls(self, docs, lbls):
        self.docs = docs
        self.lbls = lbls
    
    def __call__(self, shorty):
        if sp.issparse(shorty):
            shorty = shorty.tolil()
            shorty[self.docs, self.lbls] = 0
            shorty = shorty.dot(sp.diags(
                self.valid_lbls, shape=(self.num_lbls, self.num_lbls))).tocsr()
            return shorty.tocsr()
    
    def save(self, model_path):
        joblib.dump({"docs": self.docs, "lbls": self.lbls,
                     "valid": self.valid_lbls,
                     "n_lbls": self.num_lbls}, model_path)
        
    def load(self, model_path):
        data = joblib.load(model_path)
        self.docs = data["docs"]
        self.lbls = data["lbls"]
        self.valid_lbls = data["valid"]
        self.num_lbls = data["n_lbls"]