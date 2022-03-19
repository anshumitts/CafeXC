from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DR
import xclib.evaluation.xc_metrics as xc_metrics
from xc.libs.utils import pbar
import scipy.sparse as sp
import numpy as np
import joblib


class Fusion(object):
    def __init__(self, reg="DecisionTrees", A=0.55, B=1.5, use_psp=False):
        self.A = A
        self.B = B
        if reg == "DecisionTrees":
            self.regressor = DR(max_depth=7)
        if reg == "LogisticRegression":
            self.regressor = LR(penalty='l2', verbose=1, n_jobs=8)

        self.inv_psp = None
        self.use_psp = use_psp
        self.valid = True
        self.val = np.asarray([])

    def split(self, trn_dset, valid_frac=0.035):
        print("FUSION:Generating SPLITS")
        valid_int = int(np.ceil(len(trn_dset)*valid_frac))
        val = np.random.choice(len(trn_dset), size=valid_int, replace=False)
        trn = np.setdiff1d(np.arange(len(trn_dset)), val)
        self.val = val
        return trn, val

    @property
    def val_indices(self):
        return self.val

    def build_psp(self, trn_y):
        print("FUSION:Building PSP")
        self.inv_psp = xc_metrics.compute_inv_propesity(
            trn_y, A=self.A, B=self.B)

    def prep_descriptor(self, m2, m4):
        rows, cols = m2.nonzero()
        module2_data = np.ravel(m2.tolil()[rows, cols].todense())
        module4_data = np.ravel(m4.tolil()[rows, cols].todense())
        inv_psp = np.ravel(self.inv_psp[cols])
        return np.vstack([inv_psp, module2_data, module4_data]).T, rows, cols

    def fit(self, m2, m4, trn_y):
        self.valid = True
        print("FUSION::SETTING UP PARAMETERS")
        X, rows, cols = self.prep_descriptor(m2, m4)
        y = np.ravel(trn_y.tolil()[rows, cols].todense())
        print("FUSION::TRAINING FUSION LAYER")
        self.regressor.fit(X, y)

    def predict(self, m2, m4, batch_size=200, a_min=1e-1):
        num_docs = m4.shape[0]
        final_score = sp.lil_matrix(m4.shape)
        for start in pbar(np.arange(0, num_docs, batch_size),
                          desc="fusion", write_final=True):
            end = min(num_docs, start+batch_size)
            chotu_m4 = m4[start:end]
            chotu_m2 = m2[start:end]
            X, rows, cols = self.prep_descriptor(chotu_m2, chotu_m4)
            y = self.regressor.predict_proba(X)[:, 1]
            y = np.clip(y, a_min, 1-a_min)
            final_score[rows+start, cols] = y
            del chotu_m2, chotu_m4
        return final_score.tocsr()

    def save(self, model_path):
        joblib.dump({"regressor": self.regressor,
                     "inv_psp": self.inv_psp,
                     "valid": self.valid,
                     "val": self.val}, model_path)

    def load(self, model_path):
        data = joblib.load(model_path)
        self.regressor = data["regressor"]
        self.inv_psp = data["inv_psp"]
        self.valid = data["valid"]
        self.val = data["val"]
