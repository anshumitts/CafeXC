from sklearn.preprocessing import normalize
from .utils import load_file
import xclib.utils.sparse as xs
import scipy.sparse as sp
import numpy as np
import torch
import os



class SHORTYDataset(torch.utils.data.Dataset):
    def __init__(self, root, n_file):
        if isinstance(n_file, str):
            self.data = load_file(os.path.join(root, f"{n_file}"))
        elif sp.issparse(n_file):
            self.data = n_file

    def keep_topk(self, topk=100):
        self.data = xs.retain_topk(self.data, k=topk)
        print(f"LIBS:SHORTLIST:NNZ:{self.data.nnz}")

    def keep_negatives(self, lbl_y):
        print(f"LIBS:SHORTLIST:NNZ:BEFORE: {self.data.nnz}")
        mask = self.data.copy()
        lbl_y = lbl_y.copy()
        lbl_y.data[:] = 1
        mask = mask + lbl_y
        mask.data[:] = 1
        mask = mask - lbl_y
        mask.eliminate_zeros()
        self.data = self.data.multiply(mask)
        self.data.eliminate_zeros()
        print(f"LIBS:SHORTLIST:NNZ:AFTER: {self.data.nnz}")

    def __getitem__(self, idx):
        return self.data[idx]

    def normalize(self):
        self.data = normalize(self.data)

    def __len__(self):
        return self.data.shape[0]

    @property
    def valid(self):
        return np.where(np.ravel(self.data.getnnz(axis=1)) > 0)[0]

    @property
    def num_labels(self):
        return self.data.shape[1]

    def keep_valid(self, indices):
        self.data = self.data[indices]

    def padd(self):
        self.data = sp.hstack([self.data,
                               sp.lil_matrix(len(self), 1)]).tocsr()
        print("Adding padding index in the last")

    def get_fts(self, idx, desc="blah"):
        return self[idx]

    def filter(self, valid_pts, axis=0):
        if axis == 1:
            s_mat = sp.lil_matrix((self.num_labels, valid_pts.size))
            s_mat[valid_pts, np.arange(valid_pts.size)] = 1
            self.data = self.data.dot(s_mat).tocsr()
        if axis == 0:
            self.data = self.data[valid_pts]
    
    def vstack(self, obj):
        self.data = sp.vstack([self.data, obj.data], 'csr')
