from sklearn.preprocessing import normalize
from .utils import load_file
import scipy.sparse as sp
import numpy as np
import torch
import os


class LBLDataset(torch.utils.data.Dataset):
    def __init__(self, root, n_file):
        if isinstance(n_file, str):
            if n_file.endswith("eye"):
                self.data = sp.eye(int(n_file.split(".")[0])).tocsr()
            else:
                self.data = load_file(os.path.join(root, f"{n_file}"))
        elif sp.issparse(n_file):
            self.data = n_file

    def __getitem__(self, idx):
        return self.data[idx]

    def normalize(self):
        self.data = normalize(self.data)

    def __len__(self):
        return self.data.shape[0]

    @property
    def valid_lbls(self):
        return np.where(np.ravel(self.data.getnnz(axis=0)) > 0)[0]

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
    
    def compress(self, clusters):
        self.data = self.data.dot(clusters).tocsr()
        self.binarize()
    
    def binarize(self):
        self.data.data[:] = 1
    
    def vstack(self, obj):
        self.data = sp.vstack([self.data, obj.data], 'csr')
    
    def hstack(self, obj):
        self.data = sp.hstack([self.data, obj.data], 'csr')
    
    def transpose(self):
        return LBLDataset("", self.data.transpose().tocsr())
    
    @property
    def shape(self):
        return self.data.shape
