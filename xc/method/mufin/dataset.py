from xc.libs.dataset import *
from xc.libs.utils import ScoreEdges, normalize, pbar
from xc.libs.custom_dtypes import BatchData
from torch.utils.data import Dataset
from xc.libs.cluster import cluster
import xclib.utils.sparse as xs
import scipy.sparse as sp
import numpy as np
import copy
choice = np.random.default_rng().choice


class OnlyData(Dataset):
    def __init__(self, X, Y, mode="test"):
        self.multi = 1
        self.mode = mode
        self.Y = Y
        self.X = X

    def split_dataset(self, arg):
        split = copy.deepcopy(self)
        split.filter_rows(arg)
        return split

    def filter_rows(self, valid_idx):
        self.X.filter(valid_idx, axis=0)
        if self.Y is not None:
            self.Y.filter(valid_idx, axis=0)

    @property
    def gt(self):
        return self.Y.data

    def compress(self, clusters):
        self.Y.compress(clusters)

    def __getitem__(self, idx):
        if self.mode == "test":
            y = None
        else:
            y = self.Y.get_fts(idx)
        return {"x": idx, "y": y}

    def get_fts(self, idx):
        return self.X.get_fts(idx)

    def __len__(self):
        return len(self.X)

    def blocks(self, shuffle=False):
        return self.X.blocks(shuffle)

    @property
    def type_dict(self):
        return self.X.type_dict


class SiameseData(Dataset):
    def __init__(self, X, L, Y, multi_pos=1, mode="test"):
        self.mode = mode
        self.Y = Y
        self.L = L
        self.X = X
        self.doc_gph = None
        self.lbl_gph = None
        self.setup()
        self.hard_pos = False
        self.n_pos = multi_pos

    @property
    def gt(self):
        return self.Y.data

    def add_to_gt(self, docs, lbls):
        self.Y.data[docs, lbls] = 1
        self.setup()

    def setup(self):
        self.valid_lbls = self.Y.valid_lbls
        self.gt_rows = self.gt.T.tocsr()
        self.order = self.valid_lbls.reshape(-1, 1)
        self.shortlist = None

    def __getitem__(self, lidx):
        y = self.gt_rows[lidx]
        rpos, hpos = choice(y.indices, size=1), []
        if self.hard_pos:
            c = self.pos_scoring[lidx]
            i, p = c.indices, c.data / c.data.sum()
            hpos = choice(i, size=self.n_pos, p=p).reshape(1, self.n_pos)
        return {"hard_pos": hpos, "rand_pos": rpos, "l_idx": lidx, "y": y}

    def get_fts(self, d_idx, l_idx=None):
        docs = self.X.get_fts(d_idx)
        lbls = self.L.get_fts(l_idx)
        return BatchData({"docs": docs, "lbls": lbls})

    def __len__(self):
        return len(self.X)

    def callback_(self, lbl_xf=None, doc_xf=None, params=None):
        self.hard_pos = params.hard_pos
        if self.hard_pos:
            self.pos_scoring = ScoreEdges(self.gt_rows, lbl_xf, doc_xf,
                                          params.batch_size)
        lbl_xf = lbl_xf[self.valid_lbls]
        self.order = cluster(lbl_xf, params.min_leaf_sz,
                             params.min_splits, force_gpu=True)
        self.order = [self.valid_lbls[x] for x in self.order]

    def blocks(self, shuffle=False):
        # NOTE Assuming lbls are always sorted
        if shuffle:
            idx = np.arange(len(self.order))
            np.random.shuffle(idx)
            lbls = list(map(lambda x: self.order[x], idx))
            lbls = np.concatenate(lbls).flatten()
        else:
            lbls = np.concatenate(self.order)
        return lbls

    @property
    def type_dict(self):
        return self.X.type_dict


class CrossAttention(Dataset):
    def __init__(self, X, L, Y, S, mode="test"):
        self.mode = mode
        self.Y = Y
        self.S = S
        self.X = X
        self.L = L
        self.setup()

    def split_dataset(self, arg):
        split = copy.deepcopy(self)
        split.filter_rows(arg)
        split.setup()
        return split

    def filter_rows(self, valid_idx):
        self.X.filter(valid_idx, axis=0)
        self.Y.filter(valid_idx, axis=0)
        self.S.filter(valid_idx, axis=0)

    def setup(self):
        self.order = np.arange(len(self)).reshape(-1, 1)
        if self.mode == "train":
            self.setup_proba(self.module2)

    def setup_proba(self, shorty, batch_size=1024):
        self.pos, self.neg = self.clean_shortlist(shorty, self.gt, batch_size)

    def clean_shortlist(self, shorty, gt, batch_size=1024):
        pos_sh, neg_sh = [], []
        indexes = np.arange(0, shorty.shape[0], batch_size)
        for start in pbar(indexes, desc="Cleaning"):
            end = min(start+batch_size, shorty.shape[0])
            _sh = shorty[start:end].tolil()
            _gt = gt[start:end].tolil()
            gt_rows, gt_cols = _gt.nonzero()
            pos = sp.lil_matrix(_sh.shape)
            pos[gt_rows, gt_cols] = np.ravel(_sh[gt_rows, gt_cols].todense())
            _sh[gt_rows, gt_cols] = 0
            neg = _sh.tocsr()
            pos = pos.tocsr()
            _hard_neg = xs.retain_topk(neg, k=1)
            _hard_neg.data[_hard_neg.data[:] < 0.95] = 0
            neg = neg - _hard_neg
            pos.eliminate_zeros()
            neg.eliminate_zeros()
            pos.data[:] = 1.001 - pos.data[:]
            neg.data[:] = 1.001 + neg.data[:]
            p_rows, p_cols = pos.nonzero()
            _gt[p_rows, p_cols] = 0
            _gt_min = pos.min(axis=0)/1.5
            pos = pos + _gt.multiply(_gt_min)
            pos_sh.append(pos)
            neg_sh.append(neg)

            del _gt, _hard_neg, _sh
        pos = normalize(sp.vstack(pos_sh, "csr"), "l1").tocsr()
        neg = normalize(sp.vstack(neg_sh, "csr"), "l1").tocsr()
        print(f"#shorty: {shorty.nnz}")
        print(f"\t #pos:{pos.nnz} + #neg:{neg.nnz} = {pos.nnz+neg.nnz}")
        return pos, neg

    @property
    def module2(self):
        return self.S.data

    @property
    def gt(self):
        return self.Y.data

    def get_samples(self, doc_idx):
        if self.mode == "train":
            return self.pos[doc_idx], self.neg[doc_idx], self.Y.get_fts(doc_idx)
        return None

    def __getitem__(self, didx):
        return {"d_idx": didx}

    @property
    def shape(self):
        return self.Y.data.shape

    def get_fts(self, d_idx, l_idx=None):
        docs = self.X.get_fts(d_idx)
        lbls = self.L.get_fts(l_idx)
        return BatchData({"docs": docs, "lbls": lbls})

    def __len__(self):
        return len(self.X)

    def callback_(self, doc_xf=None, params=None):
        if doc_xf is not None:
            self.order = cluster(doc_xf, params.min_leaf_sz,
                                 params.min_splits)
        pass

    def blocks(self, shuffle=False):
        # NOTE Assuming lbls are always sorted
        if shuffle:
            idx = np.arange(len(self.order))
            np.random.shuffle(idx)
            docs = np.concatenate(list(map(lambda x: self.order[x], idx)
                                       )).flatten()
        else:
            docs = np.concatenate(self.order)
        return docs

    @property
    def type_dict(self):
        return self.X.type_dict


class RankerPredictDataset(Dataset):
    def __init__(self, X, L, shorty, mode="test"):
        self.X = X
        self.L = L
        self.S = shorty
        self.order = np.arange(len(self))

    def __getitem__(self, didx):
        return {"d_idx": didx}

    @property
    def shape(self):
        return self.S.shape

    @property
    def module2(self):
        return self.S

    def get_fts(self, d_idx, l_idx=None):
        docs = self.X[d_idx]
        lbls = self.L[l_idx]
        return BatchData({"docs": docs, "lbls": lbls})

    def __len__(self):
        return len(self.X)

    def callback_(self, *args, **kwargs):
        pass

    def blocks(self, shuffle=False):
        return self.order

    @property
    def type_dict(self):
        return {"txt": "pretrained", "img": "pretrained"}
