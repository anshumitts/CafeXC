from xc.libs.dataset import *
from xc.libs.custom_dtypes import BatchData, DataParallelList, scatter, padded_inputs, pre_split
from xc.libs.utils import ScoreEdges, normalize, pbar
from xc.libs.cluster import partial_cluster, cluster
from torch.utils.data import Dataset
import xclib.utils.sparse as xs
import scipy.sparse as sp
import numpy as np
import copy
import torch
choice = np.random.default_rng().choice


class OnlyData(DatasetBase):
    def __init__(self, X, Y, mode="test"):
        super().__init__()
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
        return idx
        
    def get_fts(self, idx):
        return self.X.get_fts(idx)

    def __len__(self):
        return len(self.X)

    def blocks(self, shuffle=False):
        return self.X.blocks(shuffle)

    @property
    def type_dict(self):
        return self.X.type_dict
    
    def collate_fn(self, batch):
        d_batch = BatchData({})
        doc_ids = np.asarray(batch)
        d_batch['b_size'] = len(doc_ids)
        d_batch['idx'] = doc_ids.reshape(-1, 1)
        d_batch["docs"] = self.X.get_fts(doc_ids)
        if self.mode == "train":
            d_batch["Y"] = self.Y.get_fts(doc_ids)
        return d_batch



def SiameseData(X, L, Y, multi_pos=1, mode="test", doc_first=False):
    if doc_first:
        return SiameseDataDOCFirst(X, L, Y, multi_pos, mode)
    return SiameseDataLBLFirst(X, L, Y, multi_pos, mode)


class SiameseDataLBLFirst(DatasetBase):
    def __init__(self, X, L, Y, multi_pos=1, mode="test"):
        super().__init__()
        self.mode = mode
        self.Y = Y
        self.L = L
        self.X = X
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
        self.valid_items = self.Y.valid_lbls
        self.gt_rows = self.gt.T.tocsr()
        self.order = self.valid_items.reshape(-1, 1)
        self.shortlist = None

    def __getitem__(self, lidx):
        return lidx

    def get_gt(self, d_idx, l_idx):
        return self.gt_rows[l_idx].tocsc()[:, d_idx].todense()

    def get_fts(self, d_idx, l_idx=None):
        docs = self.X.get_fts(d_idx)
        lbls = self.L.get_fts(l_idx)
        return BatchData({"docs": docs, "lbls": lbls})

    def __len__(self):
        return len(self.valid_items)

    def callback_(self, lbl_xf=None, doc_xf=None, params=None):
        self.hard_pos = params.hard_pos
        if self.hard_pos:
            self.pos_scoring = ScoreEdges(self.gt_rows.data,
                                          lbl_xf, doc_xf,
                                          params.batch_size)
        lbl_xf = lbl_xf[self.valid_items]
        order = partial_cluster(lbl_xf, params.min_leaf_sz,
                                torch.cuda.device_count())
        self.order = [self.valid_items[x] for x in order]

    def blocks(self, shuffle=False):
        # NOTE Assuming lbls are always sorted
        if shuffle:
            np.random.shuffle(self.order)
        if isinstance(self.order, list):
            return np.concatenate(self.order).flatten()
        return self.order

    @property
    def type_dict(self):
        return self.X.type_dict
    
    def sample(self, idx, size=1):
        hpos = []
        if self.hard_pos:
            c = self.pos_scoring[idx]
            i, p = c.indices, c.data / c.data.sum()
            hpos.append(choice(i, size=self.n_pos, p=p))
        hpos = np.asarray(hpos)
        return list(map(lambda x: choice(self.gt_rows[x].indices), idx)), idx, hpos
    
    def hard_pos_fts(self, idx):
        return self.X.get_fts(idx)
    
    def collate_fn(self, _idx):
        d_batch = BatchData({})    
        d_idx, l_idx, hp_idx = self.sample(_idx)
        y = self.get_gt(d_idx, l_idx)
        d_batch["docs"] = self.X.get_fts(np.int32(d_idx))
        d_batch["lbls"] = self.L.get_fts(np.int32(l_idx))
        val_keys = ["docs", "lbls"]
        
        d_batch["hard_pos"] = None
        if hp_idx.size > 0:
            mask = np.ones_like(hp_idx)
            shape = mask.shape
            hard_items, hard_index = np.unique(hp_idx, return_inverse=True)
            hard_index = hard_index.reshape(*shape)
            d_batch["hard_pos"] = self.hard_pos_fts(np.int32(hard_items))
            hard_index = torch.from_numpy(hard_index).type(torch.LongTensor)
            mask = torch.from_numpy(mask).type(torch.FloatTensor)
            d_batch["hard_pos_index"] = hard_index
            d_batch["hard_pos_mask"] = mask
            val_keys.append("hard_pos")
        d_batch = pre_split(d_batch, val_keys, self.num_splits)
        d_batch["Y"] = torch.from_numpy(y).type(torch.FloatTensor)
        return d_batch


class SiameseDataDOCFirst(SiameseDataLBLFirst):

    def setup(self):
        self.valid_items = self.Y.transpose().valid_lbls
        self.gt_rows = self.Y
        self.order = np.ravel(self.valid_items)
        self.shortlist = None

    def callback_(self, lbl_xf=None, doc_xf=None, params=None):
        super().callback_(doc_xf, lbl_xf, params)
    
    def get_gt(self, d_idx, l_idx):
        return self.gt_rows[d_idx].tocsc()[:, l_idx].todense()
    
    def hard_pos_fts(self, idx):
        return self.L.get_fts(idx)
    
    def sample(self, idx, size=1):
        lbl, doc, hpos = super().sample(idx, size)
        return doc, lbl, hpos


class CrossAttention(DatasetBase):
    def __init__(self, X, L, Y, S, num_p, num_n, mode="test"):
        super().__init__()
        self.mode = mode
        self.Y = Y
        self.S = S
        self.X = X
        self.L = L
        self.p = num_p
        self.n = num_n
        self.random_neg = num_n
        self.num_lbs = self.gt.shape[1]
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
        self.order = np.arange(len(self))

    @property
    def module2(self):
        return self.S.data

    @property
    def gt(self):
        return self.Y.data

    def get_samples(self, d_idx, yhat, gt):
        _sh = yhat[d_idx]
        _gt = gt[d_idx]
        pos = _gt.multiply(_sh)
        neg = _sh - pos
        _hard_neg = xs.retain_topk(neg, k=1)
        neg = neg - _hard_neg
        pos.eliminate_zeros()
        neg.eliminate_zeros()
        pos.data[:] = 1.01 - pos.data[:]
        neg.data[:] = 1.01 + neg.data[:]
        # p_rows, p_cols = pos.nonzero()
        # _gt[p_rows, p_cols] = 0
        # pos = _gt
        return pos.tolil(), neg.tolil(), self.gt[d_idx].tolil()
        
    def __getitem__(self, didx):
        return didx

    @property
    def shape(self):
        return self.Y.data.shape
    
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
            np.random.shuffle(self.order)
        if isinstance(self.order, list):
            return np.concatenate(self.order).flatten()
        return self.order

    @property
    def type_dict(self):
        return self.X.type_dict
    
    def lbls_unique(self, lbls, lb_idx):
        if lbls is None:
            return None
        if isinstance(lbls, BatchData):
            data = BatchData({})
            main_keys = ["txt", "img"]
            for key in main_keys:
                _dat = lbls[key]
                if _dat is not None:
                    data[key] = lbls[key][lb_idx]
                else:
                    data[key] = None
            return data
        else:
            raise NotImplementedError(f"{type(lbls)} not found")
    
    def random_select(self, pos, neg, return_pos = False):
        num_docs = pos.shape[0]
        p_i, p_p = pos.rows, pos.data
        n_i, n_p = neg.rows, neg.data
        pos_lbls = []
        index = np.zeros((num_docs, self.n + self.p))
        masks = np.zeros((num_docs, self.n + self.p))
        for idx, (_p_i, _p_p, _n_i, _n_p) in enumerate(zip(p_i, p_p, n_i, n_p)):
            items = choice(self.num_lbs, size=self.n + self.p)
            if len(_n_p) > 0:
                _n_p = np.exp(np.array(_n_p))
                _n_p /= _n_p.sum()
                _n_i = choice(_n_i, size=self.n)
                items[:self.n] = _n_i.tolist()
            if len(_p_p) > 0:
                _p_p = np.exp(np.array(_p_p))
                _p_p /= _p_p.sum()
                _p_i = choice(_p_i, size=self.p)
                items[-self.p:] = _p_i[:]
                pos_lbls.extend(_p_i[:])
            
            masks[idx, :len(items)] = 1
            index[idx, :len(items)] = items[:]
            
        idx = np.arange(num_docs).reshape(-1, 1)
        index = torch.from_numpy(index).type(torch.LongTensor)
        masks = torch.from_numpy(masks).type(torch.FloatTensor)
        if return_pos:
            return index, masks, np.asarray(pos_lbls)
        return index, masks

    def split(self, d_idx, index, X, L):
        content = BatchData({})
        content["docs"] = X.get_fts(d_idx)
        content["index"] = index
        if self.num_splits == 1:
            l_idx, index = torch.unique(content["index"], return_inverse=True)
            content["lbls"] = L.get_fts(l_idx)
            content["index"] = index
            content["u_lbl"] = l_idx
            return content
        contents = []
        for args in scatter(content, self.num_splits):
            l_idx, index = torch.unique(args["index"], return_inverse=True)
            args["lbls"] = L.get_fts(l_idx)
            args["index"] = index
            args["u_lbl"] = l_idx
            contents.append(BatchData(args))
        return DataParallelList(contents)

    def collate_fn(self, _idx):
        d_batch = BatchData({})
        pos, neg, Y = self.get_samples(np.int32(_idx), self.module2, self.gt)
        d_batch["index"], d_batch["masks"] = self.random_select(pos, neg)
        _Y = np.take_along_axis(Y, d_batch["index"].numpy(), 1).todense()    
        d_batch["Y"] = torch.from_numpy(_Y).type(torch.FloatTensor)
        d_batch["content"] = self.split(_idx, d_batch["index"], self.X, self.L)
        return d_batch


class RankerPredictDataset(DatasetBase):
    def __init__(self, X, L, shorty, mode="test"):
        super().__init__()
        self.X = X
        self.L = L
        self.S = shorty
        self.order = np.arange(len(self))

    def __getitem__(self, didx):
        return didx

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
    
    def build_shorty(self, shorty):
        lblIdx = np.unique(shorty.indices)
        shorty = shorty.tocsc()[:, lblIdx].tocsr()
        shorty = padded_inputs(shorty)
        return shorty["index"], shorty["mask"], lblIdx
    
    def split(self, d_idx, index):
        content = BatchData({})
        content["docs"] = self.X.get_fts(d_idx)
        content["index"] = index
        if self.num_splits == 1:
            l_idx, index = torch.unique(content["index"], return_inverse=True)
            content["lbls"] = self.L.get_fts(l_idx)
            content["index"] = index
            content["u_lbl"] = l_idx
            return content
        contents = []
        for args in scatter(content, self.num_splits):
            l_idx, index = torch.unique(args["index"], return_inverse=True)
            args["lbls"] = self.L.get_fts(l_idx)
            args["index"] = index
            args["u_lbl"] = l_idx
            contents.append(BatchData(args))
        return DataParallelList(contents)
    
    def collate_fn(self, doc_ids):
        d_batch = BatchData({})
        shorty = self.module2[doc_ids]
        index, mask, lbl_ids = self.build_shorty(shorty)
        doc_ids = np.int32(doc_ids)
        lbl_ids = np.int32(lbl_ids)
        lbl_ids = torch.from_numpy(lbl_ids).type(torch.LongTensor)
        d_batch["index"] = lbl_ids[index]
        d_batch["content"] = self.split(doc_ids, d_batch["index"])
        d_batch["mask"] = mask
        return d_batch
