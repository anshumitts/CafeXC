from xc.libs.custom_dtypes import (BatchData, DataParallelList, pre_split,
                                   scatter, padded_inputs, MultiViewData)
from xc.libs.collate_fn import collateBase
import scipy.sparse as sp
import numpy as np
import torch
choice = np.random.default_rng().choice


class SurrogateCollate(collateBase):
    def __init__(self, dset, params):
        super().__init__(dset)
        self.doc_first = params.doc_first

    def build_content(self, doc_ids, lbl_ids, b_dict):
        b_dict['b_size'] = len(doc_ids)
        data = self.dset.get_fts(doc_ids, lbl_ids)
        b_dict["docs"] = data["docs"]
        b_dict["lbls"] = data["lbls"]

    def call_batch_doc(self, batch):
        d_batch = BatchData({})
        hard_pos, docs_ids, rand_pos = [], [], []
        for data in batch:
            hard_pos.append(data["hard_pos"])
            rand_pos.extend(data["rand_pos"])
            docs_ids.append(data["d_idx"])
        rand_pos = np.uint32(rand_pos)
        docs_ids = np.uint32(docs_ids)
        y = self.dset.gt_rows[docs_ids]
        hard_pos = np.int32(np.vstack(hard_pos))
        keys = ["docs", "lbls"]
        d_batch["hard_pos"] = None
        if hard_pos.size > 0:
            mask = np.ones_like(hard_pos)
            mask[hard_pos == -1] = 0
            hard_pos[mask == 0] = 0
            shape = mask.shape
            hard_items, hard_index = np.unique(hard_pos, return_inverse=True)
            hard_index = hard_index.reshape(*shape)
            d_batch["hard_pos"] = self.dset.L.get_fts(hard_items)
            hard_index = torch.from_numpy(hard_index).type(torch.LongTensor)
            mask = torch.from_numpy(mask).type(torch.FloatTensor)
            d_batch["hard_pos_index"] = hard_index
            d_batch["hard_pos_mask"] = mask
            keys.append("hard_pos")

        if docs_ids.size > 1:
            d_batch["docs"] = self.dset.X.get_fts(docs_ids)
            d_batch["lbls"] = self.dset.L.get_fts(rand_pos)
        d_batch = pre_split(d_batch, keys, self.num_splits)
        y = y.tocsc()[:, rand_pos].todense()
        d_batch["Y"] = torch.from_numpy(y).type(torch.FloatTensor)
        return d_batch

    def call_batch_lbl(self, batch):
        d_batch = BatchData({})
        hard_pos, lbls_ids, rand_pos = [], [], []
        for data in batch:
            hard_pos.append(data["hard_pos"])
            rand_pos.extend(data["rand_pos"])
            lbls_ids.append(data["l_idx"])
        rand_pos = np.uint32(rand_pos)
        lbls_ids = np.uint32(lbls_ids)
        y = self.dset.gt_rows[lbls_ids]
        hard_pos = np.int32(np.vstack(hard_pos))
        keys = ["docs", "lbls"]
        d_batch["hard_pos"] = None
        if hard_pos.size > 0:
            mask = np.ones_like(hard_pos)
            mask[hard_pos == -1] = 0
            hard_pos[mask == 0] = 0
            shape = mask.shape
            hard_items, hard_index = np.unique(hard_pos, return_inverse=True)
            hard_index = hard_index.reshape(*shape)
            d_batch["hard_pos"] = self.dset.X.get_fts(hard_items)
            hard_index = torch.from_numpy(hard_index).type(torch.LongTensor)
            mask = torch.from_numpy(mask).type(torch.FloatTensor)
            d_batch["hard_pos_index"] = hard_index
            d_batch["hard_pos_mask"] = mask
            keys.append("hard_pos")

        if lbls_ids.size > 1:
            d_batch["docs"] = self.dset.X.get_fts(rand_pos)
            d_batch["lbls"] = self.dset.L.get_fts(lbls_ids)
        d_batch = pre_split(d_batch, keys, self.num_splits)
        y = y.tocsc()[:, rand_pos].todense()
        d_batch["Y"] = torch.from_numpy(y).type(torch.FloatTensor)
        return d_batch

    def _call_(self, batch):
        if self.doc_first:
            return self.call_batch_doc(batch)
        return self.call_batch_lbl(batch)


class SampleRankerCollate(SurrogateCollate):
    def __init__(self, dset, params):
        super().__init__(dset, params)
        self.num_lbls = self.dset.shape[1]
        self.p = params.sample_pos
        self.n = params.sample_neg
        self.use_lbls = "XAttnRanker" in params.ranker

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
        elif isinstance(lbls, MultiViewData):
            return lbls[lb_idx]
        else:
            raise NotImplementedError(f"{type(lbls)} not found")

    def build_content(self, doc_ids, lbl_ids, index, debug=False):
        content = BatchData({})
        data = self.dset.get_fts(doc_ids, lbl_ids)
        if not self.use_lbls:
            data["lbls"] = None
        content["u_doc"] = torch.from_numpy(doc_ids)
        content["u_doc"] = content["u_doc"].type(torch.LongTensor)
        content["docs"] = data["docs"]
        content["lbls"] = data["lbls"]
        if debug:
            print("DOCS", content["docs"])
            print("LBLS", content["lbls"])
            print("INDEX", index.shape)
        content["index"] = index
        lbl_ids = torch.from_numpy(lbl_ids).type(torch.LongTensor)
        if self.num_splits == 1:
            content["u_lbl"] = lbl_ids
            return content
        contents = []
        for args in scatter(content, self.num_splits):
            u_lbl, mapping = torch.unique(args["index"], return_inverse=True)
            args["lbls"] = self.lbls_unique(content["lbls"], u_lbl)
            args["index"] = mapping
            args["u_lbl"] = lbl_ids[u_lbl]
            contents.append(BatchData(args))
        return DataParallelList(contents)

    def build_shorty(self, shorty):
        lblIdx = np.unique(shorty.indices)
        shorty = shorty.tocsc()[:, lblIdx].tocsr()
        shorty = padded_inputs(shorty)
        return shorty["index"], shorty["mask"], lblIdx

    def random_pos_neg(self, pos, neg, lbl_shorty, num_p, num_n, num_c=4):
        lbl_shorty = lbl_shorty.tocsc()
        # common_neg = choice(self.num_lbls, size=num_c).tolist()
        common_neg = []
        p_indptr, p_index, p_data = pos.indptr, pos.indices, pos.data
        n_indptr, n_index, n_data = neg.indptr, neg.indices, neg.data
        smx, num_docs = 0, pos.shape[0]
        index = np.zeros((num_docs, num_n+num_p+num_c))
        masks = np.zeros((num_docs, num_n+num_p+num_c))
        for idx in np.arange(num_docs):
            _p_s = slice(p_indptr[idx], p_indptr[idx+1])
            _n_s = slice(n_indptr[idx], n_indptr[idx+1])
            _p_i, _p_p = p_index[_p_s], p_data[_p_s]
            _n_i, _n_p = n_index[_n_s], n_data[_n_s]

            size_p, size_n = min(num_p, len(_p_i)), min(num_n, len(_n_i))
            items = list(common_neg)
            if size_n > 0:
                _n_p = np.exp(np.array(_n_p))
                _n_p /= _n_p.sum()
                _n_i = choice(_n_i, size=size_n, replace=False, p=_n_p)
                items.extend(_n_i.tolist())
            if size_p > 0:
                _p_p = np.exp(np.array(_p_p))
                _p_p /= _p_p.sum()
                _p_i = choice(_p_i, size=size_p, replace=False, p=_p_p)
                items.extend(_p_i.tolist())
            _len = len(items)
            masks[idx, :_len] = 1
            index[idx, :_len] = items[:]
            smx = max(_len, smx)

        idx = np.arange(num_docs).reshape(-1, 1)
        index, masks = index[:, :smx], masks[:, :smx]
        lblys = np.asarray(lbl_shorty[idx, index].todense())
        index = torch.from_numpy(index).type(torch.LongTensor)
        masks = torch.from_numpy(masks).type(torch.FloatTensor)
        l_idx, index = torch.unique(index, return_inverse=True)
        return index, masks, l_idx, lblys

    def _call_(self, batch):
        d_batch = BatchData({})
        doc_ids = list(map(lambda x: x["d_idx"], batch))
        pos, neg, lbl = self.dset.get_samples(doc_ids)
        index, mask, lbl_ids, Y = self.random_pos_neg(
            pos, neg, lbl, self.p, self.n)
        d_batch["Y"] = torch.from_numpy(Y).type(torch.FloatTensor)
        doc_ids = np.int32(doc_ids)
        lbl_ids = np.int32(lbl_ids)
        d_batch["content"] = self.build_content(doc_ids, lbl_ids, index)
        v_lbl = torch.from_numpy(lbl_ids).type(torch.LongTensor)
        d_batch["index"] = v_lbl[index]
        d_batch["mask"] = mask
        return d_batch


class RankerCollate(SampleRankerCollate):
    def __init__(self, dset, params):
        super().__init__(dset, params)

    def _call_(self, batch):
        d_batch = BatchData({})
        doc_ids = []
        for data in batch:
            doc_ids.append(data["d_idx"])
        shorty = self.dset.module2[doc_ids]
        index, mask, lbl_ids = self.build_shorty(shorty)
        doc_ids = np.int32(doc_ids)
        lbl_ids = np.int32(lbl_ids)
        d_batch["content"] = self.build_content(doc_ids, lbl_ids, index)
        lbl_ids = torch.from_numpy(lbl_ids).type(torch.LongTensor)
        d_batch["index"] = lbl_ids[index]
        d_batch["mask"] = mask
        return d_batch


class DocumentCollate(collateBase):
    def __init__(self, dset, params):
        super().__init__(dset)

    def build_docs(self, doc_ids, b_dict):
        b_dict['b_size'] = doc_ids.size
        b_dict['idx'] = doc_ids.reshape(-1, 1)
        b_dict["docs"] = self.dset.get_fts(doc_ids)

    def _call_(self, batch):
        d_batch = BatchData({})
        doc_ids = np.int32(list(map(lambda x: x['x'], batch)))
        self.build_docs(doc_ids, d_batch)
        return d_batch
