from xclib.utils.shortlist import ShortlistCentroids as shorty
# from xclib.utils.shortlist import ShortlistInstances as shorty
from .utils import map_one, pbar
from .cluster import cluster
from .dataparallel import DataParallel
from xclib.utils import sparse as xs
import torch.nn.functional as F
import numba as nb
import numpy as np
import pickle
import torch
import tqdm
import os


class OvAModule(torch.nn.Module):
    def __init__(self, lbl_emb, remapped=None):
        super().__init__()
        self.remapped = None
        self.lbl_emb = None
        if lbl_emb is not None:
            lbl_emb = F.normalize(torch.from_numpy(lbl_emb)).T
            lbl_emb = lbl_emb.type(torch.FloatTensor)
            self.lbl_emb = torch.nn.Parameter(
                lbl_emb, requires_grad=False)
            if remapped is not None:
                remapped = torch.from_numpy(remapped).type(torch.LongTensor)
                self.remapped = torch.nn.Parameter(remapped, requires_grad=False)
    
    def forward(self, doc, K=5):
        if self.lbl_emb is None:
            return None, None
        doc = F.normalize(doc)
        scr, ind = torch.topk(doc.mm(self.lbl_emb), dim=1,
                              k=min(K, self.lbl_emb.size(1)))
        if self.remapped is not None:
            ind = self.remapped[ind]
        return scr, ind
    
    def extra_repr(self):
        _str = ""
        if self.lbl_emb is not None:
            _str+= f"vect:{self.lbl_emb.shape}"
        if self.remapped is not None:
            _str+= f"remap:{self.remapped.size}"
        return _str


def get_sub_data(doc_vect, smat, label_idx):
    if len(label_idx) == 0:
        return None, None
    
    if label_idx.size == smat.shape[0]:
        return doc_vect, smat
    _smat = smat[:, label_idx]
    valid_docs = np.ravel(_smat.getnnz(axis=1))
    valid_docs = np.where(valid_docs > 0)[0]
    return doc_vect[valid_docs], _smat[valid_docs]


@nb.njit(nb.types.Tuple(
    (nb.int64[:, :], nb.float32[:, :]))(nb.int64[:, :], nb.float32[:, :],
                                        nb.int64, nb.int64, nb.float32), parallel=True)
def map_neighbors(indices, similarity, top_k, pad_ind, pad_val):
    m = indices.shape[0]
    point_labels = np.full(
        (m, top_k), pad_ind, dtype=np.int64)
    point_label_sims = np.full(
        (m, top_k), pad_val, dtype=np.float32)
    for i in nb.prange(m):
        unique_point_labels, point_label_sim = map_one(
            indices[i], similarity[i], pad_ind)
        if top_k < len(unique_point_labels):
            top_indices = np.argsort(
                point_label_sim)[-1 * top_k:][::-1]
            point_labels[i] = unique_point_labels[top_indices]
            point_label_sims[i] = point_label_sim[top_indices]
        else:
            _k = len(unique_point_labels)
            point_labels[i, :_k] = unique_point_labels
            point_label_sims[i, :_k] = point_label_sim
    return point_labels, point_label_sims


def query_anns(anns, ova_module, docs, num_lbls, top_k=100, lbl_hash=None, desc="anns"):
    lbls, distance = anns.query(docs)
    if lbl_hash is not None:
        lbl_hash = np.concatenate((lbl_hash, [num_lbls]))
        lbls = lbl_hash[lbls]
    
    if ova_module is not None:
        ova_dist, ova_lbls = ova_module(torch.from_numpy(docs))
        ova_lbls = ova_lbls.cpu().numpy()
        ova_dist = ova_dist.cpu().numpy()
        lbls  = np.hstack([lbls, ova_lbls])
        distance = np.hstack([distance, ova_dist])
    smat = xs.csr_from_arrays(lbls, distance, (len(docs), num_lbls+1))
    return smat[:, :-1]


class ANNSBox(object):
    def __init__(self, top_k: int = 100, num_labels: int = -1, M: int = 500,
                 use_hnsw: bool = False, method: str = "hnsw"):
        self.num_labels = num_labels
        self.top_k = top_k
        self.use_hnsw = use_hnsw
        self.num_splits = 1
        self.most_freq_items = []
        self.hnsw, self.mapd = {}, {}
        self.M = M
        self.method = method
        self.lbls_emb = None

    def fit(self, docs_emb=None, lbls_emb=None, y_mat=None):
        if self.use_hnsw == False:
            self.lbls_emb = DataParallel(OvAModule(lbls_emb))
    
    def cuda(self):
        if self.lbls_emb is not None:
            self.lbls_emb.cuda()
    
    def cpu(self):
        if self.lbls_emb is not None:
            self.lbls_emb.cpu()
            self.lbls_emb.callback(clean=True)

    def fit_anns(self, vect, smat, most_freq_items=[], num_splits=1):
        if self.use_hnsw:
            self.most_freq_items = most_freq_items
            _vect, _smat = get_sub_data(vect, smat, most_freq_items)
            
            if _vect is not None:
                _vect = _smat.dot(_vect)
            self.lbls_emb = OvAModule(_vect, most_freq_items)
            self.lbls_emb = DataParallel(self.lbls_emb)
            print(self.lbls_emb)
            n_lbls = smat.shape[1]
            lbl_idx = np.arange(n_lbls)
            
            id_lbl = np.setdiff1d(lbl_idx, most_freq_items)
            id_lbl = np.array_split(id_lbl, num_splits)
            self.num_splits = len(id_lbl)
            for split in tqdm.tqdm(range(self.num_splits)):
                self.hnsw[f"part_{split}"] = shorty(
                    method=self.method, num_neighbours=3*self.M,
                    M=self.M, efS=3*self.M, efC=3*self.M)
                print(f"num_lbls = {id_lbl[split].size}")
                _mapd = id_lbl[split]
                _vect, _smat = get_sub_data(vect, smat, _mapd)
                print(_vect.shape, _smat.shape)
                self.hnsw[f"part_{split}"].fit(_vect, _smat)
                self.mapd[f"part_{split}"] = _mapd

    def __call__(self, docs, batch_size=2048):
        return self.brute_query(docs)

    def hnsw_query(self, docs):
        if self.use_hnsw:
            smat = query_anns(self.hnsw[f"part_0"], self.lbls_emb,
                              docs, self.num_labels,
                              self.top_k, self.mapd[f"part_0"])
            for split in range(1, self.num_splits):
                _smat = query_anns(self.hnsw[f"part_{split}"], None,
                                   docs, self.num_labels,
                                   self.top_k, self.mapd[f"part_{split}"])
                smat = smat + _smat
            return smat
        return self.brute_query(docs)

    def brute_query(self, docs):
        return self._exact_search(docs, self.num_labels, self.top_k)

    def _exact_search(self, docs, num_lbls, topk=5):
        scr, ind = self.lbls_emb(torch.from_numpy(docs), K=topk)
        return xs.csr_from_arrays(ind.cpu().numpy(), scr.cpu().numpy(),
                                  (docs.shape[0], num_lbls))
        
    def save(self, model_dir):
        if self.use_hnsw:
            for key in self.hnsw.keys():
                path = os.path.join(model_dir, f"{key}")
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, f"anns_m1")
                self.hnsw[key].save(path)
            path = os.path.join(model_dir, f"anns_m1_map.pkl")
            with open(path, "wb") as f:
                self.mapd["most_freq_items"] = self.most_freq_items
                pickle.dump(self.mapd, f)
            
            if self.lbls_emb is not None:
                path = os.path.join(model_dir, f"ova_module.pkl")
                torch.save(self.lbls_emb, path)

    def load(self, model_dir):
        if self.use_hnsw:
            path = os.path.join(model_dir, f"anns_m1_map.pkl")
            with open(path, "rb") as f:
                self.mapd = pickle.load(f)
                self.most_freq_items = self.mapd["most_freq_items"]
            self.num_splits = len(self.mapd.keys()) - 1
            if len(self.most_freq_items) > 0:
                path = os.path.join(model_dir, f"ova_module.pkl")
                self.lbls_emb = torch.load(path)
                print(self.lbls_emb)

                
            for split in range(self.num_splits):
                key = f"part_{split}"
                path = os.path.join(model_dir, f"{key}/anns_m1")
                self.hnsw[key] = shorty(
                    method=self.method, num_neighbours=3*self.M,
                    M=self.M, efS=3*self.M, efC=3*self.M)
                self.hnsw[key].load(path)
                self.hnsw[key].index._set_query_time_params(efS=3*self.M)

