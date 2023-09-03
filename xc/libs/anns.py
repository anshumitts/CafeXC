from xclib.utils.shortlist import Shortlist as shorty
# from xclib.utils.shortlist import ShortlistInstances as shorty
from .utils import map_one, pbar
from .cluster import cluster
from .dataparallel import DataParallel
try:
    from .diskann import DiskANN
except Exception as e:
    print(f"{e}: DiskANN is not installed correctly")
from xclib.utils import sparse as xs
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import scipy.sparse as sp
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
                              k=min(K, self.lbl_emb.shape[1]))
        if self.remapped is not None:
            ind = self.remapped[ind]
        return scr, ind
    
    def _extra_repr_(self):
        return f"vect:{self.lbl_emb.shape}: remap:{self.remapped.size}"


def get_sub_data(doc_vect, label_idx):
    if len(label_idx) == 0:
        return None
    
    if label_idx.size == doc_vect.shape[0]:
        return doc_vect
    
    return doc_vect[label_idx]


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


def set_anns(method, M):
    if method == "hnsw":
        return shorty(method=method, num_neighbours=3*M, M=M, efS=3*M, efC=3*M)
    if method == "diskann":
        return DiskANN({"L": 3*M, "R": M, "C": 7*M, "alpha": 1.2,
                        "saturate_graph": False, "num_threads": 128})


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

    def fit(self, docs_emb=None, lbls_emb=None, y_mat=None, *args, **kwargs):
        if self.use_hnsw == False:
            self.lbls_emb = DataParallel(OvAModule(lbls_emb))
    
    def cuda(self):
        if self.lbls_emb is not None:
            self.lbls_emb.cuda()
    
    def cpu(self):
        if self.lbls_emb is not None:
            self.lbls_emb.cpu()
            self.lbls_emb.callback(clean=True)

    def fit_anns(self, vect, smat=None, most_freq_items=[],
                 num_splits=1, model_path="", *args, **kwargs):
        if smat is not None:
            vect = np.asarray(smat.dot(vect))
        vect = normalize(vect)
        self.most_freq_items = most_freq_items
        lbl_idx = np.arange(vect.shape[0])

        if len(most_freq_items) > 0:
            _vect = get_sub_data(vect, most_freq_items)
            self.lbls_emb = OvAModule(_vect, most_freq_items)
            self.lbls_emb = DataParallel(self.lbls_emb)
            
        id_lbl = np.setdiff1d(lbl_idx, most_freq_items)
        id_lbl = np.array_split(id_lbl, num_splits)
        self.num_splits = len(id_lbl)

        for split in tqdm.tqdm(range(self.num_splits)):
            self.hnsw[f"part_{split}"] = set_anns(self.method, self.M)
            _mapd = id_lbl[split]
            _vect = get_sub_data(vect, _mapd)
            path = os.path.join(model_path, f"part_{split}")
            if self.method == "hnsw":
                self.hnsw[f"part_{split}"].fit(_vect)
            else:
                self.hnsw[f"part_{split}"].fit(_vect, path,
                                            *args, **kwargs)
            self.mapd[f"part_{split}"] = _mapd
    
    def get_index(self, sparse=True):
        keys = list(self.hnsw.keys())
        mat = self.hnsw[keys[0]].get_index(sparse=sparse)[1]
        rows, cols = mat.nonzero()
        rows = [self.mapd[keys[0]][rows]]
        cols = [self.mapd[keys[0]][cols]]
        
        for key in keys[1:]:
            mat = self.hnsw[key].get_index(sparse=sparse)[1]
            _rows, _cols = mat.nonzero()
            rows.append(self.mapd[key][_rows])
            cols.append(self.mapd[key][_cols])
        
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        return sp.csr_matrix((np.ones(rows.size), (rows, cols)),
                             shape=(self.num_labels, self.num_labels))
    
    def get_vect(self):
        keys = list(self.hnsw.keys())
        mat = self.hnsw[keys[0]].get_vect()
        indexes = [self.mapd[keys[0]]]
        mats = [mat]
        for key in keys[1:]:
            mat = self.hnsw[key].get_vect()
            mats.append(mat)
            indexes.append(self.mapd[key])
        indexes = np.concatenate(indexes)
        mats = np.vstack(mats)[np.argsort(indexes)]
        return mats
    
    def partial_build(self, index, vect, model_path):
        keys = list(self.hnsw.keys())
        if len(keys) == 1:
            path = os.path.join(model_path, keys[0])
            self.hnsw[keys[0]].update(index, None, vect, path)
            return
        for key in keys:
            path = os.path.join(model_path, key)
            _vect = vect[self.mapd[key]]
            _index = index[self.mapd[key]].tocsc()[:, self.mapd[key]]
            self.hnsw[key].update(_index, None, _vect, path)

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
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.mapd = pickle.load(f)
                    self.most_freq_items = self.mapd["most_freq_items"]
                self.num_splits = len(self.mapd.keys()) - 1
                if len(self.most_freq_items) > 0:
                    path = os.path.join(model_dir, f"ova_module.pkl")
                    self.lbls_emb = torch.load(path)
            else:
                print("anns_m1_map not found")
                self.num_splits = 1
                self.mapd["part_0"] = np.arange(self.num_labels)
            for split in range(self.num_splits):
                key = f"part_{split}"
                path = os.path.join(model_dir, f"{key}")
                self.hnsw[key] = set_anns(self.method, self.M)
                self.hnsw[key].load(path)
                if self.method == "hnsw":
                    self.hnsw[key].index._set_query_time_params(efS=3*self.M)
                if self.method == "diskann":
                    self.hnsw[key]._set_query_time_params(self.M)
