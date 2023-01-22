from xclib.utils.shortlist import ShortlistCentroids as shorty
# from xclib.utils.shortlist import ShortlistInstances as shorty
from .utils import map_one, pbar
from .cluster import cluster
from xclib.utils import sparse as xs
import torch.nn.functional as F
import numba as nb
import numpy as np
import pickle
import torch
import tqdm
import os


def get_sub_data(doc_vect, smat, label_idx):
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


def query_anns(anns, vect, num_lbls, top_k=100, b_size=256, lbl_hash=None, desc="anns"):
    s_lbls, s_scrs = [], []
    for start in np.arange(0, len(vect), b_size):
        end = min(start+b_size, len(vect))
        lbls, distance = anns.query(vect[start:end])
        s_lbls.append(lbls)
        s_scrs.append(distance)
    s_lbls, s_scrs = np.vstack(s_lbls).astype(np.int64), np.vstack(s_scrs)
    s_lbls, s_scrs = map_neighbors(s_lbls, s_scrs, top_k, num_lbls, -1)
    if lbl_hash is not None:
        lbl_hash = np.concatenate((lbl_hash, [num_lbls]))
        s_lbls = lbl_hash[s_lbls]
    smat = xs.csr_from_arrays(s_lbls, s_scrs, (len(vect), num_lbls+1))
    return smat[:, :-1]


class ANNSBox(object):
    def __init__(self, top_k: int = 100, num_labels: int = -1, M: int = 500,
                 use_hnsw: bool = False, num_splits: int = 1, method: str = "hnsw"):
        self.num_labels = num_labels
        self.top_k = top_k
        self.use_hnsw = use_hnsw
        self.num_splits = num_splits
        self.hnsw, self.mapd = {}, {}
        for i in range(self.num_splits):
            self.hnsw[f"part_{i}"] = shorty(method=method,
                                            num_neighbours=top_k,
                                            M=M)
            self.mapd[f"part_{i}"] = []
        self.docs = np.array([])
        self.lbls_ncc = np.array([])
        self.lbls_emb = np.array([])

    def fit(self, docs_emb=None, lbls_emb=None, y_mat=None):
        if docs_emb is not None:
            self.lbls_ncc = y_mat.T.dot(docs_emb)
            self.docs = docs_emb
        self.lbls_emb = lbls_emb

    def fit_anns(self, vect, smat, clusters=True):
        if self.use_hnsw:
            n_lbls = smat.shape[1]
            id_lbl = np.arange(n_lbls)
            if clusters:
                print("Clustering labels for ANNS")
                id_lbl = cluster(smat.T.dot(
                    vect), num_clusters=self.num_splits)
            else:
                if self.num_splits > 1:
                    np.random.shuffle(id_lbl)
                id_lbl = np.array_split(id_lbl, self.num_splits)
            for split in tqdm.tqdm(range(self.num_splits)):
                print(f"num_lbls = {id_lbl[split].size}")
                _mapd = id_lbl[split]
                _vect, _smat = get_sub_data(vect, smat, _mapd)
                print(_vect.shape, _smat.shape)
                self.hnsw[f"part_{split}"].fit(_vect, _smat)
                self.mapd[f"part_{split}"] = _mapd

    def __call__(self, docs, batch_size=512):
        return self.brute_query(docs, batch_size)

    def hnsw_query(self, docs, batch_size):
        if self.use_hnsw:
            smat = query_anns(self.hnsw[f"part_0"],
                              docs, self.num_labels,
                              self.top_k, batch_size,
                              self.mapd[f"part_0"])
            for split in range(1, self.num_splits):
                _smat = query_anns(self.hnsw[f"part_{split}"],
                                   docs, self.num_labels,
                                   self.top_k, batch_size,
                                   self.mapd[f"part_{split}"])
                smat = _smat + smat
                smat = xs.retain_topk(smat, self.top_k)
                del _smat
            return smat
        return self.brute_query(docs, batch_size)

    def brute_query(self, docs, batch_size):
        return self._exact_search(docs, self.lbls_emb, self.num_labels,
                                  batch_size, self.top_k, "brute_emb")

    def _exact_search(self, docs, lbls, num_lbls,
                      batch_size=512, topk=5, desc="desc"):
        lbls = torch.from_numpy(lbls).type(torch.FloatTensor)
        lbls = F.normalize(lbls, dim=-1).cuda().T
        docs = torch.from_numpy(docs).type(torch.FloatTensor)
        docs = F.normalize(docs, dim=-1).cuda()
        scr, ind = torch.topk(docs.mm(lbls), dim=1, k=topk)
        del lbls
        return xs.csr_from_arrays(ind.cpu().numpy(), scr.cpu().numpy(),
                                  (docs.shape[0], num_lbls))

    def gpu_ova(self, x, y, batch_size=512):
        return self._exact_search(x, y, self.num_labels,
                                  batch_size, self.top_k, "GPU_OVA")

    def save(self, model_dir):
        if self.use_hnsw:
            for key in self.hnsw.keys():
                path = os.path.join(model_dir, f"{key}")
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, f"anns_m1")
                self.hnsw[key].save(path)
            path = os.path.join(model_dir, f"anns_m1_map.pkl")
            with open(path, "wb") as f:
                pickle.dump(self.mapd, f)

    def load(self, model_dir):
        if self.use_hnsw:
            for key in self.hnsw.keys():
                path = os.path.join(model_dir, f"{key}/anns_m1")
                self.hnsw[key].load(path)
                self.hnsw[key].index._set_query_time_params(efS=self.top_k)
                self.hnsw[key].index.num_neighbours = self.top_k

            path = os.path.join(model_dir, f"anns_m1_map.pkl")
            with open(path, "rb") as f:
                self.mapd = pickle.load(f)
