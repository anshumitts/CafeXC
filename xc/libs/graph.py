from sklearn.preprocessing import normalize
from xc.libs.utils import pbar
from xclib.utils import sparse as xs
from numba import njit, prange
from xclib.utils import graph
import scipy.sparse as sp
import numpy as np


def trim_rows(smat, index):
    print(f"UTILS::TRIMMING::#ROWS({index.size})")
    num_rows = smat.shape[0]
    rows = np.ones(num_rows, dtype=np.int32)
    rows[index] = 0
    diag = sp.diags(rows, shape=(num_rows, num_rows))
    return diag.dot(smat).tocsr()


def csr_stats(mat, name="mat"):
    print(f"{name}")
    print(f"\tSHAPE:{mat.shape}")
    print(f"\tNNZ:{mat.nnz}")
    print(f"\tNNZ(axis=0):{np.where(np.ravel(mat.getnnz(axis=0))>0)[0].size}")
    print(f"\tNNZ(axis=1):{np.where(np.ravel(mat.getnnz(axis=1))>0)[0].size}")


def normalize_graph(mat):
    col_nnz = np.sqrt(1/(np.ravel(mat.sum(axis=0))+1e-6))
    row_nnz = np.sqrt(1/(np.ravel(mat.sum(axis=1))+1e-6))
    c_diags = sp.diags(col_nnz)
    r_diags = sp.diags(row_nnz)
    mat = r_diags.dot(mat).dot(c_diags)
    mat.eliminate_zeros()
    return mat


def clean_graphs(min_freq=2, max_freq=100, *graphs):
    smats = sp.vstack([graph.data for graph in graphs], "csr")
    node_freq = np.ravel(smats.getnnz(axis=0))
    num_nodes = node_freq.size
    keep_nodes = np.ones(num_nodes)
    keep_nodes[node_freq > max_freq] = 0
    keep_nodes[node_freq <= min_freq] = 0
    diag = sp.diags(keep_nodes, shape=(num_nodes, num_nodes))
    for i in range(len(graphs)):
        graphs[i].data = graphs[i].data.dot(diag)
        graphs[i].data.eliminate_zeros()
    return graphs


@njit(parallel=True, nogil=True)
def _random_walk(q_rng, q_lbl, l_rng, l_qry, walk_to,
                 p_reset, hops_per_step, start, end):
    """
    Compute random walk for a batch of labels in the label space
    One hop is consits of following steps:
        1) Randomly jumping from label to a document 
        2) Randomly jumping from the document to a document 
    Arguments:
    ---------
    q_rng: np.ndarray
        label pointers in CSR format index pointer array of the matrix
    q_lbl: np.ndarray
        label indices in CSR format index array of the matrix
    l_rng: np.ndarray
        document pointers in CSR format index pointer  array of the matrix
    l_qry: np.ndarray
        document indices in CSR format index pointer array of the matrix
    walk_to: int
        random walk length (int)
    p_reset: int
        random restart probability (float)
    start: int 
        start index of the label
    end: int
        last index of the label
    Returns:
    ---------
    np.ndarray: np.int32 [start-end x walk_to] 
                flattened array of indices for correlated
                labels with duplicate entries corresponding 
                to [start, ..., end] indices of the labels
    np.ndarray: np.float32 [start-end x walk_to] 
                flattened array of relevance for correlated
                labels with duplicate entries corresponding
                to [start, ..., end] indices of the labels
    """
    n_nodes = end - start
    nbr_idx = np.zeros((n_nodes, walk_to), dtype=np.int32)
    nbr_dat = np.zeros((n_nodes, walk_to), dtype=np.float32)
    for idx in prange(0, n_nodes):
        lbl_k = idx + start
        l_start, l_end = l_rng[lbl_k], l_rng[lbl_k+1]
        if l_start - l_end == 0:
            continue
        _qidx = np.random.choice(l_qry[l_start: l_end])
        for walk in np.arange(0, walk_to):
            p = np.random.random()
            if p < p_reset:
                l_start, l_end = l_rng[lbl_k], l_rng[lbl_k+1]
                _qidx = np.random.choice(l_qry[l_start: l_end])
            else:
                if hops_per_step == 2:
                    _lidx = nbr_idx[idx, walk-1]
                    l_start, l_end = l_rng[_lidx], l_rng[_lidx+1]
                    _qidx = np.random.choice(l_qry[l_start: l_end])
                if hops_per_step == 3:
                    _qidx = nbr_idx[idx, walk-1]

            q_start, q_end = q_rng[_qidx], q_rng[_qidx+1]
            _idx = np.random.choice(q_lbl[q_start: q_end])

            if hops_per_step == 3:
                l_start, l_end = l_rng[_idx], l_rng[_idx+1]
                _idx = np.random.choice(l_qry[l_start: l_end])

            nbr_idx[idx, walk] = _idx
            nbr_dat[idx, walk] = 1
    return nbr_idx.flatten(), nbr_dat.flatten()


def Prune(G, R, C, batch_size=1024, normalize=True):
    R = normalize(R)
    C = normalize(C)
    rows, cols = G.nonzero()
    _nnz = G.nnz
    data = np.zeros(_nnz)
    for start in pbar(np.arange(0, _nnz, batch_size), desc="Pruning"):
        end = min(start + batch_size, _nnz)
        _R = R[rows[start:end]]
        _C = C[cols[start:end]]
        data[start:end] = np.ravel(np.sum(_R*_C, axis=1))
    data[data < 0] = 0
    OG = sp.csr_matrix((data, (rows, cols)), shape=G.shape)
    OG.eliminate_zeros()
    csr_stats(OG, "GRAPH")
    if normalize:
        OG = normalize_graph(OG)
    return OG


class PrunedWalk(graph.RandomWalk):
    def __init__(self, Y, yf=None):
        self.Y = Y.transpose().tocsr()
        self.Y.sort_indices()
        self.Y.eliminate_zeros()
        self.yf = yf
        if self.yf is not None:
            self.yf = normalize(yf)
            self.yf = yf[self.valid_labels]

    def simulate(self, walk_to=100, p_reset=0.2, k=None, hops_per_step=2, b_size=1000, max_dist=2):
        q_lbl = self.Y.indices
        q_rng = self.Y.indptr
        trn_y = self.Y.transpose().tocsr()
        trn_y.sort_indices()
        trn_y.eliminate_zeros()
        l_qry = trn_y.indices
        l_rng = trn_y.indptr
        shape_idx = int(hops_per_step % 2 - 1)
        n_lbs = self.Y.shape[shape_idx]
        n_itm = self.Y.shape[1]
        mats = []
        pruned_edges = 0
        for p_idx, idx in enumerate(np.arange(0, n_itm, b_size)):
            if p_idx % 50 == 0:
                print("INFO:WALK: completed [ %d/%d ]" % (idx, n_itm))
            start, end = idx, min(idx+b_size, n_itm)
            cols, data = _random_walk(q_rng, q_lbl, l_rng, l_qry, walk_to,
                                      p_reset, hops_per_step, start=start, end=end)
            rows = np.arange(end-start).reshape(-1, 1)
            rows = np.repeat(rows, walk_to, axis=1).flatten()
            mat = sp.coo_matrix((data, (rows, cols)), dtype=np.float32,
                                shape=(end-start, n_lbs))
            mat.sum_duplicates()
            mat = mat.tocsr()
            mat.sort_indices()
            if self.yf is not None:
                _rows, _cols = mat.nonzero()
                _lbf = self.yf[start+_rows]
                _dist = 1-np.ravel(np.sum(_lbf*self.yf[_cols], axis=1))
                mat.data[_dist > max_dist] = 0
                pruned_edges += np.sum(_dist > max_dist)
                mat.eliminate_zeros()
            mats.append(mat)
            del rows, cols
        print("INFO:WALK: completed [ %d/%d ]" % (n_itm, n_itm))
        mats = sp.vstack(mats, "csr")
        mats = normalize_graph(mats)
        if k is not None:
            mats = xs.retain_topk(mats, k=k).tocsr()
        return mats


def print_stats(mat, k=10):
    _mat = mat.copy()
    _mat.__dict__['data'][:] = 1
    freqs = _mat.sum(axis=1)
    print(np.max(freqs), np.min(freqs), np.std(freqs))


def WalkItOff(mat, head_thresh=500, walk_len=400, p_reset=0.8,
              topk=10, batch_size=1023, max_dist=1):
    mat = mat.tocsr()
    mat.data[:] = 1
    doc_freq = np.ravel(mat.sum(axis=0))
    head_docs = np.where(doc_freq > head_thresh)[0]
    print(f"Head labels: {head_docs.size}")
    mat = trim_rows(mat.T.tocsr(), head_docs).T.tocsr()
    return PrunedWalk(mat).simulate(walk_len, p_reset, topk,
                                    batch_size, max_dist)
