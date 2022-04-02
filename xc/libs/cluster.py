from sklearn.preprocessing import normalize
from multiprocessing import Pool
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import functools
import operator
import torch
import time
import sys


def b_kmeans_dense_multi(fts_lbl, index, tol=1e-4):
    lbl_cent = normalize(np.squeeze(fts_lbl[:, 0, :]))
    lbl_fts = normalize(np.squeeze(fts_lbl[:, 1, :]))
    if lbl_cent.shape[0] == 1:
        return [index]
    cluster = np.random.randint(low=0, high=lbl_cent.shape[0], size=(2))
    while cluster[0] == cluster[1]:
        cluster = np.random.randint(low=0, high=lbl_cent.shape[0], size=(2))
    _centeroids = lbl_cent[cluster]
    _sim = np.dot(lbl_cent, _centeroids.T)
    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        c_lbs = np.array_split(np.argsort(_sim[:, 1]-_sim[:, 0]), 2)
        _centeroids = normalize(np.vstack([
            np.mean(lbl_cent[x, :], axis=0) for x in c_lbs
        ]))
        _sim_1 = np.dot(lbl_cent, _centeroids.T)
        _centeroids = normalize(np.vstack([
            np.mean(lbl_fts[x, :], axis=0) for x in c_lbs
        ]))
        _sim_2 = np.dot(lbl_fts, _centeroids.T)
        _sim = _sim_1 + _sim_2
        old_sim, new_sim = new_sim, np.sum([np.sum(_sim[c_lbs[0], 0]),
                                            np.sum(_sim[c_lbs[1], 1])])
    return list(map(lambda x: index[x], c_lbs))


def b_kmeans_dense(labels_features, index, tol=1e-4, *args, **kwargs):
    if labels_features.shape[0] == 1:
        return [index]
    cluster = np.random.randint(low=0, high=labels_features.shape[0], size=(2))
    while cluster[0] == cluster[1]:
        cluster = np.random.randint(
            low=0, high=labels_features.shape[0], size=(2))
    _centeroids = labels_features[cluster]
    _similarity = np.dot(labels_features, _centeroids.T)
    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        sim_diff = _similarity[:, 1] - _similarity[:, 0]
        sim_diff_idx = np.argsort(sim_diff)
        clustered_lbs = np.array_split(sim_diff_idx, 2)
        c_l = np.mean(labels_features[clustered_lbs[0], :], axis=0)
        c_r = np.mean(labels_features[clustered_lbs[1], :], axis=0)
        _centeroids = normalize(np.vstack([c_l, c_r]))
        _similarity = np.dot(labels_features, _centeroids.T)
        s_l = np.sum(_similarity[clustered_lbs[0], 0])
        s_r = np.sum(_similarity[clustered_lbs[1], 1])
        old_sim, new_sim = new_sim, s_l + s_r
    return list(map(lambda x: index[x], clustered_lbs))


def b_kmeans_sparse(labels_features, index, tol=1e-4, *args, **kwargs):
    def _sdist(XA, XB):
        return XA.dot(XB.transpose())
    labels_features = normalize(labels_features)
    if labels_features.shape[0] == 1:
        return [index]
    cluster = np.random.randint(low=0, high=labels_features.shape[0], size=(2))
    while cluster[0] == cluster[1]:
        cluster = np.random.randint(
            low=0, high=labels_features.shape[0], size=(2))
    _centeroids = normalize(labels_features[cluster].todense())
    _sim = _sdist(labels_features, _centeroids)
    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        c_lbs = np.array_split(np.argsort(_sim[:, 1]-_sim[:, 0]), 2)
        _centeroids = normalize(np.vstack([
            labels_features[x, :].mean(axis=0) for x in c_lbs]))
        _sim = _sdist(labels_features, _centeroids)
        old_sim, new_sim = new_sim, np.sum([
            np.sum(_sim[c_lbs[0], 0]), np.sum(_sim[c_lbs[1], 1])])
    return list(map(lambda x: index[x], c_lbs))


def b_kmeans_dense_gpu(labels_features, index, tol=1e-4, use_cuda=False):
    if use_cuda:
        labels_features = labels_features.cuda()
    if labels_features.shape[0] == 1:
        return [index]
    cluster = np.random.randint(low=0, high=labels_features.shape[0], size=(2))
    while cluster[0] == cluster[1]:
        cluster = np.random.randint(
            low=0, high=labels_features.shape[0], size=(2))
    _centeroids = labels_features[cluster]
    _similarity = torch.mm(labels_features, _centeroids.T)
    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        sim_diff = _similarity[:, 1]-_similarity[:, 0]
        sim_diff_idx = np.argsort(sim_diff.cpu().numpy())
        clustered_lbs = np.array_split(sim_diff_idx, 2)
        c_l = torch.mean(labels_features[clustered_lbs[0], :], dim=0)
        c_r = torch.mean(labels_features[clustered_lbs[1], :], dim=0)
        _centeroids = F.normalize(torch.stack([c_l, c_r], dim=0))
        _similarity = torch.mm(labels_features, _centeroids.T)
        s_l = torch.sum(_similarity[clustered_lbs[0], 0]).item()
        s_r = torch.sum(_similarity[clustered_lbs[1], 1]).item()
        old_sim, new_sim = new_sim, s_l+s_r
    labels_features = labels_features.cpu()
    del labels_features
    return list(map(lambda x: index[x], clustered_lbs))


def get_functions(mat):
    if torch.is_tensor(mat):
        print("Using GPU for clustering")
        return b_kmeans_dense_gpu
    if isinstance(mat, np.ndarray):
        if len(mat.shape) == 3:
            print("Using dense kmeans++ for multi-view")
            return b_kmeans_dense_multi
        elif len(mat.shape) == 2:
            print("Using dense kmeans++")
            return b_kmeans_dense
    elif sp.issparse(mat):
        print("Using sparse kmeans++")
        return b_kmeans_sparse
    print("dtype not understood!!")
    exit(0)


def _normalize(mat):
    if torch.is_tensor(mat):
        return mat
    elif isinstance(mat, np.ndarray) or sp.issparse(mat):
        return normalize(mat)
    else:
        raise TypeError(f"{type(mat)} is not supported")


def cluster(labels, max_leaf_size=None, min_splits=16, num_workers=5,
            return_smat=False, num_clusters=None, force_gpu=False):
    num_nodes = num_clusters
    if num_nodes is None:
        num_nodes = np.ceil(np.log2(labels.shape[0]/max_leaf_size))
        num_nodes = 2**num_nodes
    group = [np.arange(labels.shape[0])]
    labels = _normalize(labels)
    if force_gpu:
        labels = torch.from_numpy(labels).type(torch.FloatTensor)
    splitter = get_functions(labels)
    min_singe_thread_split = min(min_splits, num_nodes)
    if min_singe_thread_split < 1:
        if torch.is_tensor(labels):
            labels = labels.cuda()
    print(f"Max leaf size {max_leaf_size}")
    print(f"Total number of group are {num_nodes}")
    print(f"Average leaf size is {labels.shape[0]/num_nodes}")
    start = time.time()

    def splits(flag, labels, group):
        if flag or torch.is_tensor(labels):
            return map(lambda x: splitter(labels[x], x, use_cuda=not flag), group)
        else:
            with Pool(num_workers) as p:
                mapps = p.starmap(splitter, map(
                    lambda x: (labels[x], x, flag), group))
            return mapps

    def print_stats(group, end="\n", file=sys.stdout):
        string = f"Total groups {len(group)}"
        string += f", Avg. group size {np.mean(list(map(len, group)))}"
        string += f", Total time {time.time()-start} sec."
        print(string, end=end, file=file)

    while len(group) < num_nodes:
        print_stats(group, "\r", sys.stderr)
        flags = len(group) < min_singe_thread_split
        group = functools.reduce(operator.iconcat,
                                 splits(flags, labels, group), [])
    print_stats(group)
    if return_smat:
        cols = np.uint32(np.concatenate(
            [[x]*len(y) for x, y in enumerate(group)]))
        rows = np.uint32(np.concatenate(group))
        group = sp.lil_matrix((labels.shape[0], np.int32(num_nodes)))
        group[rows, cols] = 1
        group = group.tocsr()
    del labels
    return group
