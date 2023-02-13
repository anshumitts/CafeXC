from xclib.utils.sparse import retain_topk, topk as Topk
from xclib.data import data_utils as du
import matplotlib.pyplot as plt
import scipy.sparse as sp
from tqdm import tqdm
import numpy as np
import os


def load_overlap(filter_label_file='filter_labels'):
    docs = np.asarray([])
    lbs = np.asarray([])
    if os.path.exists(filter_label_file):
        filter_lbs = np.loadtxt(filter_label_file, dtype=np.int32)
        if filter_lbs.size > 0:
            return filter_lbs[:, 0], filter_lbs[:, 1]
    return docs, lbs


def _load_file(mat):
    if sp.issparse(mat):
        return mat
    elif mat.endswith(".npz"):
        return sp.load_npz(mat)
    elif mat.endswith(".txt"):
        return du.read_sparse_file(mat)

    
def _load_mat_keep_topk(score_mats, k=5, _filter=([], []), return_indx=False):
    if isinstance(score_mats, AnalyseMatrix):
        if score_mats.isequivalent(k):
            return score_mats
    data_dict = {}
    for key in tqdm(score_mats.keys(), desc="loading"):
        _mat = _load_file(score_mats[key])
        if len(_filter[0]) > 0:
            _mat = _mat.tolil()
            _mat[_filter[0], _filter[1]] = 0
            _mat = _mat.tocsr()
        if return_indx:
            data_dict[key] = Topk(_mat, k=k)
        else:
            data_dict[key] = retain_topk(_mat, k=k)
    return data_dict


class AnalyseMatrix:
    def __init__(self, score_mats, _topk, _filter, return_indx=False):
        self.score_mats = _load_mat_keep_topk(score_mats, _topk, _filter, return_indx)
        self._topk = _topk
        
    def __getitem__(self, idx):
        return self.score_mats[idx]
    
    def isequivalent(self, _topk):
        return self._topk == _topk
    
    def keys(self):
        return self.score_mats.keys()
    

def _preferred_rows(score_mats, sorted_mats, strict_mats, y_mat):
    score_values = []
    _mat = score_mats[sorted_mats[0]].multiply(y_mat)
    _mat.eliminate_zeros()
    _mat.data[:] = 1
    _method = np.ravel(_mat.sum(axis=1))
    score_values.append(_method)
    valid_rows = np.full(y_mat.shape[0], True)
    old_score = _method
    for idx, key in enumerate(sorted_mats[1:]):
        _mat = score_mats[sorted_mats[idx+1]].multiply(y_mat)
        _mat.eliminate_zeros()
        _mat.data[:] = 1
        _score = np.ravel(_mat.sum(axis=1))
        _comp = old_score >= _score + strict_mats[idx+1]
        valid_rows = np.logical_and(valid_rows, _comp)
        score_values.append(_score)
        old_score = _score
    score_values = np.asarray(np.vstack(score_values).T)
    rows_idx = np.where(valid_rows)[0]
    _idx = np.argsort(-1*score_values[rows_idx][:, 0])
    rows_idx = rows_idx[_idx]
    return rows_idx


def _make_string(key, txt_map, idx, flag, doc_freq, correlation=None):
    #TODO Add filter based on correlated predictions
    _key = f"(S = {np.sum(flag)})"
    _txt_map = list([txt_map[int(l_idx)].strip() for l_idx in idx])
    _frq_map = list([int(doc_freq[int(l_idx)]) for l_idx in idx])
    _flg_map = list("C" if flag[i] ==1 else "w" for i, _ in enumerate(idx))
    _itm_map = []
    for _txt, _frq, _flg in zip(_txt_map, _frq_map, _flg_map):
        _itm_map.append(f"{_txt} ({_flg}, {_frq})")
    return _key, _itm_map


def _print_mats(score_mats, valid_docs, doc_frq, tst_map, lbl_map,
                tst_mat, filter_correlated=False, out_file="out.txt"):
    #TODO Add filter based on correlated predictions
    examples = []
    if out_file is not None:
        f = open(out_file, "w")
    keys = list(score_mats.keys())
    tst_mat = tst_mat.tolil()
    for r_idx in tqdm(valid_docs, desc="Printing"):
        data_point = {}
        data_point["index"] = r_idx
        data_point["title"] = tst_map[r_idx].strip()
        data_point["gt"] = list([ lbl_map[l_idx].strip() for l_idx in tst_mat[r_idx].rows[0]])
        data_point["preds"]= {}
        if out_file is not None:
            print(f"{data_point['index']}->{data_point['title']}", file=f)
            print("".join(["-"]*80), file=f)
            print(f"{', '.join(data_point['gt'])}", file=f)
            print("".join(["-"]*80), file=f)

        for key in keys:                
            l_dat = score_mats[key][r_idx].data
            l_idx = score_mats[key][r_idx].indices[np.argsort(-l_dat)]
            l_flg = np.ravel(tst_mat[r_idx, l_idx].todense())
            _scr, _items = _make_string(key, lbl_map, l_idx, l_flg, doc_frq)
            data_point["preds"][key] = {"score": _scr, "items": _items}
            if out_file is not None:
                print(f"{key}: {_scr} -> {', '.join(_items)}", file=f)
                print("".join(["-"]*30), file=f)
        if out_file is not None:
            print("".join(["="]*80), file=f)
        else:
            examples.append(data_point)
    return examples

    
def _split_based_on_frequency(freq, num_splits):
    """
        Split labels based on frequency
    """
    thresh = np.sum(freq)//num_splits
    index = np.argsort(-freq)
    indx = [[]]
    cluster_frq = 0
    t_splits = num_splits -1
    for idx in index:
        cluster_frq += freq[idx]
        if cluster_frq > thresh and t_splits > 0:
            t_splits-=1
            cluster_frq = freq[idx]
            indx[-1] = np.asarray(indx[-1])
            indx.append([])
        indx[-1].append(idx)
    indx[-1] = np.asarray(indx[-1])
    freq[freq == 0] = np.nan
    xticks = ["%d\n(#%dK)\n(%0.2f)" % (i+1, freq[x].size//1000,
                                      np.nanmean(freq[x])) for i, x in enumerate(indx)]
    return indx, xticks


def _pointwise_eval(score_mats, sorted_mats, tst_mat, topk, metric="P"):
    score_mats = _load_mat_keep_topk(score_mats, topk)
    doc_lbl_freq = tst_mat.sum(axis=0)
    scores = {}
    for key in sorted_mats:
        _mat = score_mats[key].multiply(tst_mat)
        _mat.eliminate_zeros()
        _mat.data[:] = 1
        if metric == "P":
            _mat = _mat.multiply(1/(topk*tst_mat.shape[0]))
        
        if metric == "R":
            deno = tst_mat.sum(axis=1)*tst_mat.shape[0]
            _mat = _mat.multiply(1/deno)
        
        if metric == "%FN":
            _mat = tst_mat - _mat
            _mat.eliminate_zeros()
            _mat = _mat.multiply(1/(doc_lbl_freq*tst_mat.shape[1]))
        scores[key] = np.ravel(_mat.sum(axis=0))
    return scores


def _plot(scores, xticks, yticks, title='', fname='temp.eps'):
    len_methods = len(scores.keys())
    n_groups = len(xticks)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.95/(n_groups)
    shift = -bar_width*(len_methods-1)
    opacity = 1.0

    plt.grid(b=True, which='major', axis='both')
    for idx, (_name, _val) in enumerate(scores.items()):
        _x = index+shift
        ax.bar(x=_x[::-1], height=_val, width=bar_width, alpha=opacity, label=_name)
        shift += bar_width

    plt.xlabel('Quantiles \n (Increasing Freq.)', fontsize=18)
    plt.ylabel(yticks, fontsize=18)
    plt.title(title, fontsize=22)
    plt.xticks(index-bar_width*((len_methods-1)/2), xticks[::-1], fontsize=14)
    plt.legend(fontsize=10)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.0)


def _decile_plot(scores, doc_frq, num_splits, ylabel="P", title="Dataset", out_file="test.pdf"):
    indx, xticks = _split_based_on_frequency(doc_frq, num_splits)
    contribs = {}
    for mat in tqdm(scores.keys(), desc="computing"):
        contribs[mat] = []
        for idx in indx:
            contribs[mat].append(np.sum(scores[mat][idx])*100)
        contribs[mat].append(np.sum(scores[mat])*100)
    xticks+=["complete\n(#%dK)" % (doc_frq.size//1000)]
    _plot(contribs, xticks, f"{ylabel}", title, out_file)


def print_mats(score_mats, sorted_mats, strict_mats, topk, tst_map, lbl_map,
               tst_mat, trn_mat, filter_correlated=False, out_file="out.txt"):
    score_mats = _load_mat_keep_topk(score_mats, topk)
    valid_docs = _preferred_rows(score_mats, sorted_mats, strict_mats, tst_mat)
    doc_frq = np.ravel(trn_mat.sum(axis=0))
    _print_mats(score_mats, valid_docs, doc_frq, tst_map,
                lbl_map, tst_mat, filter_correlated, out_file)


def decile_plot(score_mats, sorted_mats, topk, num_splits, tst_mat, trn_mat,
                metric="P", title="Dataset", out_file="test.pdf"):    
    scores = _pointwise_eval(score_mats, sorted_mats, tst_mat, topk, metric)
    doc_frq = np.ravel(trn_mat.sum(axis=0))
    label = f"{metric}@{topk}" if metric in ["P", "R"] else metric
    _decile_plot(scores, doc_frq, num_splits, label, title, out_file)