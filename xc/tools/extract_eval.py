# Example to evaluate
import xclib.evaluation.xc_metrics as xc_metrics
import xclib.utils.sparse as xs
from xc.libs.utils import load_file
import scipy.sparse as sp
import numpy as np
import json
import sys
import os


def load_overlap(data_dir, filter_label_file='filter_labels.txt'):
    docs = np.asarray([])
    lbs = np.asarray([])
    if os.path.exists(os.path.join(data_dir, filter_label_file)):
        filter_lbs = np.loadtxt(os.path.join(
            data_dir, filter_label_file), dtype=np.int32)
        if filter_lbs.size > 0:
            docs = filter_lbs[:, 0]
            lbs = filter_lbs[:, 1]
    else:
        print("Overlap not found")
    print("Overlap is:", docs.size)
    return docs, lbs


def _remove_overlap(score_mat, docs, lbs):
    score_mat[docs, lbs] = 0
    score_mat = score_mat.tocsr()
    score_mat.eliminate_zeros()
    return score_mat


def main(targets_label_file, train_label_file, result_dir, predictions_file,
         A, B, docs, lbls, mode="m2", alpha=0.5):
    true_labels = _remove_overlap(load_file(targets_label_file).tolil(),
                                  docs, lbls)
    trn_labels = load_file(train_label_file)
    inv_propen = xc_metrics.compute_inv_propesity(trn_labels, A=A, B=B)
    acc = xc_metrics.Metrics(
        true_labels, inv_psp=inv_propen, remove_invalid=False)
    if mode == "m2":
        score_mat_dir = os.path.join(result_dir, f"{predictions_file}.npz")
        scr_mat = _remove_overlap(load_file(score_mat_dir).tolil(), docs, lbls)
        rec = xc_metrics.recall(scr_mat, true_labels, k=10)[-1]*100
        print("R@10=%0.2f" % (rec))
        args = acc.eval(scr_mat, 5)
        print(xc_metrics.format(*args))
    if mode == "m4":
        m2_score_mat_dir = os.path.join(
            result_dir, f"module4/m2_{predictions_file}.npz")
        m2 = _remove_overlap(load_file(m2_score_mat_dir).tolil(), docs, lbls)

        m4_score_mat_dir = os.path.join(
            result_dir, f"module4/m4_{predictions_file}.npz")
        m4 = _remove_overlap(load_file(m4_score_mat_dir).tolil(), docs, lbls)
        m2 = xs.retain_topk(m2, k=100)
        m4 = xs.retain_topk(m4, k=100)


        scr_mat = m4.copy().multiply(alpha)
        scr_mat = scr_mat + m2.copy().multiply(1-alpha)
        print(f"alpha={alpha}")
        rec = xc_metrics.recall(scr_mat, true_labels, k=10)[-1]*100
        print("R@10=%0.2f" % (rec))
        args = acc.eval(scr_mat, 5)
        print(xc_metrics.format(*args))
    sp.save_npz(f"{result_dir}/score_{predictions_file}.npz", scr_mat)


if __name__ == '__main__':
    train_label_file = sys.argv[1]
    targets_file = sys.argv[2]  # Usually test data file
    result_dir = sys.argv[3]
    file_name = sys.argv[4]  # In mat format
    data_dir = sys.argv[5]
    configs = json.load(open(sys.argv[6]))["DEFAULT"]
    filter_data = sys.argv[7]
    mode = sys.argv[8]
    alpha = float(sys.argv[9])
    A = configs["A"]
    B = configs["B"]
    docs, lbls = load_overlap(data_dir, filter_label_file=filter_data)
    main(targets_file, train_label_file, result_dir,
         file_name, A, B, docs, lbls, mode, alpha)
