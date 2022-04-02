import os
import argparse
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from xclib.data import data_utils as du


def setup():
    parser = argparse.ArgumentParser("Combine evaluate")
    parser.add_argument('--docs_input', action='store', type=str)
    parser.add_argument('--lbls_input', action='store', type=str)
    parser.add_argument('--in_dir', action='store', type=str)
    parser.add_argument('--ot_dir', action='store', type=str)
    return parser.parse_args()


def build_docs(args):
    input_file = os.path.join(args.in_dir, args.docs_input)
    return dict(list(map(lambda x: tuple(x.strip().split('->', 1)),
                         open(input_file, "r"))))


def print_lines(indx, dict_lines, idx2keys, path):
    with open(path, "w") as f:
        for idx in indx:
            key = idx2keys[idx]
            f.write(f"{key}->{dict_lines[key]}\n")


def build_lbls(args, dict_lines):
    input_file = os.path.join(args.in_dir, args.lbls_input)
    key2idx = dict(list(zip(dict_lines.keys(), np.arange(len(dict_lines)))))
    idx2keys = dict(list((v, k) for k, v in key2idx.items()))
    rows, cols, data = [], [], []
    relevance = list(map(lambda x: x.strip().split("\t"), open(input_file)))
    for d, l, score in tqdm(relevance):

        l_idx = key2idx.get(l, len(key2idx))
        d_idx = key2idx.get(d, len(key2idx))

        if l_idx == len(key2idx) or d_idx == len(key2idx):
            continue
        cols.append(l_idx)
        rows.append(d_idx)
        data.append(float(score))
    max_cols = np.max(cols) + 1
    max_rows = np.max(rows) + 1
    gt = sp.lil_matrix((max_rows, max_cols))
    gt[rows, cols] = data
    gt = gt.tocsr()

    valid_rows = np.where(np.ravel(gt.sum(axis=1)) > 0)[0]
    valid_cols = np.where(np.ravel(gt.sum(axis=0)) > 0)[0]
    print(f"shape=({valid_rows.size}, {valid_cols.size})")
    print_lines(valid_rows, dict_lines, idx2keys,
                f"{args.ot_dir}/raw_data/corpus.raw.txt")
    print_lines(valid_cols, dict_lines, idx2keys,
                f"{args.ot_dir}/raw_data/label.raw.txt")
    gt = gt[valid_rows].tocsc()[:, valid_cols].tocsr()
    du.write_sparse_file(gt, f"{args.ot_dir}/corpus_X_Y.txt")


if __name__ == '__main__':
    args = setup()
    print(args)
    os.makedirs(args.ot_dir, exist_ok=True)
    lines = build_docs(args)
    build_lbls(args, lines)
