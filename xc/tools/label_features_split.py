from libs.utils import load_file
import scipy.sparse as sp
import numpy as np
import sys
import os


def write(tmp_mdata, labels, features):
    print("# labels:", labels.size)
    print("# features:", features.size)
    path = os.path.join(tmp_mdata, 'labels_split.txt')
    np.savetxt(path, labels, fmt='%d')
    path = os.path.join(tmp_mdata, 'features_split.txt')
    np.savetxt(path, features, fmt='%d')


def main():
    trn_y = load_file(sys.argv[1])
    if os.path.exists(sys.argv[2]) or os.path.exists(sys.argv[2]+".dat"):
        trn_x_xf = load_file(sys.argv[2])
        lbl_x_xf = load_file(sys.argv[3])
        if isinstance(trn_x_xf, np.ndarray):
            print("Figure it out")
        elif sp.issparse(trn_x_xf):
            fts_docs = np.where(trn_x_xf.getnnz(axis=0) > 0)[0]
            fts_lbls = np.where(lbl_x_xf.getnnz(axis=0) > 0)[0]
            features = np.union1d(fts_docs, fts_lbls)
        elif isinstance(trn_x_xf, tuple):
            features = np.arange(trn_x_xf[1])
    else:
        features = np.arange(10)
    labels = np.where(trn_y.getnnz(axis=0) > 0)[0]
    tmp_mdata = sys.argv[4]
    write(tmp_mdata, labels, features)


if __name__ == '__main__':
    main()
