from .custom_dtypes import MultiViewData, BatchData, padded_inputs
from xc.tools.tokenize_text import setup_tokenizer, tokens
from sklearn.preprocessing import normalize
from .utils import load_file, fasterTxtRead
from xc.libs.custom_dtypes import save
from .data_base import NullDataset
from copy import copy, deepcopy
import scipy.sparse as sp
import numpy as np
import torch
import glob
import os

def load_txt(root, n_file):
    read_full = os.environ['RESTRICTMEM'] == '0'
    print(f"TXT:{n_file}(read_full={read_full})")
    if n_file.endswith("npz"):
        return TXTDataset(root, n_file)
    elif n_file.endswith("txt"):
        return RAWDataset(root, n_file)
    elif n_file.endswith("seq.memmap"):
        return SEQDataset(root, n_file)
    elif n_file.endswith("pretrained"):
        if os.environ['RESTRICTMEM'] == '1':
            return MEMTXTDataset(root, n_file, read_full)
        return NPYTXTDataset(root, n_file)
    else:
        return None


class TXTDataset(NullDataset):
    def __init__(self, root, n_file):
        self.data = load_file(os.path.join(root, f"{n_file}"))
        self._type = "bow"
        self.normalize()

    def __getitem__(self, idx):
        return self.data[idx]

    def normalize(self):
        self.data = normalize(self.data)

    def __len__(self):
        return self.data.shape[0]

    def __deepcopy__(self, memo):
        # create a copy with self.linked_to *not copied*, just referenced.
        obj = copy(self)
        obj.data = deepcopy(self.data, memo)
        return obj

    @property
    def valid(self):
        return np.where(np.ravel(self.data.getnnz(axis=1)) > 0)[0]

    @property
    def num_features(self):
        return self.data.shape[-1]

    def keep_valid(self, indices):
        self.data = self.data[indices]

    def padd(self):
        self.data = sp.hstack([self.data,
                               sp.lil_matrix(len(self), 1)]).tocsr()
        print("Adding padding index in the last")

    def get_fts(self, idx, _desc):
        return padded_inputs(self[idx])

    def filter(self, valid_pts, axis=0):
        if axis == 1:
            s_mat = sp.lil_matrix((self.num_features, valid_pts.size))
            s_mat[valid_pts, np.arange(valid_pts.size)] = 1
            self.data = self.data.dot(s_mat).tocsr()
        if axis == 0:
            self.data = self.data[valid_pts]

    def vstack(self, obj):
        if isinstance(self.data, np.ndarray):
            self.data = np.vstack([self.data, obj.data])
        elif sp.issparse(self.data):
            self.data = sp.vstack([self.data, obj.data], 'csr')


class RAWDataset(TXTDataset):
    def __init__(self, root, n_file):
        self.data = fasterTxtRead(os.path.join(root, f"{n_file}"))
        self.data = list(map(lambda x: x.strip().replace("_", " "), self.data))
        if len(self.data[0].split('->', 1)) == 2:
            self.data = list(map(lambda x: x.split('->', 1)[1], self.data))    
        self.tokenizer = None
        self.max_len = None
        self._type = "txt"

    def __len__(self):
        return len(self.data)

    def keep_valid(self, indices):
        self.data = self[indices]

    def filter(self, valid_pts, axis=0):
        if axis == 1:
            pass
        if axis == 0:
            self.keep_valid(valid_pts)

    def __getitem__(self, idx):
        data = list(map(lambda x: self.data[x], idx))
        return data    

    def get_fts(self, idx, _desc):
        data = self[idx]
        index, mask = tokens(data, self.tokenizer, self.max_len)
        return MultiViewData(BatchData({"mask": mask, "index": index}))

    def build_pre_trained(self, txt_model, data_dir, file_name,
                          params, prefix="txt.seq.memmap", thresh=5e6):
        self.max_len = params.max_len
        self.tokenizer = setup_tokenizer(txt_model)
        if len(self.data) > thresh:
            return self
        file_name = f"{file_name}.{prefix}"
        cached_path = os.path.join(data_dir, txt_model)
        print(f"{cached_path}/{file_name}")
        if len(glob.glob(f"{cached_path}/{file_name}*")) > 0:
            return load_txt(cached_path, file_name)
        input_idx, attention = tokens(self.data, self.tokenizer, self.max_len)
        _tokens = np.stack([input_idx, attention], axis=1)
        os.makedirs(cached_path, exist_ok=True)
        save(f"{cached_path}/{file_name}", "memmap", _tokens)
        del _tokens
        return load_txt(cached_path, file_name)


class SEQDataset(TXTDataset):
    def __init__(self, root, n_file):
        self.data = load_file(os.path.join(root, f"{n_file}"))
        self._type = "seq"
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def valid(self):
        return np.where(np.ravel(np.sum(self.data[:, 1, :], axis=-1)) > 0)[0]

    @property
    def num_features(self):
        return np.max(self.data[:, 0, :])

    def filter(self, valid_pts, axis=0):
        if axis == 1:
            pass
        if axis == 0:
            self.keep_valid(valid_pts)

    def __getitem__(self, idx):

        if not isinstance(idx, int):
            sorted_idx = np.argsort(idx)
            idx = idx[sorted_idx]
            min_ind, max_ind = min(idx), max(idx)+1
            data = np.array(self.data[min_ind:max_ind][idx-min_ind])
            data[sorted_idx] = data.copy()
        else:
            data = self.data[idx]

        return data

    def get_fts(self, idx, _desc):
        data = self[idx]
        data = torch.from_numpy(data).type(torch.LongTensor)
        index = data[:, 0, :].squeeze()
        mask = data[:, 1, :].squeeze()
        max_seq = mask.sum(dim=1).max()
        index, mask = index[:, :max_seq], mask[:, :max_seq]
        return MultiViewData(BatchData({"mask": mask, "index": index}))


class NPYTXTDataset(TXTDataset):
    def __init__(self, root, n_file):
        self.vect = load_file(os.path.join(root, f"{n_file}.npy"))
        self.data = load_file(os.path.join(root, f"{n_file}.npz"))
        self.shape = self.data.shape
        self._type = "pretrained"

    @property
    def valid(self):
        return np.where(np.ravel(np.sum(self.data[:, 1, :], axis=-1)) > 0)[0]

    @property
    def num_features(self):
        return self.fts

    def filter(self, valid_pts, axis=0):
        if axis == 1:
            pass
        if axis == 0:
            self.keep_valid(valid_pts)

    def __getitem__(self, idx):
        flags = self.data[idx]
        ind = np.unique(flags.indices)
        flags = flags.tocsc()[:, ind].tocsr()
        return MultiViewData(flags, self.vect[ind])

    def get_fts(self, idx, _desc):
        return self[idx]

    @property
    def pretrained_vect(self):
        return self.data.dot(self.vect)

    def vstack(self, objt):
        ob1_shape = self.data.shape
        ob2_shape = objt.data.shape
        padd1 = sp.csr_matrix((ob1_shape[0], ob2_shape[1]))
        padd2 = sp.csr_matrix((ob2_shape[0], ob1_shape[1]))
        self.data = sp.vstack([sp.hstack([self.data, padd1]),
                               sp.hstack([padd2, objt.data])], 'csr')
        self.vect = np.vstack([self.vect, objt.vect])


class MEMTXTDataset(NPYTXTDataset):
    def __init__(self, root, n_file, read_full=True):
        self.vect = load_file(os.path.join(root, f"{n_file}.memmap"))
        self.data = load_file(os.path.join(root, f"{n_file}.npz"))
        self.shape = self.data.shape
        self._type = "pretrained"
        if read_full:
            self.vect = np.array(self.vect[:])

    def __getitem__(self, idx):

        if not isinstance(idx, int):
            sorted_idx = np.argsort(idx)
            idx = idx[sorted_idx]

        flags = self.data[idx]
        ind = np.unique(flags.indices)
        min_ind, max_ind = min(ind), max(ind) + 1
        flags = flags.tocsc()[:, ind]
        txts = self.vect[min_ind:max_ind][ind-min_ind]
        if not isinstance(idx, int):
            flags = flags.tolil()
            flags[sorted_idx] = flags.copy()
        return MultiViewData(flags.tocsr(), txts)
