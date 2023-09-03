from .custom_dtypes import MultiViewData, BatchData, padded_inputs
from xc.tools.tokenize_text import setup_tokenizer, tokens, tokenize_corpus
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
    elif n_file.endswith("seq.memmap") or n_file.endswith("seq.npy"):
        return SEQDataset(root, n_file, read_full)
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
        self.data = os.path.join(root, f"{n_file}")
        self.tokenizer = None
        self.max_len = None
        self._type = "txt"
    
    @property
    def shape(self):
        return (len(self.data), self.max_len)
        

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
                          params, prefix="txt.seq.memmap",
                          thresh=5e9, accelerator=None):
        self.max_len = params.max_len
        self.tokenizer = setup_tokenizer(txt_model)
        file_name = f"{file_name}.{prefix}"
        _txt_model = txt_model
        if _txt_model in ["custom"]:
            _txt_model = "sentencebert"
        cached_path = os.path.join(data_dir, _txt_model)
        if len(glob.glob(f"{cached_path}/{file_name}*")) > 0:
            return load_txt(cached_path, file_name)
        self.data = fasterTxtRead(self.data)
        print(f"{cached_path}/{file_name}")
        input_idx, attention = tokenize_corpus(self.data, self.tokenizer, self.max_len)
        _tokens = np.hstack([input_idx, attention])
        os.makedirs(cached_path, exist_ok=True)
        save(f"{cached_path}/{file_name}", "memmap", _tokens)
        del _tokens
        return load_txt(cached_path, file_name)
    
    def vstack(self, objt):
        self.data.extend(objt.data)


class SEQDataset(TXTDataset):
    def __init__(self, root, n_file, read_full=False):
        self.data = load_file(os.path.join(root, f"{n_file}"))
        if read_full:
            self.data = np.array(self.data[0:None])
        self._type = "seq"
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def valid(self):
        return np.arange(self.data.shape[1])

    @property
    def num_features(self):
        return np.max(self.data)

    def filter(self, valid_pts, axis=0):
        if axis == 1:
            pass
        if axis == 0:
            self.keep_valid(valid_pts)

    def __getitem__(self, idx):
        idx = np.int32(idx)
        return self.data[idx]

    def get_fts(self, idx, _desc):
        data = self[idx]
        data = torch.from_numpy(np.int64(data)).type(torch.LongTensor)
        length = data.shape[1]
        index = data[:, :length//2]
        mask = data[:, length//2:]
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