from xc.libs.data_img import load_img
from xc.libs.data_txt import load_txt
from xc.libs.data_lbl import LBLDataset
from xc.libs.data_shorty import SHORTYDataset
from xc.libs.data_base import NullDataset
from xc.libs.custom_dtypes import BatchData
from torch.utils.data import Dataset
import numpy as np


def FtsData(data_dir, n_file, _type="img", rand_k=-1, max_worker_thread=10):
    if n_file is None:
        return NullDataset()
    elif _type == "img":
        return load_img(data_dir, n_file, max_worker_thread, rand_k)
    elif _type == "txt":
        return load_txt(data_dir, n_file)
    elif _type == "lbl":
        return LBLDataset(data_dir, n_file)
    elif _type == "shorty":
        return SHORTYDataset(data_dir, n_file)


class GroupFts(Dataset):
    def __init__(self, data_dir, n_file_img, n_file_txt,
                 _type="docs", max_worker_thread=10, rand_k=-1):
        self.data_dir = data_dir
        self.dtype = _type
        self.IMG = FtsData(data_dir, n_file_img, "img", rand_k=rand_k,
                           max_worker_thread=max_worker_thread)
        self.TXT = FtsData(data_dir, n_file_txt, "txt")

    def filter(self, indices, axis=0):
        if axis == 0:
            self.IMG.filter(indices, axis)
        self.TXT.filter(indices, axis)

    def get_fts(self, idx):
        img, txt = None, None
        if idx is not None:
            img = self.IMG.get_fts(idx, self.dtype)
            txt = self.TXT.get_fts(idx, self.dtype)
        if img is None and txt is None:
            return None
        return BatchData({"img": img, "txt": txt})

    def blocks(self, shuffle=False):
        index = np.arange(len(self))
        if shuffle:
            np.random.shuffle(index)
        return index

    @property
    def valid(self):
        valid_txt = self.TXT.valid
        valid_img = self.IMG.valid
        if valid_txt is None and valid_img is not None:
            return valid_img
        if valid_txt is not None and valid_img is None:
            return valid_txt
        valid = np.intersect1d(valid_txt, valid_img)
        return valid

    def __len__(self):
        length = len(self.TXT)
        if length > 0:
            return length
        return len(self.IMG)

    def __getitem__(self, idx):
        return {"x": idx}

    @property
    def type_dict(self):
        return {"txt": self.TXT._type, "img": self.IMG._type}

    @property
    def pretrained_vect(self):
        IMG = self.IMG.pretrained_vect
        TXT = self.IMG.pretrained_vect
        return IMG + TXT

    def vstack(self, obj):
        self.TXT.vstack(obj.TXT)
        self.IMG.vstack(obj.IMG)

    def build_pre_trained(self, txt_model, img_model, file_name, params):
        if txt_model is not None:
            self.TXT = self.TXT.build_pre_trained(txt_model, self.data_dir,
                                                  file_name, params)
        if img_model is not None:
            self.IMG = self.IMG.build_pre_trained(img_model, self.data_dir,
                                                  file_name, params)
