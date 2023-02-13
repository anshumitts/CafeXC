import copy
import torch
import numpy as np
import scipy.sparse as sp
from torch.nn.parallel._functions import Scatter
from functools import partial
from typing import Any, Union, List, Tuple

StrVector = List[str]
TupleVector = List[Tuple[str, str]]

def padded_inputs(smat, index_pad_value=0, mask_pad_value=0, return_bool=True):
    smat = smat.tolil()
    rows = map(lambda x: torch.from_numpy(
        np.asarray(x)).type(torch.LongTensor), smat.rows)
    data = map(lambda x: torch.from_numpy(
        np.asarray(x)).type(torch.FloatTensor), smat.data)
    index = torch.nn.utils.rnn.pad_sequence(
        list(rows), batch_first=True,
        padding_value=index_pad_value)
    weigh = torch.nn.utils.rnn.pad_sequence(
        list(data), batch_first=True,
        padding_value=mask_pad_value)
    index, mask = index.data, weigh.data
    if return_bool:
        mask[mask != mask_pad_value] = 1
    return BatchData({"index": index, "mask": mask})


def is_namedtuple(obj):
    return (
        isinstance(obj, tuple) and hasattr(
            obj, "_asdict") and hasattr(obj, "_fields")
    )


def clean_str(_str: str, ext: str) -> str:
    clean_dict = {"sparse": "npz"}
    ext = clean_dict.get(ext, ext)
    _str = _str.split(".")
    if _str[-1] == ext:
        del _str[-1]
    return ".".join(_str)


def save(fname: str, save_type: str, arr: any) -> None:
    fname = clean_str(fname, save_type)
    if save_type == "memmap":
        _dtype = arr.dtype.name
        with open(fname+".memmap.meta", "w") as f:
            line = _dtype
            for content in list(arr.shape):
                line += f",{content}"
            f.write(f"{line}\n")
        features = np.memmap(fname+".memmap.dat", dtype=_dtype,
                             mode='w+', shape=arr.shape)
        features[:] = arr
        features.flush()

    if save_type == "npy":
        np.save(fname+".npy", arr)

    if save_type == "sparse":
        sp.save_npz(fname+".npz", arr, compressed=False)


def scatter(inputs: Any, split_on_device: Union[int, StrVector],
            dim: int = 0, debug: bool = False) -> list:
    if not inputs:
        return []
    num_splits = 0
    if isinstance(split_on_device, int):
        action = "split"
        num_splits = split_on_device
    else:
        action = "scatter"
        num_splits = len(split_on_device)
    if debug:
        print(inputs, type(inputs))

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            if action == "split":
                return obj.chunk(num_splits, dim)
            elif action == "scatter":
                return Scatter.apply(split_on_device, None, dim, obj)

        if isinstance(obj, MultiViewData):
            if action == "split":
                return obj.chunk(num_splits)
            elif action == "scatter":
                return obj.scatter(split_on_device)

        if isinstance(obj, DataParallelList):
            assert len(
                obj) == num_splits, "Splits in DPList should match number of gpus"
            if action == "scatter":
                obj.parallel_to(split_on_device)
            return obj

        if isinstance(obj, (dict, BatchData)) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]

        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        return [obj for _ in range(num_splits)]
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None
    return res


def pre_split(batch: Any, keys: list, num_splits: int):
    if num_splits == 1:
        return batch

    for key in keys:
        batch[key] = DataParallelList(scatter(batch[key], num_splits))
    return batch


class DataParallelList(object):
    def __init__(self, *input):
        self.data = list(*input)
        self.is_pinned = False

    def pin_memory(self):
        self.is_pinned = True
        for idx in range(len(self)):
            data = self[idx]
            if torch.is_tensor(data) or isinstance(data, __CUSTOM_DATA__):
                self[idx] = data.pin_memory()
        return self

    def to(self, device, **kwargs):
        self.device = torch.device(f"cuda:{device}")
        for idx in range(len(self)):
            data = self[idx]
            if torch.is_tensor(data) or isinstance(data, __CUSTOM_DATA__):
                if torch.is_tensor(data):
                    print(data.shape, data.is_pinned())
                self[idx] = data.to(device, non_blocking=True)
        return self

    def cpu(self, device, **kwargs):
        self.device = torch.device("cpu")
        for idx in range(len(self)):
            data = self[idx]
            if torch.is_tensor(data) or isinstance(data, __CUSTOM_DATA__):
                self[idx] = data.cpu()
        return self

    def parallel_to(self, deviceids):
        assert len(deviceids) == len(
            self), f"devices ({len(deviceids)}) should match list size ({len(self)})"
        for idx in range(len(self)):
            data = self[idx]
            if torch.is_tensor(data) or isinstance(data, __CUSTOM_DATA__):
                self[idx] = data.to(deviceids[idx], non_blocking=True)
        return self

    def __repr__(self):
        return f"DP{self.data.__repr__()}"

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value) -> None:
        self.data[idx] = value

    def __len__(self):
        return len(self.data)

    def append(self, item):
        self.data.append(item)


class BatchData(object):
    def __init__(self, data_dict={}):
        self.data = {}
        self.update(data_dict)
        self.device = 'cpu'
        self.is_pinned = False

    def update(self, items):
        self.data.update(items)

    def __getitem__(self, key: str):
        try:
            return self.data[key]
        except KeyError:
            return None

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def action_pin(self, content, **kwargs):
        return content.pin_memory()

    def pin_memory(self):
        self.is_pinned = True
        self = self.recrusive(self, self.action_pin)
        return self

    def action_to(self, content, device, non_blocking):
        return content.to(device, non_blocking=non_blocking)

    def action_cpu(self, content):
        return content.cpu()

    def to(self, device=None, non_blocking=False):
        self.device = torch.device(f"cuda:{device}")
        return self.recrusive(self, self.action_to, device=device,
                              non_blocking=non_blocking)

    def cpu(self):
        self.device = torch.device('cpu')
        return self.recrusive(self, self.action_cpu)

    def recrusive(self, data, action, **kwargs):
        if isinstance(data, (BatchData)):
            for i, content in data.items():
                if torch.is_tensor(content):
                    data[i] = action(data[i], **kwargs)
                elif isinstance(content, __CUSTOM_DATA__):
                    data[i] = self.recrusive(content, action, **kwargs)
                else:
                    continue
        elif isinstance(data, (DataParallelList)):
            for i in range(len(data)):
                content = data[i]
                if torch.is_tensor(content):
                    data[i] = action(content, **kwargs)
                elif isinstance(content, __CUSTOM_DATA__):
                    data[i] = self.recrusive(content, action, **kwargs)
                else:
                    continue
            return data

        elif isinstance(data, (MultiViewData)):
            return action(data, **kwargs)
        else:
            raise TypeError(f"Dtype {type(data)} not understood")
        return BatchData(data)

    def __repr__(self):
        return f"BatchData{self.data.__repr__()}"


class MultiViewData(object):
    def __init__(self, smat, vect=None, dtype=torch.FloatTensor, pad_len=0):
        self.data = BatchData({})
        self.dtype = dtype
        self.pad_len = pad_len
        self.is_pinned = False

        if isinstance(smat, BatchData):
            self.data["smat"] = smat

        elif sp.issparse(smat):
            self.data["smat"] = padded_inputs(smat.tocsr())

        self.data["vect"] = vect
        if vect is not None and not torch.is_tensor(vect):
            self.data["vect"] = torch.from_numpy(vect).type(dtype)

        self._valid_vect = True if vect is not None else False
        self.device = self.data["smat"].device
        self.len = self.data["smat"]['mask'].size(0)

    def __getitem__(self, idx):
        if not isinstance(idx, (int, torch.Tensor)) \
                or isinstance(idx, slice) or len(idx.shape) > 2:
            raise NotImplementedError(f"{type(idx)}: {idx} is not implemented")

        mask = self.data["smat"]["mask"][idx]
        indx = self.data["smat"]["index"][idx]
        vect = None
        if self._valid_vect:
            idx, indx = torch.unique(indx, return_inverse=True)
            vect = self.data["vect"][idx]
        return MultiViewData(BatchData({"index": indx, "mask": mask}),
                             vect, self.dtype, self.pad_len)

    def pin_memory(self):
        self.is_pinned = True
        self.data = self.data.pin_memory()
        return self

    def get_raw_vect(self):
        return self.data["vect"]

    def set_raw_vect(self, vect):
        self.data["vect"] = vect
        return vect

    def to(self, device=0, **kwargs):
        self.device = torch.device(f"cuda:{device}")
        self.data.to(device, **kwargs)
        return self

    def cpu(self):
        self.device = torch.device('cpu')
        self.data.cpu()
        return self

    def __len__(self):
        return self.len

    def scatter(self, device_ids):
        return self.chunk(len(device_ids)).parallel_to(device_ids)

    def chunk(self, chunks, dim=0):
        vect = None
        _chunks = DataParallelList()
        for smat in scatter(self.data['smat'], chunks, dim):
            mask, indx = smat["mask"], smat["index"]
            if self._valid_vect:
                idx, indx = torch.unique(indx, return_inverse=True)
                vect = self.data["vect"][idx]
            data = MultiViewData(BatchData({"index": indx, "mask": mask}),
                                 vect, self.dtype, self.pad_len)
            _chunks.append(data)
        if self.is_pinned:
            _chunks = _chunks.pin_memory()
        return _chunks

    @property
    def mask(self):
        return self.data["smat"]["mask"]

    @property
    def index(self):
        return self.data["smat"]["index"]

    @property
    def vect(self):
        if self._valid_vect:
            return self.data["vect"][self.data["smat"]["index"]]
        return self.data["smat"]["index"]

    def __repr__(self) -> str:
        _str = f"DATA"
        _str += f" M: {self.mask.shape}"
        _str += f" I: {self.index.shape}"
        _str += f" V: {self.vect.shape}"
        return _str


class FeaturesAccumulator(object):
    def __init__(self, desc="", _type="npy", _posfix=".pretrained"):
        self._type = _type
        self._posfix = _posfix
        self.num_rows = 0
        self.num_cols = 0
        self._data = []
        self.rows = []
        self.cols = []
        self.compiled = False
        self._smat = None
        self.desc = desc

    def transform(self, data, mask=None):
        if data is not None:
            data = data.cpu().numpy()
            if mask is not None and len(data.shape) > 2:
                mask = mask.cpu().numpy()
                n_rows = mask.shape[0]
                rows, cols = np.where(mask == 1)
                data = data.copy()[rows, cols, :]
            else:
                n_rows = data.shape[0]
                rows = np.arange(n_rows)
            cols = np.arange(rows.size)
            if rows.size > 0:
                self._data.append(data)
                self.rows.append(rows + self.num_rows)
                self.cols.append(cols + self.num_cols)
            self.num_cols += cols.size
            self.num_rows += n_rows

    def compile(self):
        if self.compiled:
            return
        if self.num_rows > 0:
            self._data = np.vstack(self._data)
            rows = np.concatenate(self.rows)
            cols = np.concatenate(self.cols)
            smat = sp.lil_matrix((self.num_rows, self.num_cols))
            smat[rows, cols] = 1
            self._smat = smat.tocsr()
        else:
            print("Empty array found")
            self._data = None
            self._smat = None
        del self.rows, self.cols
        self.compiled = True

    def remap(self, map_idx):
        if self.compiled:
            self._smat = map_idx.dot(self._smat)
        else:
            raise ValueError(f"{self.desc} not compiled")

    @property
    def data(self):
        if not self.compiled:
            self.compile()
        return self._data

    @property
    def smat(self):
        if not self.compiled:
            self.compile()
        return self._smat

    @property
    def stat(self):
        if self.data is not None:
            print(f"Stats are as follows")
            print(f"Calculated shape: ({self.num_rows}, {self.num_cols})")
            print(f"\t data: {self.data.shape}")
            print(f"\t smat (shape): {self.smat.shape}")
            print(f"\t smat (nnz): {self.smat.nnz}")
        else:
            print(f"{self.desc} is empty")

    @property
    def mean_pooled(self):
        if self.data is not None:
            nume = torch.from_numpy(self.smat.dot(self.data))
            return nume.type(torch.FloatTensor)
        return None

    def save(self, out_dir):
        self.stat
        print(out_dir+f"{self._posfix}.{ self._type}")
        if self.data is not None:
            save(out_dir+f"{self._posfix}", self._type, self.data)
            save(out_dir+f"{self._posfix}", "sparse", self.smat)

    def __len__(self):
        if self.compiled:
            return self.smat.shape[0]
        return 0

    def __getitem__(self, idx):
        if self.compiled and self.data is not None:
            temp_mat = self._smat[idx]
            indices = np.unique(temp_mat.indices)
            temp_mat = temp_mat.tocsc()[:, indices].tocsr()
            return MultiViewData(temp_mat, self._data[indices])
        return None

    def get_fts(self, idx):
        return self[idx]

    def hstack(self, fts, inplace=True):
        self.compile()
        if not fts.compiled:
            fts.compiled
        _data = np.vstack([self.data, fts.data])
        _smat = sp.hstack([self.smat, fts.smat], "csr")
        _other = self
        if not inplace:
            _other = copy.deepcopy(self)
        _other._data = _data
        _other._smat = _smat
        return _other


__CUSTOM_DATA__ = (DataParallelList, MultiViewData, BatchData)
