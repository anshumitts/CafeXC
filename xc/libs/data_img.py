import os
import glob
import tqdm
import torch
import base64
import numpy as np
from io import BytesIO
import scipy.sparse as sp
from copy import copy, deepcopy
import xclib.utils.sparse as xs
from PIL import Image, ImageOps, ImageFile
from torchvision.transforms.functional import to_tensor
from concurrent.futures import ThreadPoolExecutor as tpe
from xc.models.models_img import Model
from xc.libs.dataparallel import DataParallel
from xc.libs.custom_dtypes import FeaturesAccumulator
from .utils import load_file
from .data_base import NullDataset
from .custom_dtypes import MultiViewData


ImageFile.LOAD_TRUNCATED_IMAGES = True
levels = {"docs": 2, "lbls": 3, "pl": 2, "default": 2}
size_img = (128, 128)


def read_img_bin(dat: str):
    return to_tensor(Image.open(BytesIO(base64.b64decode(dat))))


def read_raw_img_bin(dat, size=size_img):
    try:
        img = Image.open(BytesIO(base64.b64decode(dat)))
        img = img.convert("RGB")
        img.thumbnail(size, Image.LANCZOS)
        final_size = img.size
        delta_w = size[0] - final_size[0]
        delta_h = size[1] - final_size[1]
        padding = (delta_w//2, delta_h//2, delta_w -
                   (delta_w//2), delta_h-(delta_h//2))
        l_img = ImageOps.expand(img, padding, fill=(255, 255, 255))
        return to_tensor(l_img)
    except Exception as e:
        print(e)
        return torch.zeros((3, size[0], size[1]))


def read_img_and_resize(dat, size=size_img):
    img = Image.open(BytesIO(base64.b64decode(dat)))
    img.thumbnail(size, Image.LANCZOS)
    return to_tensor(img)


def load_img(root, n_file, max_worker_thread=10,
             random_k=-1, img_db="images/img.bin"):
    try:
        print(f"IMG:{n_file}(keep_k={random_k})")
        read_full = os.environ['RESTRICTMEM'] == '0'
        if n_file.endswith(".pretrained"):
            if os.environ['RESTRICTMEM'] == '1':
                return MEMIMGDataset(root, n_file, read_full, random_k)
            return NPYIMGDataset(root, n_file, random_k)
        elif n_file.endswith(".vect"):
            return MEMIMGDataset(root, n_file, read_full, random_k)
        elif n_file.endswith(".bin"):
            return IMGBINDataset(root, n_file, random_k=random_k,
                                 max_thread=max_worker_thread,
                                 img_db="images/img.bin")
        raise NotImplementedError
    except FileNotFoundError as e:
        print(f"{root}/{n_file} not found!!")
        return NullDataset()


class IMGDatasetBase(NullDataset):
    def __init__(self, root, index="", random_k=-1):
        super().__init__()
        self.random_k = random_k
        index = os.path.join(root, index) + ".npz"
        self.data = self.load_hash_map(index)
        self._desc = "default"

    def load_hash_map(self, path):
        img_idx = load_file(path)
        if self.random_k > -1:
            print(f"Keeping {self.random_k} images")
            img_idx.data[:] = -1*img_idx.data[:]
            img_idx = xs.retain_topk(img_idx, k=self.random_k)
            img_idx.data[:] = -1*img_idx.data[:]
        return img_idx.tocsr()

    def get_k_imgs(self, smat):
        indices = smat.indices
        return indices

    def __getitem__(self, idx):
        flags = self.data[idx]
        ind = self.get_k_imgs(flags)
        flags = flags.tocsc()[:, ind].tocsr()
        imgs = self._load_imgs(idx, _desc=self._desc)
        return MultiViewData(flags, imgs)

    def __deepcopy__(self, memo):
        obj = copy(self)
        obj.data = deepcopy(self.data, memo)
        return obj

    def get_fts(self, idx, _desc):
        self._desc = _desc
        return self[idx]

    def __len__(self):
        return self.data.shape[0]

    @ property
    def valid(self):
        return np.where(np.ravel(self.data.getnnz(axis=1)) > 0)[0]

    def filter(self, indices, axis=0):
        if axis == 0:
            self.data = self.data[indices]

    def _load_imgs(self, indices, _desc="pl"):
        return indices

    def vstack(self, objt):
        ob1_shape = self.data.shape
        ob2_shape = objt.data.shape
        padd1 = sp.csr_matrix((ob1_shape[0], ob2_shape[1]))
        padd2 = sp.csr_matrix((ob2_shape[0], ob1_shape[1]))
        self.data = sp.vstack([sp.hstack([self.data, padd1]),
                               sp.hstack([padd2, objt.data])], 'csr')


class IMGBINDataset(IMGDatasetBase):
    def __init__(self, root, n_file, max_thread=10,
                 random_k=-1, img_db="images/img.bin"):
        super().__init__(root, n_file, random_k)
        self.max_thread = max_thread
        self.vect = os.path.join(root, img_db)
        self.read_func = read_img_bin
        self.size = size_img
        self._type = "base64"

    def _load_imgs(self, ind, _desc="pl"):
        _sort = np.argsort(ind)

        def _read(f, inds):
            f.seek(int(inds))
            data = f.readline().split(b"\t", 1)
            return data[1].strip()

        with open(self.vect, 'rb') as f, tpe(self.max_thread) as exe:
            list_img = map(lambda x: _read(f, x), ind[_sort])
            imgs = list(exe.map(self.read_func, list_img))

        imgs = torch.stack(imgs, dim=0)
        imgs[_sort] = imgs.clone()
        return imgs

    def __getitem__(self, idx):
        flags = self.data[idx]
        ind = self.get_k_imgs(flags)
        ind = np.unique(ind)
        flags = flags.tocsc()[:, ind].tocsr()
        data = flags.data[:] - 1  # NOTE: Handling offset
        if len(data) > 0:
            imgs = self._load_imgs(data, _desc=self._desc)
        else:
            len_idx = 1
            if hasattr(idx, "__len__"):
                len_idx = len(idx)
            flags = sp.eye(len_idx).tocsr()
            imgs = torch.zeros((len_idx, 3, self.size[0], self.size[1]))
        flags.data[:] = 1
        return MultiViewData(flags, imgs)

    def build_pre_trained(self, img_model, data_dir, file_name,
                          params, prefix="img.vect"):
        file_name = f"{file_name}.{prefix}"

        cached_path = os.path.join(data_dir, img_model)
        print(f"{cached_path}/{file_name}")
        if len(glob.glob(f"{cached_path}/{file_name}*")) > 0:
            return load_img(cached_path, f"{file_name}",
                            self.max_thread, self.random_k)

        def collate_fn(batch):
            imgs = torch.cat(list(map(lambda x: x.get_raw_vect(),
                                      batch)), dim=0)
            return imgs
        params.project_dim = -1
        pre_trained = Model(img_model, params)
        pre_trained.project = torch.nn.Identity()
        pre_trained = DataParallel(pre_trained)
        dl = torch.utils.data.DataLoader(
            self, batch_size=512, collate_fn=collate_fn,
            shuffle=False, num_workers=6, prefetch_factor=2)
        features = FeaturesAccumulator("Image features", "memmap", f"")
        with torch.no_grad():
            pre_trained = pre_trained.cuda()
            pre_trained = pre_trained.eval()
            with torch.cuda.amp.autocast():
                for data in tqdm.tqdm(dl):
                    embs, mask = pre_trained(data)
                    features.transform(embs, mask)
        features.compile()
        print(f"total number of images in datasets are {self.data.nnz}")
        os.makedirs(cached_path, exist_ok=True)
        features.remap(self.data)
        features.save(os.path.join(cached_path, f"{file_name}"))
        pre_trained.cpu()
        del features, pre_trained
        return load_img(cached_path, f"{file_name}",
                        self.max_thread, self.random_k)


class NPYIMGDataset(IMGDatasetBase):
    def __init__(self, root, n_file, random_k=-1):
        super().__init__(root, n_file, random_k)
        self.vect = load_file(os.path.join(root, n_file) + ".npy")
        self.shape = self.vect.shape
        self._type = "pretrained"

    @ property
    def pretrained_vect(self):
        return self.data.dot(self.vect)

    def __getitem__(self, idx):
        flags = self.data[idx]
        ind = flags.indices
        ind = np.unique(ind)
        flags = flags.tocsc()[:, ind].tocsr()
        imgs = self.vect[ind]
        return MultiViewData(flags, imgs)

    def vstack(self, objt):
        super().vstack(objt)
        self.vect = np.vstack([self.vect, objt.vect])


class MEMIMGDataset(IMGDatasetBase):
    def __init__(self, root, n_file, read_full=False, random_k=-1):
        super().__init__(root, n_file, random_k)
        self.vect = load_file(os.path.join(root, n_file) + ".memmap")
        self.shape = self.vect.shape
        self._type = "pretrained"
        if read_full:
            self.vect = np.array(self.vect[:])

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            sorted_idx = np.argsort(idx)
            idx = idx[sorted_idx]

        flags = self.data[idx]
        ind = flags.indices
        ind = np.unique(ind)
        min_ind, max_ind = min(ind), max(ind) + 1
        flags = flags.tocsc()[:, ind]
        imgs = self.vect[min_ind:max_ind][ind-min_ind]
        if not isinstance(idx, int):
            flags = flags.tolil()
            flags[sorted_idx] = flags.copy()
        return MultiViewData(flags.tocsr(), imgs)
