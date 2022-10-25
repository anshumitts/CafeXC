from torch.utils.data import Dataset
from copy import copy, deepcopy


class NullDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = None
        self._type = None
        self.shape = None

    def __getitem__(self, idx):
        return {"x": idx}

    def get_fts(self, *args, **kwargs):
        return None

    def __len__(self):
        return 0

    def blocks(self, *args, **kwargs):
        return None

    @property
    def filter(self):
        return lambda x, axis: x

    @property
    def type(self):
        return self._type
    
    @property
    def pretrained_vect(self):
        return 0
    
    def build_pre_trained(self, *args, **kwargs):
        return self
    
    @property
    def valid(self):
        return list(range(len(self)))

    @property
    def num_features(self):
        return -1
    
    def vstack(self, *args, **kwargs):
        pass
    
    def copy(self):
        return deepcopy(self)

