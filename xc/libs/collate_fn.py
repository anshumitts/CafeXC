from torch._utils import _get_all_device_indices


class collateBase(object):
    def __init__(self, dataset):
        self.dset = dataset
        self.type_dict = self.dset.type_dict
        self.num_process = 1
        self.num_devices = len(_get_all_device_indices())

    def _call_(self, batch):
        return batch

    def __call__(self, batch):
        return self._call_(batch)
    
    @property
    def num_splits(self):
        return self.num_devices // self.num_process

