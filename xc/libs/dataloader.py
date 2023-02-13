from .custom_dtypes import DataParallelList
import numpy as np
import torch


class CustomSampler:
    def __init__(self, dset, shuffle=False):
        self.shuffle = shuffle
        self.dset = dset

    def __iter__(self):
        for x in self.dset.blocks(self.shuffle):
            yield x

    def __len__(self):
        return len(self.dset.blocks(self.shuffle))


class XCDataLoader(torch.utils.data.DataLoader):
    def __init__(self, data, batch_size=256, num_workers=0, shuffle=False,
                 pin_memory=False, sampler=None, collate_fn=None,
                 drop_last=False, prefetch_factor=2):
        self.num_process = 1
        super().__init__(data, batch_size=batch_size, num_workers=num_workers,
                         shuffle=shuffle, pin_memory=pin_memory,
                         sampler=sampler, collate_fn=collate_fn,
                         drop_last=drop_last, prefetch_factor=prefetch_factor)


class XCDistributedDataLoader(object):
    def __init__(self, dataloader: XCDataLoader, num_process: int):
        self.dataloader = dataloader
        self.num_process = num_process

    def __iter__(self):
        buffer = []
        for batch in self.dataloader:
            buffer.append(batch)
            if len(buffer) % self.num_process == 0:
                yield DataParallelList(buffer)
                buffer = []
        if len(buffer) > 0:
            yield DataParallelList(buffer)

    def __len__(self):
        return int(np.ceil(len(self.dataloader)/self.num_process))

    @property
    def dataset(self):
        return self.dataloader.dataset


def DataLoader(data, batch_size=256, drop_last=False, params=None,
               shuffle=False, num_workers=2, pin_memory=False,
               collate_fn=None, prefetch_factor=2, num_process=1):
    dl = XCDataLoader(data, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, pin_memory=pin_memory,
                      sampler=CustomSampler(data, shuffle),
                      collate_fn=collate_fn, drop_last=drop_last,
                      prefetch_factor=prefetch_factor)
    return dl
#     if num_process == 1:
#         return dl
#     return XCDistributedDataLoader(dl, num_process)
