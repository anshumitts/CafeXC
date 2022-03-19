import torch
import torch.nn as nn
import numpy as np
from torch._utils import (
    _get_all_device_indices,
    _get_available_device_type,
    _get_device_index,
)
from torch.nn.parallel import (replicate, parallel_apply, gather)
from .custom_dtypes import DataParallelList, scatter


class DistributedTraining(nn.Module):
    def __init__(self, module, num_process):
        super().__init__()
        device_type = _get_available_device_type()
        self.module = module
        self.num_process = num_process
        if device_type is None:
            self.module = module
            self.bDevIDX = None
            return
        device_ids = _get_all_device_indices()
        assert len(device_ids) % num_process == 0, "NUM DEVICES % BUCKECTS !=0"
        output_device = device_ids[0]
        self.device_ids = [_get_device_index(x, True) for x in device_ids]
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])
        self.device_type = device_type
        self.module.setup_data_parallel(self.device_ids, self.device_type)

        if len(self.device_ids) == 1 or num_process == 1:
            self.module.to(self.src_device_obj)
            self.bDevIDX = None
            return

        self.device_ids = np.array_split(self.device_ids, num_process)
        self.bDevIDX = [device[0] for device in self.device_ids]
        self._replicas = None

    def setup_id_dp(self, replicas):
        for rep_id, replica in enumerate(replicas):
            devices = self.device_ids[rep_id]
            replica.setup_data_parallel(devices, self.device_type, False)
        return replicas

    def callback(self, clean=False, num_rep=np.inf):
        self._replicas = None
        if not clean and self.bDevIDX:
            num_rep = min(num_rep, len(self.bDevIDX))
            self._replicas = replicate(self.module, self.bDevIDX[:num_rep],
                                       not torch.is_grad_enabled())
            self._replicas = self.setup_id_dp(self._replicas)
        else:
            self.module.callback(clean)

    def forward(self, input, **kwargs):

        if (not self.bDevIDX) or (not isinstance(input, DataParallelList)):
            return self.module(input, **kwargs)
        else:
            num_rep = len(input)
            if num_rep == 1:
                return self.module(input[0], **kwargs)
            gpus = self.bDevIDX[:num_rep]
            if self._replicas is None:
                self.callback(num_rep=num_rep)
            inputs = input.parallel_to(gpus)
            kwargs = scatter(kwargs, gpus)
            if len(inputs) < len(kwargs):
                inputs.extend(() for _ in range(len(kwargs) - len(inputs)))
            elif len(kwargs) < len(inputs):
                kwargs.extend({} for _ in range(len(inputs) - len(kwargs)))
            inputs = tuple(inputs)
            kwargs = tuple(kwargs)
            output = parallel_apply(self._replicas, inputs, kwargs, gpus)
        return gather(output, self.output_device)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.module, item)

    def to(self, element=None):
        if not torch.is_tensor(element):
            super().to(self.output_device)
            return self
        else:
            return element.to(self.output_device)
