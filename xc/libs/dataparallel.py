import torch
from .custom_dtypes import scatter
import time


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend(() for _ in range(len(kwargs) - len(inputs)))
    elif len(kwargs) < len(inputs):
        kwargs.extend({} for _ in range(len(inputs) - len(kwargs)))
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class DataParallel(torch.nn.DataParallel):
    
    def callback(self, clean=False):
        self._replicas = None
        if not clean and len(self.device_ids) > 1:
            self._replicas = self.replicate(self.module, self.device_ids)

    def train(self, mode=True):
        self.callback(clean=True)
        return super().train(mode)

    def forward(self, *inputs, **kwargs):
        if self._replicas is None:
            self.callback()
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)

        if not inputs and not kwargs:
            inputs, kwargs = ((),), ({},)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        outputs = self.parallel_apply(self._replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.module, item)

    def setup(self, device_ids, device="cuda"):
        self.device_ids = device_ids
        self.output_device = device_ids[0]
        self.src_device_obj = torch.device(device, device_ids[0])
        self.module.to(self.src_device_obj)
