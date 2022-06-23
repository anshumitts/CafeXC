from dataclasses import replace
import torch
from .custom_dtypes import scatter


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


def parameters(self, recurse=True):
    def model_parameters(self):
        ps = self._former_parameters.values() \
            if hasattr(self, "_former_parameters") \
            else self.parameters(recurse=False)
        for p in ps:
            yield p

    for self in self.modules() if recurse else [self]:
        for p in model_parameters(self):
            yield p


class DataParallel(torch.nn.DataParallel):
    def __init__(self, module):
        super().__init__(module)
        
    def callback(self, clean=False):
        self._replicas = None
        if not clean and self.device_ids is not None:
            self._replicas = self.replicate(self.module, self.device_ids)

    def train(self, mode=True):
        self.callback(clean=True)
        return super().train(mode)

    def forward(self, *inputs, **kwargs):

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if not inputs and not kwargs:
            inputs, kwargs = ((),), ({},)
        if len(inputs) == 1:
            return self.module(*inputs[0], **kwargs[0])
        if self._replicas is None:
            self.callback()
        outputs = self.parallel_apply(self._replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.module, item)

    def setup_data_parallel(self, device_ids, device="cuda", clean=False):
        self.device_ids = device_ids
        self.src_device_obj = torch.device(device, device_ids[0])
        self.output_device = self.src_device_obj
        if len(self.device_ids) == 1:
            self.device_ids = None
        if not clean:
            self.callback()

    def to(self, element=torch.device("cuda:0")):
        super().to(element)
        return self
