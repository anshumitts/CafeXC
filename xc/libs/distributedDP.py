import numpy as np
import torch
from torch._utils import (_get_available_device_type,
                          _get_device_index,
                          _get_all_device_indices)
from .custom_dtypes import DataParallelList
from .dataparallel import scatter_kwargs




class GroupDataParallel(torch.nn.DataParallel):
    def __init__(self, module, num_process):
        super(torch.nn.DataParallel, self).__init__()
        self.device_ids = None

    def train(self, mode=True):
        self._replicas = None
        return super().train(mode)

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if not inputs and not kwargs:
            inputs, kwargs = ((),), ({},)
        outputs = self.parallel_apply(self._replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def scatter(self, inputs, kwargs, device_ids):
       return scatter_kwargs(inputs, kwargs, device_ids) 

    def setup_data_parallel(self, device_ids, device="cuda", clean=False):
        self.device_ids = device_ids[1]
        self._replicas = device_ids[0]
        self.output_device = self.src_device_obj
        if not clean:
            self.callback()
    
    def to(self, element=torch.device("cuda:0")):
        super().to(element)
        return self


def parameters(m, recurse=True):
    print("in")
    def model_parameters(m):
        ps = m._former_parameters.values() \
            if hasattr(m, "_former_parameters") \
            else m.parameters(recurse=False)
        for p in ps:
            yield p

    for m in m.modules() if recurse else [m]:
        for p in model_parameters(m):
            yield p


class DistributedTraining(torch.nn.DataParallel):
    def __init__(self, module, num_process):
        super(torch.nn.DataParallel, self).__init__()
        self.dim = 0
        self.module = module
        self.device_type = _get_available_device_type()
        device_ids = _get_all_device_indices()
        g_cl = len(device_ids)//num_process
        d_device_ids = [device_ids[x*g_cl: (x+1)*g_cl]
                        for x in range(num_process)]
        assert num_process==len(device_ids), "num_process should be eq. to num_cuda_devices or 1"
        # if num_process != len(device_ids):
            # device_ids = [_get_device_index(device_ids[x*num_process], True)
            #               for x in range(len(device_ids)//num_process)]
        self.device_ids = device_ids
        self.d_device_ids = d_device_ids
        self.output_device = _get_device_index(device_ids[0], True)


    def train(self, mode=True):
        self.callback(clean=True)
        return super().train(mode)

    def setup_data_parallel(self, replicas):
        for rep_id in range(len(replicas)):
            devices = self.d_device_ids[rep_id]
            replicas[rep_id].parameters = parameters
            replicas[rep_id].setup_data_parallel(
                devices, self.device_type, False)
        return replicas

    def callback(self, clean=False, num_rep=np.inf):
        self._replicas = None
        if not clean and self.device_ids is not None:
            num_rep = min(num_rep, len(self.device_ids))
            self._replicas = self.replicate(
                self.module, self.device_ids[:num_rep])
            self._replicas = self.setup_data_parallel(self._replicas)
            
    def forward(self, *inputs, **kwargs):

        if (not self.device_ids) or (not isinstance(inputs[0], DataParallelList)):
            return self.module(*inputs, **kwargs)
        inputs = inputs[0]
        num_rep = len(inputs)
        gpus = self.device_ids[:num_rep]
        if self._replicas is None:
            self.callback(num_rep=num_rep)
        inputs, kwargs = self.scatter(inputs, kwargs, gpus)
        output = self.parallel_apply(self._replicas[:num_rep], inputs, kwargs)
        return self.gather(output, self.output_device)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.module, item)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids)

    def to(self, element=torch.device("cuda:0")):
        super().to(element)
        return self
