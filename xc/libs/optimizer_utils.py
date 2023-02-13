from transformers import (get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup)
# (@anshumitts) NOTE: Changed optimzer to transformer from GRN
from torch.optim import SparseAdam, AdamW
from copy import deepcopy
import torch.nn as nn
import numpy as np


class Optimizer(object):
    def __init__(self, special=["bias", "LayerNorm"],
                 type_scheudle="linear", optim="Adam"):
        self.optimizer = {}
        self.scheudler = {}
        self.optim = optim
        print(f"Using {type_scheudle} scheudler")
        if type_scheudle == "linear":
            self.typ = get_linear_schedule_with_warmup
        elif type_scheudle == "cosine":
            self.typ = get_cosine_schedule_with_warmup
        self.special = special
        self.ignore = tuple([nn.LayerNorm])

    def _get_opt(self, optim, is_sparse, params):
        if optim == 'Adam':
            if is_sparse == "sparse":
                return SparseAdam(params)
            return AdamW(params, weight_decay=0.0, eps=1e-6)
        elif optim == 'AdamW':
            if is_sparse == "sparse":
                return SparseAdam(params)
            return AdamW(params, weight_decay=0.01, eps=1e-6)
        raise NotImplementedError("Unknown optimizer!")

    def construct(self, model, lr=0.01, trn_dl=[], warmup_steps=1,
                  num_epochs=10, use_scheudler=True, accumulate=1):
        self.clear()
        model_params = self.get_params(model, lr)
        warmup_steps = int(len(trn_dl)*warmup_steps)
        total_steps = int(len(trn_dl)*num_epochs/accumulate)
        string = f"""
        args=(warmup_steps={warmup_steps}, num_epochs={num_epochs}, optim={self.optim},
              lr={lr}, use_scheudler={use_scheudler}, accumulate={accumulate})
        """
        print(string)
        for optim_type in model_params.keys():
            for is_sparse in model_params[optim_type].keys():
                params = model_params[optim_type][is_sparse]
                if len(params) == 0:
                    continue
                optim = self._get_opt(optim_type, is_sparse, params)
                self.optimizer[f"{optim_type}_{is_sparse}"] = optim
        if use_scheudler:
            for key in self.optimizer.keys():
                self.scheudler[key] = self.typ(
                    self.optimizer[key], warmup_steps, total_steps)

    def get_params(self, net, lr, base_dict={"Adam": {"sparse": [],
                                                      "dense": []},
                                             "AdamW": {"dense": []}},
                   default={"sparse": False, "lr_mf": 1, "optim": "Adam"}):
        default["optim"] = self.optim
        net_params = deepcopy(base_dict)
        children = [(net, "network", deepcopy(default))]
        while len(children) > 0:
            child, parent, params = children.pop(0)

            for _key in default.keys():
                params[_key] = child.__dict__.get(_key, params[_key])
            _dict = self.get_module_params(
                list(child.named_parameters(recurse=False)) +
                list(child.named_buffers(recurse=False)),
                child, deepcopy(base_dict), parent,
                params, lr, default)

            for key1 in _dict.keys():
                for key2 in _dict[key1].keys():
                    net_params[key1][key2].extend(_dict[key1][key2])

            grand_kid = list((m, n, deepcopy(params))
                             for n, m in child.named_children())

            if len(grand_kid) > 0:
                children.extend(list((m, f"{parent}.{n}", p)
                                for m, n, p in grand_kid))
        return net_params

    def get_module_params(self, iter_child, child, module_params,
                          parent_key, parent_params, lr, default):
        _lr = lr*parent_params["lr_mf"]
        for p_name, para in iter_child:
            if para.requires_grad:
                p_name = parent_key.split(".") + [p_name]
                _optim, key = default["optim"], "dense"

                if (np.intersect1d(p_name, self.special).size > 0
                        or isinstance(child, self.ignore)):
                    _optim = self.optim

                if not (np.intersect1d(p_name, self.special).size > 0
                        or isinstance(child, self.ignore)):
                    if "features" in p_name:
                        _optim = parent_params["optim"]
                    if parent_params["sparse"]:
                        key = "sparse"
                        _optim = "Adam"
                args = {"params": para, "lr": _lr}
                module_params[_optim][key].append(args)
                print(_lr, _optim, para.shape, key)
        return module_params

    def adjust_lr(self):
        for key in self.scheudler.keys():
            self.scheudler[key].step()

    def clear(self):
        del self.optimizer, self.scheudler
        self.optimizer = {}
        self.scheudler = {}

    def step(self):
        for key in self.optimizer.keys():
            self.optimizer[key].step()

    def zero_grad(self):
        for key in self.optimizer.keys():
            self.optimizer[key].zero_grad()

    def load_state_dict(self, state_dict):
        for idx, item in state_dict.items():
            self.optimizer[idx].load_state_dict(item)

    def state_dict(self):
        return dict([(k, v.state_dict()) for k, v in self.optimizer.items()])

    @property
    def param_groups(self):
        for opt in self.optimizer.keys():
            for group in self.optimizer[opt].param_groups:
                yield group
