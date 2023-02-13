from .custom_transformer import Projection
from xc.libs.dataparallel import DataParallel
import torch.nn as nn
import torch


class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._to_device = torch.device("cpu")
        if torch.cuda.is_available():
            self._to_device = torch.device("cuda:0")

    def to(self, element=None):
        if element is None:
            element = self._to_device
        super().to(element)
        return self

    def callback(self, clean=False):
        for modules in self.children():
            if isinstance(modules, DataParallel):
                modules.callback(clean)

    def freeze_params(self, *args, **kwargs):
        pass

    @property
    def mm_encoder(self):
        return self.item_encoder

    def save_encoder(self):
        if isinstance(self.mm_encoder, DataParallel):
            module = self.mm_encoder.module.eval()
        else:
            module = self.mm_encoder.eval()
        module.merge_embds = None
        return module.state_dict()

    def init_encoder(self, path):
        enc = torch.load(path)
        try:
            return self.mm_encoder.load_state_dict(enc)
        except RuntimeError as e:
            print("Ignoring missing keys")
            return self.mm_encoder.load_state_dict(enc, strict=False)


class BottleNeck(nn.Module):
    def __init__(self, model_dim, compare_with_dim):
        super().__init__()
        self.features = nn.Sequential()
        self.optim = "Adam"
        self.model_dim = model_dim
        if compare_with_dim != model_dim and compare_with_dim != -1:
            if model_dim > compare_with_dim:
                self.features = nn.AdaptiveMaxPool1d(compare_with_dim)
            elif model_dim < compare_with_dim:
                self.features = Projection(model_dim, compare_with_dim)
            self.model_dim = compare_with_dim

    @property
    def fts(self):
        return self.model_dim

    def forward(self, input):
        return self.features(input)


class EncoderBase(Base):
    def __init__(self, model, project_dim):
        super().__init__()
        self.features = model
        self.project_dim = project_dim
        self.bottle_neck = BottleNeck(self.fts, self.compare_with_dim)

    @property
    def compare_with_dim(self):
        return self.project_dim

    @property
    def fts(self):
        return self.project_dim

    def freeze_params(self, *args, **kwargs):
        for params in self.features.parameters():
            params.requires_grad = False

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def remove_encoder(self):
        self.features = Projection(self.fts, self.fts)

    def set_for_multi_gpu(self):
        return nn.DataParallel(self)


class Identity(EncoderBase):
    def __init__(self, params):
        self.ranker_project = params.ranker_project_dim
        super(Identity, self).__init__(
            nn.Identity(), params.ranker_project_dim)
        self.features = Projection(params.project_dim, None, residual=True)

    def forward(self, vect, mask=None):
        if mask is None:
            mask = torch.ones((vect.size(0), 1),
                              device=vect.device)
        if len(vect.size()) == 2:
            vect = vect.unsqueeze(1)
        return self.bottle_neck(self.features(vect)), mask

    def set_pretrained(self):
        return self

    def freeze_params(self, *args, **kwargs):
        pass
