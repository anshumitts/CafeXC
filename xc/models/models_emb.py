from xc.models.custom_transformer import RTEncoder, RTDecoder
import torch.nn as nn


def Model(model_fname, params):
    if "XAttnRanker" in model_fname:
        return RankInstances(params)
    return MergeInstances(params)


class MergeInstances(nn.Module):
    def __init__(self, params):
        super(MergeInstances, self).__init__()
        self.setup(params)
        self.optim = "Adam"

    def setup(self, params):
        self.features = RTEncoder(params.project_dim, params.head_dims,
                                  n_heads=params.n_heads,
                                  n_layers=params.n_layer,
                                  dropout=params.dropout)

    def remove_encoder(self):
        self.features = nn.Sequential()

    def forward(self, vect, mask, apply_pooling=True, output_attn_wts=False):
        return self.features(vect, mask, apply_pooling, output_attn_wts)


class RankInstances(MergeInstances):
    def __init__(self, params):
        super(RankInstances, self).__init__(params)

    def setup(self, params):
        self.features = RTDecoder(params.project_dim, params.head_dims,
                                  n_heads=params.n_heads,
                                  n_layers=params.n_layer,
                                  dropout=params.dropout)

    def forward(self, itm1, itm1_mask, itm2, itm2_mask, apply_pooling=True, output_attn_wts=False):
        return self.features(itm1, itm1_mask, itm2, itm2_mask, apply_pooling, output_attn_wts)

