from xc.libs.dataparallel import DataParallel
import torch.nn.functional as F
import torch.nn as nn
import torch


def Model(mode_type, params):
    if "XAttnRankerv" in mode_type:
        return XAttnv(params)
    if "XAttnRanker" in mode_type:
        return XAttn(params)
    return CLFEmbeddings(params)


def l2_norm(vect, dim=-1):
    return F.normalize(vect, dim=dim)


class CLFBase(nn.Module):
    def __init__(self, norm=l2_norm):
        super(CLFBase, self).__init__()
        self.sparse = False
        self.padd = True
        self.norm = norm
        self.optim = "Adam"
        self.register_buffer("__preset__", None, persistent=False)
        self.register_buffer("__empty__", None, persistent=False)

    def train(self, mode=True):
        if mode:
            self.__preset__ = self.__empty__
        return super().train(mode)

    def set_for_multi_gpu(self):
        module = DataParallel(self)
        module.sparse = self.sparse
        return module

    def train_forward(self, *args, **kwargs):
        raise NotImplementedError("train forward not implelemented")

    def predict_forward(self, batch):
        index = batch["lbl_indx"][batch["hash_map"]]
        clf_vect = F.embedding(index, self.__preset__)
        crx_vect = batch['crx_vect']['clf_docs']
        crx_vect = self.norm(crx_vect, dim=-1)
        return (clf_vect*crx_vect).sum(dim=-1)

    def forward(self, batch, overwrite=False):
        if self.training or overwrite:
            return self.train_forward(batch)
        return self.predict_forward(batch)


class CLFEmbeddings(CLFBase):
    def __init__(self, params, norm=l2_norm):
        super(CLFEmbeddings, self).__init__(norm=norm)
        self.sparse = True
        self.features = nn.Embedding(params.num_labels+self.padd,
                                     params.project_dim,
                                     sparse=self.sparse)
        nn.init.kaiming_uniform_(self.features.weight)

    def build_clf(self, index, *args, **kwargs):
        xml_vect = self.features(index)
        return self.norm(xml_vect, dim=-1)

    def train_forward(self, batch):
        xml_vect = self.build_clf(batch["lbl_indx"])
        xml_vect = xml_vect[batch["hash_map"]]

        crx_vect = batch['crx_vect']['clf_docs']
        crx_vect = self.norm(crx_vect, dim=-1)
        return (xml_vect*crx_vect).sum(dim=-1)

    def preset_weights(self, lbl_vect):
        self.__preset__ = self.norm(self.features.weight, dim=-1)
        self.__preset__ = self.__preset__.detach()
        if self.sparse:
            return self.__preset__[:-1]
        return self.__preset__


class XAttn(CLFEmbeddings):
    def __init__(self, params, norm=l2_norm):
        super(XAttn, self).__init__(params, norm)
        self.weights = nn.Embedding(params.num_labels+self.padd, 2,
                                    sparse=self.sparse)
        self.activation = nn.Softmax(dim=-1)
        nn.init.constant_(self.weights.weight, 0)

    def build_clf(self, index=None, lbl_vect=None):
        lbl_vect = self.norm(lbl_vect, dim=-1)
        if index is not None:
            xml_vect = super().build_clf(index)
            weights = self.weights(index)
            weights = self.activation(weights)
            xml_wts, lbl_wts = torch.chunk(weights, 2, dim=-1)
            xml_vect = xml_vect*xml_wts
            lbl_vect = lbl_vect*lbl_wts
            clf_vect = self.norm(xml_vect + lbl_vect, dim=-1)
            return clf_vect
        return lbl_vect

    def train_forward(self, batch):
        lbl_vect = batch['crx_vect']['clf_lbls']
        xml_vect = self.build_clf(batch['lbl_indx'], lbl_vect)
        xml_vect = xml_vect[batch["hash_map"]]

        crx_vect = batch['crx_vect']['clf_docs']
        crx_vect = self.norm(crx_vect, dim=-1)
        return (xml_vect*crx_vect).sum(dim=-1)

    def preset_weights(self, lbl_vect):
        weights = self.activation(self.weights.weight)
        xml_vect = self.features.weight
        if self.padd:
            weights = weights[:-1]
            xml_vect = xml_vect[:-1]
        xml_wts, lbl_wts = torch.chunk(weights, 2, dim=-1)
        xml_vect = self.norm(xml_vect, dim=-1)*xml_wts
        lbl_vect = self.norm(lbl_vect, dim=-1)*lbl_wts
        self.__preset__ = self.norm(xml_vect + lbl_vect, dim=-1)
        self.__preset__ = self.__preset__.detach()
        return self.__preset__


class XAttnv(CLFBase):
    def __init__(self, params, norm=l2_norm):
        super(XAttnv, self).__init__(norm=norm)

    def train_forward(self, batch):
        xml_vect = batch['crx_vect']['clf_lbls']
        xml_vect = self.norm(xml_vect, dim=-1)
        xml_vect = xml_vect[batch["hash_map"]]

        crx_vect = batch['crx_vect']['clf_docs']
        crx_vect = self.norm(crx_vect, dim=-1)
        return (xml_vect*crx_vect).sum(dim=-1)

    def preset_weights(self, lbl_vect):
        if self.padd:
            padd = torch.zeros((1, lbl_vect.size(1)),
                               device=lbl_vect.device)
            lbl_vect = torch.cat([lbl_vect, padd])
        lbl_vect = self.norm(lbl_vect, dim=-1)
        self.__preset__ = self.norm(lbl_vect, dim=-1)
        self.__preset__ = self.__preset__.detach()
        return self.__preset__
