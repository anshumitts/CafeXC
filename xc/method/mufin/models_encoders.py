import copy
import torch
import xc.libs.utils as ut
from xc.libs.dataparallel import DataParallel
from xc.models.models_txt import Model as TXTModel
from xc.models.models_img import Model as IMGModel
from xc.models.models_clf import Model as CLFModel
from xc.models.models_emb import Model as EMBModel
from xc.models.models_base import Base


def Model(params, module=1):
    if module == 4:
        return MultiModalRanker(params)
    if params.txt_model == "VisualBert":
        return VisualBert(params)
    return MultiModalEncoder(params)


class ModalEncoder(Base):
    def __init__(self, params):
        super(ModalEncoder, self).__init__()
        self.txt_encoder = TXTModel(params.txt_model, params)
        self.img_encoder = IMGModel(params.img_model, params)
        self.module = params.module

    def set_for_multi_gpu(self):
        if torch.cuda.is_available():
            return DataParallel(self)
        return self

    def forward(self, batch):
        content = {}
        if batch['img'] is not None:
            vect, mask = self.img_encoder(batch['img'].get_raw_vect())
            batch['img'].set_raw_vect(vect)
            vect = batch['img'].vect
            mask = mask[batch['img'].index]*batch['img'].mask.unsqueeze(-1)
            bs, im, sq, dm = vect.size()
            vect = vect.view(bs, im*sq, dm)
            mask = mask.view(bs, im*sq)
            content["img"] = {"vect": vect, "mask": mask}

        if batch['txt'] is not None:
            txt_vect, txt_mask = self.txt_encoder(batch["txt"].vect,
                                                  batch["txt"].mask)
            content["txt"] = {"vect": txt_vect, "mask": txt_mask}
        return content

    def freeze_params(self, keep_layer=-1):
        if self.img_encoder is not None:
            self.img_encoder.freeze_params(keep_layer)
        if self.txt_encoder is not None:
            self.txt_encoder.freeze_params(keep_layer)

    def set_pretrained(self):
        self.img_encoder = self.img_encoder.set_pretrained()

    def remove_encoders(self, rm_txt=True, rm_img=True):
        if rm_txt:
            self.txt_encoder = None
        if rm_img:
            self.img_encoder = None

    def encoder_dim(self):
        return self.txt_encoder.compare_with_dim


class MultiModalEncoder(ModalEncoder):
    def __init__(self, params):
        super(MultiModalEncoder, self).__init__(params)
        _params = copy.deepcopy(params)
        _params.project_dim = self.txt_encoder.compare_with_dim
        self.merge_embds = EMBModel("DEFAULT", _params)

    def forward(self, batch, pool=True, output_attn_wts=False):
        content = super().forward(batch)
        if self.module == 3:
            dict_return = {}
            for idx, key in enumerate(content.keys()):
                dict_return[f"{key}_vect"] = content[key]["vect"]
                dict_return[f"{key}_mask"] = content[key]["mask"]
            return dict_return

        vects, masks = [], []
        for idx, key in enumerate(content.keys()):
            vects.append(content[key]["vect"])
            masks.append(content[key]["mask"])
        vects = torch.cat(vects, dim=1)
        masks = torch.cat(masks, dim=1)
        
        vect, mask, attn = self.merge_embds(vects, masks, apply_pooling=pool,
                                            output_attn_wts=output_attn_wts)
        if output_attn_wts:
            return vect, mask, attn
        else:
            return vect, mask


class VisualBert(ModalEncoder):
    def __init__(self, params):
        super(VisualBert, self).__init__(params)
        self.img_encoder.bottle_neck = torch.nn.Identity()

    def freeze_params(self):
        self.img_encoder.freeze_params()

    def forward(self, batch, pool=True, output_attn_wts=False):
        if batch['img'] is not None:
            vect, mask = self.img_encoder(batch['img'].get_raw_vect())
            batch['img'].set_raw_vect(vect)
            vect = batch['img'].vect
            mask = mask[batch['img'].index]*batch['img'].mask.unsqueeze(-1)
            bs, im, sq, dm = vect.size()
            img_vect = vect.view(bs, im*sq, dm)
            img_mask = mask.view(bs, im*sq)

        vects, masks = self.txt_encoder(batch["txt"].vect,
                                        batch["txt"].mask,
                                        img_vect, img_mask)
        if self.module == 3:
            return {"txt_vect": vects, "txt_mask": masks}
        if self.module in [1, 2, 4]:
            return vect, mask


class MultiModalRanker(Base):
    def __init__(self, params):
        super(MultiModalRanker, self).__init__()
        self.attn_encoder = Model(params)
        self.attn_encoder.freeze_params(params.freeze_layer)
        self.attn_encoder.lr_mf = params.lr_mf_enc

        self.pool_docs = True
        self.use_cross = False
        self.attn_type = params.ranker

        if "XAttnRanker" in self.attn_type:
            self.pool_docs = False
            self.use_cross = True
            _params = copy.deepcopy(params)
            _params.project_dim = self.attn_encoder.txt_encoder.compare_with_dim
            self.cross_encoder = EMBModel("XAttnRanker", _params)
            self.cross_encoder.lr_mf = 0.01

        _params = copy.deepcopy(params)
        _params.project_dim = self.attn_encoder.txt_encoder.compare_with_dim
        self.label_clf = CLFModel(self.attn_type, _params)
        self.label_clf.lr_mf = params.lr_mf_clf

    def remove_encoders(self, rm_txt=True, rm_img=True):
        self.attn_encoder.remove_encoders(rm_txt, rm_img)

    def set_pretrained(self):
        self.attn_encoder.set_pretrained()

    def set_for_multi_gpu(self):
        return DataParallel(self)

    def _build_vects(self, data, output_attn_wts=False):
        return self.attn_encoder(data, self.pool_docs, output_attn_wts)

    def _apply_cross(self, data, output_attn_wts=False):
        return_data = {}
        if self.use_cross:
            crx_vect, _, attn_wts = self.cross_encoder(
                data['docs_vect'], data['docs_mask'],
                data['lbls_vect'], data['lbls_mask'],
                apply_pooling=self.use_cross,
                output_attn_wts=output_attn_wts)
            return_data["clf_docs"] = crx_vect
            return_data["cross_attn"] = attn_wts
        else:
            docs_vect = data['docs_vect']
            if self.pool_docs and len(docs_vect.size()) > 2:
                docs_vect = ut.mean_pooling(data['docs_vect'],
                                            data['docs_mask'])
            return_data["clf_docs"] = docs_vect.unsqueeze(1)
            return_data["cross_attn"] = None
        return return_data

    def train_forward(self, batch):
        docs_vect, docs_mask = self._build_vects(batch['docs'])
        data = {"docs_vect": docs_vect, "docs_mask": docs_mask}
        if batch["lbls"] is not None:
            lbls_vect, lbls_mask = self._build_vects(batch["lbls"])
            data.update({"lbls_vect": lbls_vect[batch["index"]],
                         "lbls_mask": lbls_mask[batch["index"]]})
        return_data = self._apply_cross(data)
        if "XAttnRanker" in self.attn_type:
            return_data["clf_lbls"] = ut.mean_pooling(lbls_vect, lbls_mask)
        score = self.label_clf({"crx_vect": return_data,
                                "hash_map": batch["index"],
                                "lbl_indx": batch["u_lbl"]})
        return score, return_data["cross_attn"]

    def forward(self, batch, only_docs=False, only_ranker=False,
                overwrite=False, output_attn_wts=False):
        if self.training or overwrite:
            return self.train_forward(batch)
        elif only_docs:
            return self._build_vects(batch, output_attn_wts)
        elif only_ranker:
            data = {"docs_vect": batch["docs"].vect,
                    "docs_mask": batch["docs"].mask}
            if batch['lbls'] is not None:
                data.update({"lbls_vect": batch["lbls"].vect[batch["index"]],
                             "lbls_mask": batch["lbls"].mask[batch["index"]]})
            data = self._apply_cross(data, output_attn_wts)
            score = self.label_clf({"crx_vect": data,
                                    "hash_map": batch["index"],
                                    "lbl_indx": batch["u_lbl"]})
            return score, data["clf_docs"], data["cross_attn"]
        else:
            return self.train_forward(batch)

    def freeze_params(self):
        pass
