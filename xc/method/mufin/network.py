import torch
import numpy as np
import xc.libs.loss as lossy
from xc.models.models_base import Base
from torch.nn.functional import normalize
from .models_encoders import Model as ENCModel


class MufinMultiModal(Base):
    def __init__(self, params):
        super(MufinMultiModal, self).__init__()
        self.doc_first = params.doc_first
        self.criterian = lossy.CustomMarginLoss(params.margin,
                                                params.neg_sample)
        self.item_encoder = ENCModel(params)
        if "Text" in params.model_fname:
            self.item_encoder.remove_encoders(False, True)
        if "Image" in params.model_fname:
            self.item_encoder.remove_encoders(True, False)
        if "PreTrained" in params.model_fname:
            self.item_encoder.set_pretrained()
        self.params = params
        self.setup(params)

    def setup(self, params):
        self.item_encoder.freeze_params(params.freeze_layer)
        self.set_for_multi_gpu()

    def set_for_multi_gpu(self):
        self.item_encoder = self.item_encoder.set_for_multi_gpu()
        self.item_encoder.lr_mf = 0.01

    def encode(self, batch, pool=True, output_attn_wts=False):
        return self.item_encoder(batch, pool=pool, output_attn_wts=output_attn_wts)

    def forward(self, batch, encode=False):
        if encode:
            return self.encode(batch)
        vect, _ = self.encode(batch['lbls'])
        lbls = normalize(vect.squeeze(1))
        vect, _ = self.encode(batch['docs'])
        docs = normalize(vect.squeeze(1))
        hard_pos, pos_mask = None, None
        if batch['hard_pos'] is not None:
            vect, _ = self.encode(batch['hard_pos'])
            hard_pos = normalize(vect.squeeze(1))
            hard_pos = hard_pos[batch["hard_pos_index"]]
            pos_mask = batch["hard_pos_mask"].to(hard_pos.device)

        if self.doc_first:
            sim_b, sim_p = self.compute_score(docs, lbls, hard_pos, pos_mask)
        else:
            sim_b, sim_p = self.compute_score(lbls, docs, hard_pos, pos_mask)
        if self.training:
            loss = self.criterian(sim_b, batch["Y"], sim_p, batch["mask"])
            if np.isnan(loss.item()):
                import pdb; pdb.set_trace()
                    
            return {"DL": loss}
        return sim_b

    def compute_score(self, lbls, docs, hard_pos=None, pos_mask=None):
        score = lbls.mm(docs.T)
        if hard_pos is not None:
            hard_pos = (lbls.unsqueeze(1)*hard_pos).sum(dim=-1)
            hard_pos.masked_fill_(pos_mask == 0, 100)
        return score, hard_pos


class MufinRanker(Base):
    def __init__(self, params):
        super(MufinRanker, self).__init__()
        self.criterian = lossy.CosineEmbeddingLoss(margin=params.cosine_margin)
        params.project_dim = params.ranker_project_dim
        self.item_encoder = ENCModel(params, 4)
        if "Text" in params.model_fname:
            self.item_encoder.remove_encoders(False, True)
        if "Image" in params.model_fname:
            self.item_encoder.remove_encoders(True, False)
        if "PreTrained" in params.model_fname:
            self.item_encoder.set_pretrained()
        self.params = params
        self.set_for_multi_gpu()

    @property
    def mm_encoder(self):
        return self.item_encoder.attn_encoder

    def set_for_multi_gpu(self):
        self.item_encoder = self.item_encoder.set_for_multi_gpu()

    def encode(self, batch, output_attn_wts=False):
        return self.item_encoder(batch, only_docs=True,
                                 output_attn_wts=output_attn_wts)

    def predict(self, batch, output_attn_wts=False):
        score, clf_docs, attn_wts = self.item_encoder(
            batch["content"], False, True, False, output_attn_wts)
        if output_attn_wts:
            return score, clf_docs, attn_wts
        return score

    def forward(self, batch, encode=False, overwrite=False, cached=False):
        if encode:
            return self.encode(batch)
        if cached:
            return self.predict(batch)
        torch.cuda.synchronize()
        output, _ = self.item_encoder(batch["content"], overwrite=overwrite)
        torch.cuda.synchronize()
        if self.training:
            return {"DL": self.criterian(output, batch["Y"], mask=batch["mask"])}
        return output

    def clf_init(self, clf_vects):
        print("INIT::CLF_EMB")
        _, D = clf_vects.shape
        clf_vects = np.vstack([clf_vects, np.zeros((1, D))])
        clf_vects = torch.from_numpy(clf_vects).type(torch.FloatTensor)
        self.item_encoder.label_clf.set_params(clf_vects)

    def preset_weights(self, lbl_clf):
        return self.item_encoder.label_clf.preset_weights(lbl_clf).detach().cpu().numpy()


"""
Ranker for MUFIN
"""


class MufinXAttnRanker(MufinRanker):
    def __init__(self, params):
        params.txt_model = "Identity"
        params.img_model = "Identity"
        super(MufinXAttnRanker, self).__init__(params)


class MufinXAttnRankerpp(MufinRanker):
    def __init__(self, params):
        super(MufinXAttnRankerpp, self).__init__(params)


class MufinXAttnRankerv(MufinRanker):
    def __init__(self, params):
        params.txt_model = "Identity"
        params.img_model = "Identity"
        super(MufinXAttnRankerv, self).__init__(params)


class MufinXAttnRankervpp(MufinRanker):
    def __init__(self, params):
        super(MufinXAttnRankervpp, self).__init__(params)


class MufinOva(MufinRanker):
    def __init__(self, params):
        params.txt_model = "Identity"
        params.img_model = "Identity"
        super(MufinOva, self).__init__(params)


class MufinOvapp(MufinRanker):
    def __init__(self, params):
        super(MufinOvapp, self).__init__(params)


r"""
Ablation Modules for MultiModalTraining
"""


class MufinMultiModalImageXC(MufinMultiModal):
    def __init__(self, params):
        super(MufinMultiModalImageXC, self).__init__(params)


class PreTrainedMufinMultiModal(MufinMultiModal):
    def __init__(self, params):
        super(PreTrainedMufinMultiModal, self).__init__(params)


class PreTrainedMufinImageXML(MufinMultiModalImageXC):
    def __init__(self, params):
        super(PreTrainedMufinImageXML, self).__init__(params)


class MufinTextXC(MufinMultiModal):
    def __init__(self, params):
        super(MufinTextXC, self).__init__(params)
