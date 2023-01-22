from transformers import AutoModel, CLIPModel
from .models_base import EncoderBase, Identity
import xc.libs.utils as ut
import torch


def Model(mode_type, params):

    if mode_type == "sentencebert":
        return SentenceBert(params)

    if mode_type == "ClipViT":
        return ClipViT(params)

    elif mode_type == "VisualBert":
        return VisualBert(params)

    elif mode_type in ["bert-tiny", "bert-mini-mnli"]:
        return BertTiny(params)

    elif mode_type in ["Identity"]:
        return Identity(params)

    elif mode_type == "Base":
        return TxtEncoderBase(params)


class TxtEncoderBase(EncoderBase):
    def __init__(self, model, project_dim):
        super(TxtEncoderBase, self).__init__(model, project_dim)
        self.apply_pooling = True

    def encode(self, index, mask):
        embs = self.features(index, mask)[0]
        if self.apply_pooling:
            embs = ut.mean_pooling(embs, mask).unsqueeze(1)
            mask = torch.ones((embs.size(0), 1), device=embs.device)
        return embs, mask

    def forward(self, index, mask):
        vect, mask = self.encode(index, mask)
        return self.bottle_neck(vect), mask

    def extra_repr(self):
        return f"apply_pooler={self.apply_pooling}, device={next(self.parameters()).device}"


class BertTiny(TxtEncoderBase):
    def __init__(self, params):
        features = AutoModel.from_pretrained(f"prajjwal1/{params.txt_model}")
        super().__init__(features, params.project_dim)
        self.apply_pooling = False

    @property
    def fts(self):
        return 768


class SentenceBert(TxtEncoderBase):
    def __init__(self, params):
        features = AutoModel.from_pretrained(
            "sentence-transformers/msmarco-distilbert-base-v4")
        super().__init__(features, params.project_dim)

    @property
    def fts(self):
        return 768

    def freeze_params(self, keep_last=-1):
        if keep_last == -1:
            return
        super().freeze_params()
        if keep_last is None:
            return
        for params in self.features.transformer.layer[-keep_last:].parameters():
            params.requires_grad = True


class ClipViT(TxtEncoderBase):
    def __init__(self, params):
        clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        super().__init__(clipmodel, params.project_dim)
        self.features = clipmodel.text_model
        self.project = clipmodel.text_projection

    def encode(self, index, mask):
        embs = self.project(self.features(index, mask)[1]).unsqueeze(1)
        return embs, torch.ones((embs.size(0), 1), device=embs.device)

    @property
    def fts(self):
        return 512
    
    def freeze_params(self, keep_last=-1):
        if keep_last == -1:
            return
        super().freeze_params()
        for layer in self.features.encoder.layers[-keep_last:]:
            for params in layer.parameters():
                params.requires_grad = True
        for params in self.features.post_layernorm.parameters():
            params.requires_grad = True


class VisualBert(TxtEncoderBase):
    def __init__(self, params):
        features = AutoModel.from_pretrained(
            "uclanlp/visualbert-nlvr2-coco-pre")
        super().__init__(features, params.project_dim)

    def encode(self, index, mask, visual_vect=None, visual_mask=None):
        content = {"visual_embeds": visual_vect,
                   "visual_attention_mask": visual_mask,
                   "input_ids": index, "attention_mask": mask}
        data = self.features(**content)
        embs = data.pooler_output.unsqueeze(1)
        mask = torch.ones((embs.size(0), 1), device=embs.device)
        return embs, mask

    @property
    def fts(self):
        return 768
