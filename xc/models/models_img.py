from transformers import ViTModel, CLIPModel, BeitModel
from .custom_transformer import Projection
from torchvision import transforms
from .models_base import EncoderBase, Identity
import xc.libs.utils as ut
import torch.nn as nn
import torchvision
import torch


def trans(img_model):
    if img_model in ['ViT', "BeiT"]:
        preprocess = torch.nn.Sequential(
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]))
    elif img_model in ['inception_v3']:
        preprocess = torch.nn.Sequential(transforms.Resize(299))
    elif img_model in ['resnet50FPN', 'resnet101FPN']:
        preprocess = torch.nn.Sequential()
    elif img_model in ['ClipViT']:
        preprocess = torch.nn.Sequential(
            transforms.Resize(224),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]))
    else:
        print("UTILS:TRANSFORM:USING DEFAULT")
        preprocess = torch.nn.Sequential(
            transforms.Resize(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
    return torch.jit.export(preprocess)


def load_pre_trained(model):
    if model in ["Identity"]:
        return None
    elif model in ["ViT"]:
        return ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    elif model in ["BeiT"]:
        return BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
    elif model in ["Dino"]:
        return ViTModel.from_pretrained("facebook/dino-vits8")
    elif model in ["ClipViT"]:
        clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        model = nn.ModuleDict({"model": clipmodel.vision_model,
                               "project": clipmodel.visual_projection})
        return model
    elif model in ["inception_v3"]:
        return torchvision.models.inception_v3(pretrained=True, transform_input=True)
    elif model in ["resnet18"]:
        return torchvision.models.resnet18(pretrained=True)
    elif model in ["resnet50FPN", "resnet101FPN"]:
        res_file = model.replace("FPN", "")
        return ut.fasterrcnn_resnet_fpn(res_file, pretrained=True)
    elif model in ["vgg11"]:
        return torchvision.models.vgg11(pretrained=True)
    return torch.hub.load('pytorch/vision:v0.10.0', model, pretrained=True)


def Model(model_type, params):
    transf = trans(model_type)
    model = load_pre_trained(model_type)
    if model_type == "Identity":
        return Identity(params)
    if model_type == "vgg11":
        return ModelVGG(model, transf, params)
    elif model_type == "resnet18":
        return ModelResnet18(model, transf, params)
    elif model_type in ["resnet50FPN", "resnet101FPN"]:
        return ModelResnetsFPN(model, transf, params)
    elif model_type == "googlenet":
        return ModelGoogleNet(model, transf, params)
    elif model_type == "inception_v3":
        return ModelInceptionNetV3(model, transf, params)
    elif model_type in ["ViT"]:
        return ModelViT(model, transf, params)
    elif model_type in ["BeiT"]:
        return ModelBeiT(model, transf, params)
    elif model_type in ["Dino"]:
        return ModelDino(model, transf, params)
    elif model_type == "ClipViT":
        return ModelCLIP(model, transf, params)


class ImgEncoderBase(EncoderBase):
    def __init__(self, model, transf, project_dim):
        super(ImgEncoderBase, self).__init__(model, project_dim)
        self.transform = transf
        self.apply_pooling = True

    def encode(self, images):
        images = self.transform(images)
        return self.features(images).unsqueeze(1)

    def forward(self, images):
        images = self.encode(images)
        mask = torch.ones((images.size(0), images.size(1)),
                          device=images.device)
        return self.bottle_neck(images), mask

    def extra_repr(self):
        return f"apply_pooler={self.apply_pooling}"

    def set_pretrained(self):
        return PreTrainedImg(self)


class PreTrainedImg(EncoderBase):
    def __init__(self, model):
        self.model_dim = model.fts
        project_dim = model.compare_with_dim
        model = Projection(model.fts, model.fts, residual=True)
        super(PreTrainedImg, self).__init__(model, project_dim)
        self.apply_pooling = True

    def forward(self, images):
        images = self.features(images).unsqueeze(1)
        mask = torch.ones((images.size(0), images.size(1)),
                          device=images.device)
        return self.bottle_neck(images), mask

    @property
    def fts(self):
        return self.model_dim

    def freeze_params(self, *args, **kwargs):
        pass


class ModelVGG(ImgEncoderBase):
    def __init__(self, model, transf, params):
        super(ModelVGG, self).__init__(model, transf, params.project_dim)
        self.features.classifier = nn.Sequential(
            *list(self.features.classifier.children())[:2])

    def freeze_params(self):
        super(ModelVGG, self).freeze_params()
        for params in self.features.classifier.parameters():
            params.requires_grad = True

    @property
    def fts(self):
        return 4096


class ModelResnet18(ImgEncoderBase):
    def __init__(self, model, transf, params):
        super(ModelResnet18, self).__init__(model, transf, params.project_dim)
        self.features.fc = nn.Identity()
        self.optim = "Adam"

    def freeze_params(self):
        pass
        super().freeze_params()
        for params in self.features.layer4.parameters():
            params.requires_grad = True
        for params in self.features.avgpool.parameters():
            params.requires_grad = True
        pass

    @property
    def fts(self):
        return 512


class ModelGoogleNet(ImgEncoderBase):
    def __init__(self, model, transf, params):
        super(ModelGoogleNet, self).__init__(model, transf, params.project_dim)
        self.features.fc = nn.Identity()

    @property
    def fts(self):
        return 1024

    def freeze_params(self):
        super().freeze_params()
        for params in self.features.inception5b.parameters():
            params.requires_grad = True
        pass


class ModelInceptionNetV3(ImgEncoderBase):
    def __init__(self, model, transf, params):
        super(ModelInceptionNetV3, self).__init__(
            model, transf, params.project_dim)
        self.features.fc = nn.Identity()
        self.features.dropout = nn.Identity()
        self.features.AuxLogits = None

    def encode(self, images):
        images = self.transform(images)
        output = self.features(images)
        if isinstance(output, tuple):
            return output[0].unsqueeze(1)
        else:
            return output.unsqueeze(1)

    @property
    def fts(self):
        return 2048

    def freeze_params(self):
        super().freeze_params()
        for params in self.features.Mixed_7a.parameters():
            params.requires_grad = True
        for params in self.features.Mixed_7b.parameters():
            params.requires_grad = True
        for params in self.features.Mixed_7c.parameters():
            params.requires_grad = True
        pass


class ModelViT(ImgEncoderBase):
    def __init__(self, model, transf, params):
        super(ModelViT, self).__init__(model, transf, params.project_dim)
        self.features.pooler = None

    def encode(self, images):
        images = self.transform(images)
        images = self.features(images, interpolate_pos_encoding=True)
        images = images.last_hidden_state[:, 0, :]
        return images.unsqueeze(1)

    @property
    def fts(self):
        return self.features.config.hidden_size

    def freeze_params(self, keep_last=-1):
        if keep_last == -1:
            return
        super().freeze_params()
        if keep_last is None:
            return
        for params in self.features.encoder.layer[-keep_last:].parameters():
            params.requires_grad = True


class ModelBeiT(ModelViT):
    def __init__(self, model, transf, params):
        super(ModelBeiT, self).__init__(model, transf, params)

    def encode(self, images):
        images = self.transform(images)
        images = self.features(images, return_dict=True)
        return images.last_hidden_state[:, 0].unsqueeze(1)


class ModelDino(ModelViT):
    def __init__(self, model, transf, params):
        super(ModelDino, self).__init__(model, transf, params)


class ModelCLIP(ImgEncoderBase):
    def __init__(self, model, transf, params):
        self.project = model["project"]
        super(ModelCLIP, self).__init__(
            model["model"], transf, params.project_dim)

    def encode(self, images):
        images = self.transform(images)
        images = self.features(images)[1]
        images = self.project(images)
        return images.unsqueeze(1)

    @property
    def fts(self):
        return 512

    def freeze_params(self, keep_last=-1):
        if keep_last == -1:
            return
        super().freeze_params(keep_last)
        for params in self.features.post_layernorm.parameters():
            params.requires_grad = True


class ModelResnetsFPN(ImgEncoderBase):
    def __init__(self, model, transf, params):
        super(ModelResnetsFPN, self).__init__(
            model, transf, params.project_dim)
        self.apply_pooling = True
        self.output_accumuator = {}
        self.get_fts = 2
        self.features.roi_heads.detections_per_img = self.get_fts

        self.hook = self.features.roi_heads.box_head.fc6.register_forward_hook(
            ut.accumulate_output(self.output_accumuator, "fc6"))

        self.features.roi_heads.postprocess_detections = ut.postprocess_detections.__get__(
            (self.features.roi_heads, "fc6", self.output_accumuator),
            type(self.features.roi_heads))

    def encode(self, images):
        images = self.transform(images)
        _ = self.features(images)
        embs = self.output_accumuator[f'fc6_fts_{images.device}']
        return embs["vect"], embs["mask"]

    def forward(self, images, extract=False):
        emb, mask = self.encode(images)
        return self.bottle_neck(emb), mask

    @property
    def fts(self):
        return 1024
