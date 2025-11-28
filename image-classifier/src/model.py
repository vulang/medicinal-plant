import torch
import torch.nn as nn
from torchvision import models
from .data import IMAGENET_MEAN, IMAGENET_STD

try:
    import timm
    from timm.data import resolve_model_data_config
except ImportError:
    timm = None
    resolve_model_data_config = None

def _build_timm_model(model_name: str, num_classes: int, pretrained: bool):
    if timm is None:
        raise ImportError(f"timm is required for {model_name}. Install it with `pip install timm`.")
    if resolve_model_data_config is None:
        raise ImportError("timm>=0.9 is required for data config resolution.")
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    model.data_config = resolve_model_data_config(model)
    return model

def _attach_default_data_config(model, img_size: int = 224):
    model.data_config = {
        "input_size": (3, img_size, img_size),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "interpolation": "bilinear",
        "crop_pct": 1.0,
        "crop_mode": "center",
    }

def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    if model_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)
        _attach_default_data_config(m, 224)
        return m
    if model_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)
        _attach_default_data_config(m, 224)
        return m
    if model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)
        _attach_default_data_config(m, 224)
        return m
    if model_name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        in_feat = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_feat, num_classes)
        _attach_default_data_config(m, 224)
        return m
    if model_name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feat, num_classes)
        _attach_default_data_config(m, 224)
        return m
    if model_name == "convnext_small":
        m = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None)
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feat, num_classes)
        _attach_default_data_config(m, 224)
        return m
    if model_name == "convnext_base":
        m = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None)
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feat, num_classes)
        _attach_default_data_config(m, 224)
        return m
    if model_name == "convnext_large":
        m = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT if pretrained else None)
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feat, num_classes)
        _attach_default_data_config(m, 224)
        return m
    if model_name == "convnextv2_base.fcmae_ft_in22k_in1k":
        return _build_timm_model(model_name, num_classes, pretrained)
    if model_name == "swin_t":
        m = models.swin_t(weights=models.Swin_T_Weights.DEFAULT if pretrained else None)
        in_feat = m.head.in_features
        m.head = nn.Linear(in_feat, num_classes)
        _attach_default_data_config(m, 224)
        return m
    if model_name == "swin_b":
        m = models.swin_b(weights=models.Swin_B_Weights.DEFAULT if pretrained else None)
        in_feat = m.head.in_features
        m.head = nn.Linear(in_feat, num_classes)
        _attach_default_data_config(m, 224)
        return m
    if model_name.startswith(("vit_", "deit_", "swin_", "beit_")):
        return _build_timm_model(model_name, num_classes, pretrained)
    raise ValueError(f"Unknown model_name: {model_name}")
