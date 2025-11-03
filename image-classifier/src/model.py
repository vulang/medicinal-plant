from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models

def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    if model_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)
        return m
    if model_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)
        return m
    if model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)
        return m
    if model_name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        in_feat = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_feat, num_classes)
        return m
    raise ValueError(f"Unknown model_name: {model_name}")
