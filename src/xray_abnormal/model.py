from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    if name != "densenet121":
        raise ValueError("Only densenet121 is currently supported.")

    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    if name == "densenet121":
        return model.features.denseblock4
    raise ValueError(f"No Grad-CAM target layer configured for model: {name}")


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state" not in checkpoint:
        raise ValueError("Checkpoint must contain 'model_state'.")
    return checkpoint
