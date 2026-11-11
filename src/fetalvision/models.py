import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights




def build_model(name: str, num_classes: int, pretrained: bool = True,) -> nn.Module:
    """Build an image classification model."""

    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {name}")

    return model

