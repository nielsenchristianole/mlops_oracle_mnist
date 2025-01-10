import timm
import torch.nn as nn
from torch.nn import Module


def create_model(num_classes: int = 10, pretrained: bool = True) -> Module:
    """
    Creates a lightweight image classification model using MobileNetV3 from TIMM.

    Args:
        num_classes (int): Number of output classes for classification.
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        Module: PyTorch model instance.
    """
    model_name: str = "mobilenetv3_small_100"
    model: Module = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes
    )
    return model


if __name__ == "__main__":
    model = create_model(num_classes=10)
    print(model)
