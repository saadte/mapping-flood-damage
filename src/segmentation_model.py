"""Build a randomly initialised segmentation model from segmentation_models_pytorch."""

import segmentation_models_pytorch as smp
from torch import nn


def build(arch: str, encoder_name: str, in_channels: int, classes: int) -> nn.Module:
    """`arch` is the smp class name, e.g. 'Unet'."""
    return getattr(smp, arch)(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=classes)
