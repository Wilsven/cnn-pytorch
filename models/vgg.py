"""
VGG model architecture from `Very Deep Convolutional Networks for 
Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>` paper.
"""

from typing import Any, cast

import torch
import torch.nn as nn
from _utils import _ovewrite_named_param
from torchvision.models import WeightsEnum
from weights import (
    VGG11_BN_Weights,
    VGG11_Weights,
    VGG13_BN_Weights,
    VGG13_Weights,
    VGG16_BN_Weights,
    VGG16_Weights,
    VGG19_BN_Weights,
    VGG19_Weights,
)

__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]

CFGS: dict[str, list[str | int]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        dropout: float = 0.5,
    ) -> None:
        """
        Initializes the VGG model with the given features, number of classes, initialization flag, and dropout rate.

        Args:
            features (nn.Module): The features module of the VGG model.
            num_classes (int, optional): The number of classes. Defaults to 1000.
            init_weights (bool, optional): Flag indicating whether to initialize the weights. Defaults to True.
            dropout (float, optional): The dropout rate. Defaults to 0.5.

        Returns:
            None
        """
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: list[str | int], batch_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(**layers)


def _vgg(
    cfg: str,
    batch_norm: bool,
    weights: WeightsEnum | None = None,
    progress: bool = True,
    **kwargs: Any
) -> VGG:
    """
    Create a VGG model with the given configuration and optional pre-trained weights.

    Args:
        cfg (str): The configuration of the VGG model.
        batch_norm (bool): Whether to use batch normalization layers.
        weights (WeightsEnum | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the VGG constructor.

    Returns:
        VGG: The created VGG model.
    """
    if weights is not None:
        kwargs["init_weights"] = False
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = VGG(make_layers(CFGS[cfg], batch_norm=batch_norm), **kwargs)

    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True)
        )

    return model


def vgg11(
    *, weights: VGG11_Weights | None = None, progress: bool = True, **kwargs: Any
) -> VGG:
    """
    Create a VGG-11 model (configuration "A") with the given pre-trained weights and
    optional additional keyword arguments.

    Args:
        weights (VGG11_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the VGG constructor.

    Returns:
        VGG: The created VGG-11 model (configuration "A").
    """
    return _vgg("A", False, weights, progress, **kwargs)


def vgg11_bn(
    *, weights: VGG11_BN_Weights | None = None, progress: bool = True, **kwargs: Any
) -> VGG:
    """
    Create a VGG-11 model (configuration "A") with batch normalization layers and
    the given pre-trained weights and optional additional keyword arguments.

    Args:
        weights (VGG11_BN_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the VGG constructor.

    Returns:
        VGG: The created VGG-11 model (configuration "A") with batch normalization layers.
    """
    return _vgg("A", True, weights, progress, **kwargs)


def vgg13(
    *, weights: VGG13_Weights | None = None, progress: bool = True, **kwargs: Any
) -> VGG:
    """
    Create a VGG-13 model (configuration "B") with the given pre-trained weights and
    optional additional keyword arguments.

    Args:
        weights (VGG13_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the VGG constructor.

    Returns:
        VGG: The created VGG-13 model (configuration "B").
    """
    return _vgg("B", False, weights, progress, **kwargs)


def vgg13_bn(
    *, weights: VGG13_BN_Weights | None = None, progress: bool = True, **kwargs: Any
) -> VGG:
    """
    Create a VGG-13 model (configuration "B") with batch normalization layers and
    the given pre-trained weights and optional additional keyword arguments.

    Args:
        weights (VGG13_BN_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the VGG constructor.

    Returns:
        VGG: The created VGG-13 model (configuration "B") with batch normalization layers.
    """
    return _vgg("B", True, weights, progress, **kwargs)


def vgg16(
    *, weights: VGG16_Weights | None = None, progress: bool = True, **kwargs: Any
) -> VGG:
    """
    Create a VGG-16 model (configuration "D") with the given pre-trained weights and
    optional additional keyword arguments.

    Args:
        weights (VGG16_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the VGG constructor.

    Returns:
        VGG: The created VGG-16 model (configuration "D").
    """
    return _vgg("D", False, weights, progress, **kwargs)


def vgg16_bn(
    *, weights: VGG16_BN_Weights | None = None, progress: bool = True, **kwargs: Any
) -> VGG:
    """
    Create a VGG-16 model (configuration "D") with batch normalization layers and
    the given pre-trained weights and optional additional keyword arguments.

    Args:
        weights (VGG16_BN_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the VGG constructor.

    Returns:
        VGG: The created VGG-16 model (configuration "D") with batch normalization layers.
    """
    return _vgg("D", True, weights, progress, **kwargs)


def vgg19(
    *, weights: VGG19_Weights | None = None, progress: bool = True, **kwargs: Any
) -> VGG:
    """
    Create a VGG-19 model (configuration "E") with the given pre-trained weights and
    optional additional keyword arguments.

    Args:
        weights (VGG19_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the VGG constructor.

    Returns:
        VGG: The created VGG-19 model (configuration "E").
    """
    return _vgg("E", False, weights, progress, **kwargs)


def vgg19_bn(
    *, weights: VGG19_BN_Weights | None = None, progress: bool = True, **kwargs: Any
) -> VGG:
    """
    Create a VGG-19 model (configuration "E") with batch normalization layers and
    the given pre-trained weights and optional additional keyword arguments.

    Args:
        weights (VGG19_BN_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the VGG constructor.

    Returns:
        VGG: The created VGG-19 model (configuration "E") with batch normalization layers.
    """
    return _vgg("E", True, weights, progress, **kwargs)
