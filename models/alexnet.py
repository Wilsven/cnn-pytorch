"""
AlexNet model architecture from `One weird trick for parallelizing 
convolutional neural networks <https://arxiv.org/abs/1404.5997>` paper.

AlexNet was originally introduced in the `ImageNet Classification with
Deep Convolutional Neural Networks
<https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>`
paper. The implementation below is based instead on the "One weird trick" paper above.
"""

from typing import Any

import torch
import torch.nn as nn
from _utils import _ovewrite_named_param
from weights import AlexNet_Weights

__all__ = ["AlexNet", "alexnet"]


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        """
        Initializes the AlexNet model.

        Args:
            num_classes (int, optional): The number of classes in the classification problem. Defaults to 1000.
            dropout (float, optional): The dropout rate for the dropout layers in the classifier. Defaults to 0.5.

        Returns:
            None
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(
    *, weights: AlexNet_Weights | None = None, progress: bool = True, **kwargs: Any
) -> AlexNet:
    """
    Create an AlexNet model with optional pre-trained weights.

    Args:
        weights (AlexNet_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the AlexNet constructor.

    Returns:
        AlexNet: The created AlexNet model.
    """
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = AlexNet(**kwargs)

    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True)
        )

    return model
