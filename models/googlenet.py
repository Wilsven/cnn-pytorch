"""
GoogLeNet (Inception v1) model architecture from `Going Deeper with Convolutions 
<http://arxiv.org/abs/1409.4842>` paper.
"""

import warnings
from typing import Any, Callable

import torch
import torch.nn.functional as F
from _utils import _ovewrite_named_param
from torch import nn
from weights import *

__all__ = [
    "GoogLeNet",
    "GoogLeNet_Weights",
    "googlenet",
]


class GoogLeNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        init_weights: bool = True,
        blocks: list[Callable[..., nn.Module]] | None = None,
        dropout: float = 0.2,
        dropout_aux: float = 0.7,
    ) -> None:
        """
        Initializes the GoogleNet model.

        Args:
            num_classes (int, optional): The number of classes. Defaults to 1000.
            aux_logits (bool, optional): Whether to include auxiliary logits. Defaults to True.
            transform_input (bool, optional): Whether to transform the input. Defaults to False.
            init_weights (bool, optional): Whether to initialize the weights. Defaults to True.
            blocks (list[Callable[..., nn.Module]], optional): The blocks to use. Defaults to None.
            dropout (float, optional): The dropout rate. Defaults to 0.2.
            dropout_aux (float, optional): The dropout rate for auxiliary logits. Defaults to 0.7.

        Returns:
            None
        """
        super().__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]

        if len(blocks) != 3:
            raise ValueError(f"blocks length should be 3 instead of {len(blocks)}")

        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes, dropout=dropout_aux)
            self.aux2 = inception_aux_block(528, num_classes, dropout=dropout_aux)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # N x 3 x 224 x 224
        x = self.conv1(x)
        aux2 = x
        aux1 = x

        return x, aux2, aux1

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | torch.Tensor:
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        if not self.aux_logits:
            return x, aux2, aux1
        else:
            return x


class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Callable[..., nn.Module] | None = None,
    ) -> None:
        """
        Initializes the Inception module.

        Args:
            in_channels (int): The number of input channels.
            ch1x1 (int): The number of channels for the 1x1 convolution branch.
            ch3x3red (int): The number of channels for the reduction 1x1 convolution branch.
            ch3x3 (int): The number of channels for the 3x3 convolution branch.
            ch5x5red (int): The number of channels for the reduction 1x1 convolution branch.
            ch5x5 (int): The number of channels for the 5x5 convolution branch.
            pool_proj (int): The number of channels for the max pooling and 1x1 convolution branch.
            conv_block (Callable[..., nn.Module] | None, optional): The convolution block to use. Defaults to None.

        Returns:
            None
        """
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )

    def _forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forwards(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Callable[..., nn.Module] | None = None,
        dropout: float = 0.7,
    ) -> None:
        """
        Initializes the InceptionAux module.

        Args:
            in_channels (int): The number of input channels.
            num_classes (int): The number of output classes.
            conv_block (Callable[..., nn.Module] | None, optional): The convolution block to use. Defaults to None.
            dropout (float, optional): The dropout rate. Defaults to 0.7.

        Returns:
            None
        """
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1048, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = self.dropout(x)
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        """
        Initializes a BasicConv2d module.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            **kwargs (Any): Additional keyword arguments to be passed to the nn.Conv2d module.

        Returns:
            None
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

    """
    Create a GoogleNet model with optional pre-trained weights.

    Args:
        weights (GoogLeNet_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs (Any): Additional keyword arguments passed to the GoogLeNet constructor.

    Returns:
        GoogLeNet: The created GoogleNet model.

    Raises:
        ValueError: If the number of blocks is not 3.

    Note:
        If `weights` is not None, the `num_classes` parameter will be overwritten with the number of categories
        in the pre-trained weights. Additionally, the `transform_input` parameter will be set to True if not provided,
        and the `aux_logits` parameter will be set to True. The `init_weights` parameter will be set to False.

        If `weights` is not None and `original_aux_logits` is False, the `aux_logits` attribute of the model will be
        set to False, and the `aux1` and `aux2` attributes will be set to None. Otherwise, a warning will be raised.
    """


def googlenet(
    *, weights: GoogLeNet_Weights | None = None, progress: bool = True, **kwargs: Any
) -> GoogLeNet:
    original_aux_logits = kwargs.get("aux_logits", False)
    if weights is not None:
        if "transform_input" not in kwargs:
            _ovewrite_named_param(kwargs, "transform_input", True)
        _ovewrite_named_param(kwargs, "aux_logits", True)
        _ovewrite_named_param(kwargs, "init_weights", False)
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = GoogLeNet(**kwargs)

    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True)
        )
        if not original_aux_logits:
            model.aux_logits = False
            model.aux1 = None  # type: ignore[assignment]
            model.aux2 = None  # type: ignore[assignment]
        else:
            warnings.warn(
                "auxiliary heads in the pretrained googlenet model are NOT pretrained, so make sure to train them"
            )

    return model
