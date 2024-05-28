"""
Various ResNet model architectures from: 
    - `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>` paper.
    - `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>` paper.
    - `Wide Residual Networks <https://arxiv.org/abs/1605.07146>` paper.
"""

from typing import Any, Callable, Type

import torch
import torch.nn as nn
from _utils import _ovewrite_named_param
from torchvision.models import WeightsEnum
from weights import *

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """
    Create a 3x3 convolutional layer with the specified input and output channels.

    Args:
        in_planes (int): The number of input channels.
        out_planes (int): The number of output channels.
        stride (int, optional): The stride of the convolution. Defaults to 1.
        groups (int, optional): The number of groups for grouped convolution. Defaults to 1.
        dilation (int, optional): The dilation rate of the convolution. Defaults to 1.

    Returns:
        nn.Conv2d: The 3x3 convolutional layer.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    Create a 1x1 convolutional layer with the specified input and output channels.

    Args:
        in_planes (int): The number of input channels.
        out_planes (int): The number of output channels.
        stride (int, optional): The stride of the convolution. Defaults to 1.

    Returns:
        nn.Conv2d: The 1x1 convolutional layer.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        """
        Initializes a BasicBlock object.

        Args:
            in_planes (int): The number of input channels.
            planes (int): The number of output channels.
            stride (int, optional): The stride of the convolution. Defaults to 1.
            downsample (nn.Module | None, optional): The downsampling layer. Defaults to None.
            groups (int, optional): The number of groups for grouped convolution. Defaults to 1.
            base_width (int, optional): The base width for the convolutional layers. Defaults to 64.
            dilation (int, optional): The dilation rate of the convolution. Defaults to 1.
            norm_layer (Callable[..., nn.Module] | None, optional): The normalization layer. Defaults to None.

        Raises:
            ValueError: If groups is not equal to 1 or base_width is not equal to 64.
            NotImplementedError: If dilation is greater than 1.

        Returns:
            None
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution (self.conv2)
    # while original implementation places the stride at the first 1x1 convolution (self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
    ):
        """
        Initializes a Bottleneck object.

        Args:
            in_planes (int): The number of input channels.
            planes (int): The number of output channels.
            stride (int, optional): The stride of the convolution. Defaults to 1.
            downsample (nn.Module | None, optional): The downsampling layer. Defaults to None.
            groups (int, optional): The number of groups for grouped convolution. Defaults to 1.
            base_width (int, optional): The base width for the convolutional layers. Defaults to 64.
            dilation (int, optional): The dilation rate of the convolution. Defaults to 1.
            norm_layer (Callable[..., nn.Module] | None, optional): The normalization layer. Defaults to None.

        Returns:
            None
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock | Bottleneck],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: list[bool] | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        """
        Initializes a ResNet model.

        Args:
            block (Type[BasicBlock | Bottleneck]): The type of block to use in the ResNet.
            layers (list[int]): A list of integers representing the number of layers in each block.
            num_classes (int, optional): The number of output classes. Defaults to 1000.
            zero_init_residual (bool, optional): Whether to initialize the residual connections with zeros. Defaults to False.
            groups (int, optional): The number of groups for grouped convolution. Defaults to 1.
            width_per_group (int, optional): The width of each group. Defaults to 64.
            replace_stride_with_dilation (list[bool] | None, optional): A list of booleans indicating whether to replace the 2x2 stride with a dilated convolution. Defaults to None.
            norm_layer (Callable[..., nn.Module] | None, optional): The normalization layer to use. Defaults to None.

        Raises:
            ValueError: If replace_stride_with_dilation is not None and not a 3-element tuple.

        Returns:
            None
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # Each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        # For ResNeXt
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[BasicBlock | Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """
        Creates a layer of blocks for the ResNet model.

        Args:
            block (Type[BasicBlock | Bottleneck]): The type of block to use in the layer.
            planes (int): The number of output channels in the layer.
            blocks (int): The number of blocks in the layer.
            stride (int, optional): The stride of the first block in the layer. Defaults to 1.
            dilate (bool, optional): Whether to apply dilation to the blocks in the layer. Defaults to False.

        Returns:
            nn.Sequential: A sequential module containing the created blocks.
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_planes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[BasicBlock | Bottleneck],
    layers: list[int],
    weights: WeightsEnum | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True)
        )

    return model


def resnet18(
    *, weights: ResNet18_Weights | None = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """
    Create a ResNet-18 model with optional pre-trained weights from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/abs/1512.03385>` paper.

    Args:
        weights (ResNet18_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the ResNet constructor.

    Returns:
        ResNet: The created ResNet-18 model.
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


def resnet34(
    *, weights: ResNet34_Weights | None = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """
    Create a ResNet-34 model with optional pre-trained weights from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/abs/1512.03385>` paper.

    Args:
        weights (ResNet34_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the ResNet constructor.

    Returns:
        ResNet: The created ResNet-34 model.
    """
    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)


def resnet50(
    *, weights: ResNet50_Weights | None = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """
    Create a ResNet-50 model with optional pre-trained weights from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/abs/1512.03385>` paper.

    Args:
        weights (ResNet50_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the ResNet constructor.

    Returns:
        ResNet: The created ResNet-50 model.
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def resnet101(
    *, weights: ResNet101_Weights | None = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """
    Create a ResNet-101 model with optional pre-trained weights from `Deep Residual Learning for Image
    Recognition <https://arxiv.org/abs/1512.03385>` paper.

    Args:
        weights (ResNet101_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the ResNet constructor.

    Returns:
        ResNet: The created ResNet-101 model.
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def resnet152(
    *, weights: ResNet152_Weights | None = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """
    Create a ResNet-152 model with optional pre-trained weights from the `Deep Residual Learning for Image
    Recognition <https://arxiv.org/abs/1512.03385>` paper.

    Args:
        weights (ResNet152_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the ResNet constructor.

    Returns:
        ResNet: The created ResNet-152 model.
    """
    return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)


def resnext50_32x4d(
    *,
    weights: ResNeXt50_32X4D_Weights | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """
    Create a ResNeXt-50 model with 32 groups and 4 width per group from `Aggregated Residual Transformation for
    Deep Neural Networks <https://arxiv.org/abs/1611.05431>` paper.

    Args:
        weights (ResNeXt50_32X4D_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the ResNet constructor.

    Returns:
        ResNet: The created ResNeXt-50 model.
    """
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def resnext101_32x8d(
    *,
    weights: ResNeXt101_32X8D_Weights | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """
    Create a ResNeXt-101 model with 32 groups and 8 width per group from `Aggregated Residual Transformation for
    Deep Neural Networks <https://arxiv.org/abs/1611.05431>` paper.

    Args:
        weights (ResNeXt101_32X8D_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the ResNet constructor.

    Returns:
        ResNet: The created ResNeXt-101 model.
    """
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def resnext101_64x4d(
    *,
    weights: ResNeXt101_64X4D_Weights | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """
    Creates a ResNeXt-101 model with 64 groups and 4 channels per group from `Aggregated Residual Transformation for
    Deep Neural Networks <https://arxiv.org/abs/1611.05431>` paper.

    Args:
        weights (ResNeXt101_64X4D_Weights | None, optional): The pre-trained weights to use. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs: Additional keyword arguments passed to the ResNet constructor.

    Returns:
        ResNet: The created ResNeXt-101 model.
    """
    _ovewrite_named_param(kwargs, "groups", 64)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def wide_resnet50_2(
    *,
    weights: Wide_ResNet50_2_Weights | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """
    Create a Wide ResNet-50-2 model from `Wide Residual Networks <https://arxiv.org/abs/1605.07146>` paper.

    Args:
        weights (Wide_ResNet50_2_Weights | None, optional): The pretrained weights for the model. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs (Any): Additional keyword arguments to be passed to the ResNet constructor.

    Returns:
        ResNet: The created Wide ResNet-50-2 model.
    """
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def wide_resnet101_2(
    *,
    weights: Wide_ResNet101_2_Weights | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """
    Create a Wide ResNet-101-2 model from `Wide Residual Networks <https://arxiv.org/abs/1605.07146>` paper.

    Args:
        weights (Wide_ResNet101_2_Weights | None, optional): The pretrained weights for the model. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Defaults to True.
        **kwargs (Any): Additional keyword arguments to be passed to the ResNet constructor.

    Returns:
        ResNet: The created Wide ResNet-101-2 model.
    """
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)
