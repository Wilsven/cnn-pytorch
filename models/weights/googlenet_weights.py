from functools import partial

from torchvision.models import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.transforms._presets import ImageClassification

__all__ = ["GoogLeNet_Weights"]


class GoogLeNet_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/googlenet-1378be20.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            "num_params": 6624904,
            "min_size": (15, 15),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#googlenet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.778,
                    "acc@5": 89.530,
                }
            },
            "_ops": 1.498,
            "_file_size": 49.731,
            "_docs": """These weights are ported from the original paper.""",
        },
    )
    DEFAULT = IMAGENET1K_V1
