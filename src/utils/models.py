
import timm
# from vim.models_mamba import VisionMamba  
from transformers import ConvNextV2Model, ConvNextV2Config

from collections import OrderedDict
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from omegaconf import DictConfig
from torch import Tensor

from src.utils.constants import DATA_SHAPE, INPUT_CHANNELS, NUM_CLASSES


def _largest_divisor(num_channels: int, max_groups: int) -> int:
    """Return the largest divisor of num_channels that is <= max_groups.

    GroupNorm requires num_channels % num_groups == 0.  EfficientNet-B0 has
    early layers with as few as 8 or 16 channels, so num_groups=32 would
    crash.  This helper finds the largest valid group count.
    """
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1  # fallback: InstanceNorm-equivalent


def replace_batchnorm(model: nn.Module, norm_type: str = "group", num_groups: int = 32) -> nn.Module:
    """Recursively replace all BatchNorm2d layers in *model* in-place.

    Args:
        model:      The model to modify (modified in-place and returned).
        norm_type:  ``"group"`` → GroupNorm(num_groups, C);
                    ``"layer"`` → GroupNorm(1, C)  ≡ LayerNorm for 2-D feature maps.
        num_groups: Target number of groups for GroupNorm.  Automatically
                    reduced to the largest divisor of ``num_channels`` when the
                    channel count is smaller than ``num_groups`` (e.g. early
                    EfficientNet layers with 8 or 16 channels).

    Returns:
        The modified model (same object, modified in-place).
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            if norm_type == "group":
                actual_groups = _largest_divisor(num_features, num_groups)
                new_norm = nn.GroupNorm(
                    num_groups=actual_groups,
                    num_channels=num_features,
                )
            elif norm_type == "layer":
                # GroupNorm with 1 group is equivalent to LayerNorm over the
                # spatial dimensions — standard practice for 2-D feature maps.
                new_norm = nn.GroupNorm(1, num_features)
            else:
                raise ValueError(f"Unknown norm_type '{norm_type}'. Use 'group' or 'layer'.")
            setattr(model, name, new_norm)
        else:
            replace_batchnorm(module, norm_type, num_groups)
    return model


class DecoupledModel(nn.Module):
    def __init__(self):
        super(DecoupledModel, self).__init__()
        self.need_all_features_flag = False
        self.all_features = []
        self.base: nn.Module = None
        self.classifier: nn.Module = None
        self.dropout: list[nn.Module] = []

    def need_all_features(self):
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]

        def _get_feature_hook_fn(model, input, output):
            if self.need_all_features_flag:
                self.all_features.append(output.detach().clone())

        for module in target_modules:
            module.register_forward_hook(_get_feature_hook_fn)

    def check_and_preprocess(self, args: DictConfig):
        if self.base is None or self.classifier is None:
            raise RuntimeError(
                "You need to re-write the base and classifier in your custom model class."
            )
        self.dropout = [
            module for module in self.modules() if isinstance(module, nn.Dropout)
        ]
        if args.common.buffers == "global":
            for module in self.modules():
                if isinstance(
                    module,
                    (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
                ):
                    buffers_list = list(module.named_buffers())
                    for name_buffer, buffer in buffers_list:
                        # transform buffer to parameter
                        # for showing out in model.parameters()
                        delattr(module, name_buffer)
                        module.register_parameter(
                            name_buffer,
                            torch.nn.Parameter(buffer.float(), requires_grad=False),
                        )

    def forward(self, x: Tensor) -> Tensor:
        # print(f"x.shape: {x.shape}") # torch.Size([64, 3, 224, 224])
        return self.classifier(self.base(x))

    def get_last_features(self, x: Tensor, detach=True) -> Tensor:
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
        try:
            out = self.base(x)
        except RuntimeError as err:
            if x.shape[1] == 1:
                x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
                try:
                    out = self.base(x)
                except RuntimeError as err:
                    raise RuntimeError(
                        f"Seems {self.__class__.__name__} does not support this dataset. Data resizing may help."
                    ) from err
            else:
                raise RuntimeError(
                    f"Seems {self.__class__.__name__} does not support this dataset."
                ) from err
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return func(out)

    def get_all_features(self, x: Tensor) -> Optional[list[Tensor]]:
        feature_list = None
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        self.need_all_features_flag = True
        try:
            _ = self.base(x)
        except RuntimeError as err:
            if x.shape[1] == 1:
                x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
                try:
                    _ = self.base(x)
                except RuntimeError as err:
                    raise RuntimeError(
                        f"Seems {self.__class__.__name__} does not support this dataset. Data resizing may help."
                    ) from err
            else:
                raise RuntimeError(
                    f"Seems {self.__class__.__name__} does not support this dataset."
                ) from err
        self.need_all_features_flag = False

        if len(self.all_features) > 0:
            feature_list = self.all_features
            self.all_features = []

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return feature_list


# CNN used in FedAvg
class FedAvgCNN(DecoupledModel):
    feature_length = {
        "mnist": 1024,
        "medmnistS": 1024,
        "medmnistC": 1024,
        "medmnistA": 1024,
        "covid19": 196736,
        "fmnist": 1024,
        "emnist": 1024,
        "femnist": 1,
        "cifar10": 1600,
        "cinic10": 1600,
        "cifar100": 1600,
        "tiny_imagenet": 3200,
        "celeba": 133824,
        "svhn": 1600,
        "usps": 800,
    }

    def __init__(self, dataset: str, pretrained):
        super(FedAvgCNN, self).__init__()
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 32, 5),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(32, 64, 5),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(self.feature_length[dataset], 512),
                activation3=nn.ReLU(),
            )
        )
        self.classifier = nn.Linear(512, NUM_CLASSES[dataset])


class LeNet5(DecoupledModel):
    feature_length = {
        "mnist": 256,
        "medmnistS": 256,
        "medmnistC": 256,
        "medmnistA": 256,
        "covid19": 49184,
        "fmnist": 256,
        "emnist": 256,
        "femnist": 256,
        "cifar10": 400,
        "cinic10": 400,
        "svhn": 400,
        "cifar100": 400,
        "celeba": 33456,
        "usps": 200,
        "tiny_imagenet": 2704,
    }

    def __init__(self, dataset: str, pretrained):
        super(LeNet5, self).__init__()
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 6, 5),
                bn1=nn.BatchNorm2d(6),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(6, 16, 5),
                bn2=nn.BatchNorm2d(16),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(self.feature_length[dataset], 120),
                activation3=nn.ReLU(),
                fc2=nn.Linear(120, 84),
                activation4=nn.ReLU(),
            )
        )

        self.classifier = nn.Linear(84, NUM_CLASSES[dataset])


class TwoNN(DecoupledModel):
    feature_length = {
        "mnist": 784,
        "medmnistS": 784,
        "medmnistC": 784,
        "medmnistA": 784,
        "fmnist": 784,
        "emnist": 784,
        "femnist": 784,
        "cifar10": 3072,
        "cinic10": 3072,
        "svhn": 3072,
        "cifar100": 3072,
        "usps": 1536,
        "synthetic": DATA_SHAPE["synthetic"],
    }

    def __init__(self, dataset: str, pretrained):
        super(TwoNN, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(self.feature_length[dataset], 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
        )
        # self.base = nn.Linear(features_length[dataset], 200)
        self.classifier = nn.Linear(200, NUM_CLASSES[dataset])

    def need_all_features(self):
        return

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(self.base(x))
        return x

    def get_last_features(self, data, detach=True):
        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        data = torch.flatten(data, start_dim=1)
        data = self.base(data)
        return func(data)

    def get_all_features(self, x):
        raise RuntimeError("2NN has 0 Conv layer, so is unable to get all features.")


class AlexNet(DecoupledModel):
    def __init__(self, dataset, pretrained):
        super().__init__()

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        alexnet = models.alexnet(
            weights=models.AlexNet_Weights.DEFAULT if pretrained else None
        )
        self.base = alexnet
        self.classifier = nn.Linear(
            alexnet.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        self.base.classifier[-1] = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # if input is grayscale, repeat it to 3 channels
        if x.shape[1] == 1:
            x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
        return super().forward(x)


class SqueezeNet(DecoupledModel):
    def __init__(self, version, dataset, pretrained):
        super().__init__()

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        archs = {
            "0": (models.squeezenet1_0, models.SqueezeNet1_0_Weights.DEFAULT),
            "1": (models.squeezenet1_1, models.SqueezeNet1_1_Weights.DEFAULT),
        }
        squeezenet: models.SqueezeNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = squeezenet.features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(
                squeezenet.classifier[1].in_channels,
                NUM_CLASSES[dataset],
                kernel_size=1,
            ),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # if input is grayscale, repeat it to 3 channels
        if x.shape[1] == 1:
            x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
        return super().forward(x)


class DenseNet(DecoupledModel):
    def __init__(self, version, dataset, pretrained):
        super().__init__()
        archs = {
            "121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
            "161": (models.densenet161, models.DenseNet161_Weights.DEFAULT),
            "169": (models.densenet169, models.DenseNet169_Weights.DEFAULT),
            "201": (models.densenet201, models.DenseNet201_Weights.DEFAULT),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        densenet: models.DenseNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = densenet
        self.classifier = nn.Linear(
            densenet.classifier.in_features, NUM_CLASSES[dataset]
        )
        self.base.classifier = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # if input is grayscale, repeat it to 3 channels
        if x.shape[1] == 1:
            x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
        return super().forward(x)


class ResNet(DecoupledModel):
    archs = {
        "18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
        "50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
        "152": (models.resnet152, models.ResNet152_Weights.DEFAULT),
    }

    def __init__(self, version, dataset, pretrained):
        super().__init__()

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        resnet: models.ResNet = self.archs[version][0](
            weights=self.archs[version][1] if pretrained else None
        )
        self.base = resnet
        self.classifier = nn.Linear(self.base.fc.in_features, NUM_CLASSES[dataset])
        self.base.fc = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # if input is grayscale, repeat it to 3 channels
        if x.shape[1] == 1:
            x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
        return super().forward(x)


class MobileNet(DecoupledModel):
    archs = {
        "2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
        "3s": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
        "3l": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
    }

    def __init__(self, version, dataset, pretrained):
        super().__init__()
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        mobilenet = self.archs[version][0](
            weights=self.archs[version][1] if pretrained else None
        )
        self.base = mobilenet
        self.classifier = nn.Linear(
            mobilenet.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        self.base.classifier[-1] = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # if input is grayscale, repeat it to 3 channels
        if x.shape[1] == 1:
            x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
        return super().forward(x)


# EfficientNet-B7

class EfficientNet(DecoupledModel):
    archs = {
        "0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        "1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
        "2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
        "3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
        "4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
        "5": (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT),
        "6": (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT),
        "7": (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
    }

    def __init__(self, version, dataset, pretrained):
        super().__init__()

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        efficientnet: models.EfficientNet = self.archs[version][0](
            weights=self.archs[version][1] if pretrained else None
        )
        self.base = efficientnet
        self.classifier = nn.Linear(
            efficientnet.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        self.base.classifier[-1] = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # if input is grayscale, repeat it to 3 channels
        if x.shape[1] == 1:
            x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
        return super().forward(x)


class ShuffleNet(DecoupledModel):
    archs = {
        "0_5": (models.shufflenet_v2_x0_5, models.ShuffleNet_V2_X0_5_Weights.DEFAULT),
        "1_0": (models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
        "1_5": (models.shufflenet_v2_x1_5, models.ShuffleNet_V2_X1_5_Weights.DEFAULT),
        "2_0": (models.shufflenet_v2_x2_0, models.ShuffleNet_V2_X2_0_Weights.DEFAULT),
    }

    def __init__(self, version, dataset, pretrained):
        super().__init__()
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        shufflenet: models.ShuffleNetV2 = self.archs[version][0](
            weights=self.archs[version][1] if pretrained else None
        )
        self.base = shufflenet
        self.classifier = nn.Linear(shufflenet.fc.in_features, NUM_CLASSES[dataset])
        self.base.fc = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # if input is grayscale, repeat it to 3 channels
        if x.shape[1] == 1:
            x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
        return super().forward(x)


class VGG(DecoupledModel):
    archs = {
        "11": (models.vgg11, models.VGG11_Weights.DEFAULT),
        "13": (models.vgg13, models.VGG13_Weights.DEFAULT),
        "16": (models.vgg16, models.VGG16_Weights.DEFAULT),
        "19": (models.vgg19, models.VGG19_Weights.DEFAULT),
    }

    def __init__(self, version, dataset, pretrained):
        super().__init__()
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        vgg: models.VGG = self.archs[version][0](
            weights=self.archs[version][1] if pretrained else None
        )
        self.base = vgg
        self.classifier = nn.Linear(
            vgg.classifier[-1].in_features, NUM_CLASSES[dataset]
        )
        self.base.classifier[-1] = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # if input is grayscale, repeat it to 3 channels
        if x.shape[1] == 1:
            x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
        return super().forward(x)

class ViTTiny(DecoupledModel):
    def __init__(self, dataset: str, pretrained: bool):
        super().__init__()
        model = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained)

        self.base = model
        self.base.head = nn.Identity()

        self.classifier = nn.Linear(model.embed_dim, NUM_CLASSES[dataset])


class EfficientNetB0_GN(DecoupledModel):
    def __init__(self, dataset: str, pretrained: bool):
        super().__init__()
        # Load weights if pretrained
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model = replace_batchnorm(model, norm_type="group", num_groups=32)
        
        self.base = model
        self.classifier = nn.Linear(model.classifier[1].in_features, NUM_CLASSES[dataset])
        self.base.classifier[1] = nn.Identity()


class EfficientNetB0_LN(DecoupledModel):
    def __init__(self, dataset: str, pretrained: bool):
        super().__init__()
        # Load weights if pretrained
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model = replace_batchnorm(model, norm_type="layer")
        
        self.base = model
        self.classifier = nn.Linear(model.classifier[1].in_features, NUM_CLASSES[dataset])
        self.base.classifier[1] = nn.Identity()


class EfficientNetB1_GN(DecoupledModel):
    """EfficientNet-B1 with all BatchNorm2d layers replaced by GroupNorm(32).

    Mirrors EfficientNetB0_GN but uses the B1 backbone (~6.5M params vs ~4.0M).
    Enables a capacity-controlled normalization ablation: comparing B1-GN against
    B0-GN isolates the effect of model size while keeping normalization type fixed.
    """

    def __init__(self, dataset: str, pretrained: bool):
        super().__init__()
        weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b1(weights=weights)
        model = replace_batchnorm(model, norm_type="group", num_groups=32)

        self.base = model
        self.classifier = nn.Linear(model.classifier[1].in_features, NUM_CLASSES[dataset])
        self.base.classifier[1] = nn.Identity()


class EfficientNetB1_LN(DecoupledModel):
    """EfficientNet-B1 with all BatchNorm2d layers replaced by GroupNorm(1) ≡ LayerNorm.

    Mirrors EfficientNetB0_LN but uses the B1 backbone (~6.5M params vs ~4.0M).
    GroupNorm with num_groups=1 is equivalent to LayerNorm over 2-D feature maps.
    """

    def __init__(self, dataset: str, pretrained: bool):
        super().__init__()
        weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b1(weights=weights)
        model = replace_batchnorm(model, norm_type="layer")

        self.base = model
        self.classifier = nn.Linear(model.classifier[1].in_features, NUM_CLASSES[dataset])
        self.base.classifier[1] = nn.Identity()

class ViGTiny(DecoupledModel):
    """Vision GNN Tiny (ViG-Ti) wrapped as a DecoupledModel.

    Architecture (from https://github.com/jichengyuan/Vision_GNN):
      - Stem: 4× strided Conv2d stack → (B, 192, 14, 14)
      - Backbone: 12 × [Grapher (dynamic kNN, MR-Conv) + FFN] blocks
      - Head: AdaptiveAvgPool → Linear(192, num_classes)

    Normalization: BatchNorm2d throughout (same family as efficient0).
    Embedding dim: 192 (same as vit_tiny and vim_tiny for fair comparison).
    Input: 224×224 RGB (no grayscale broadcast needed for primary datasets).

    Import strategy mirrors VimTiny: Vision_GNN/ is added to sys.path so that
    ``gcn_lib`` (pure-PyTorch, no torch_geometric) can be resolved regardless
    of the working directory.
    """

    def __init__(self, dataset: str, pretrained: bool):
        super().__init__()

        import sys
        import os

        # Locate Vision_GNN/ relative to this file (src/utils/ → ../../Vision_GNN)
        _vig_repo = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "Vision_GNN")
        )
        if not os.path.isdir(_vig_repo):
            raise ImportError(
                f"Vision GNN repo not found at '{_vig_repo}'. "
                "Please clone https://github.com/jichengyuan/Vision_GNN "
                "into the FL-bench root directory."
            )
        if _vig_repo not in sys.path:
            sys.path.insert(0, _vig_repo)

        try:
            from vig import vig_ti_224_gelu
        except ImportError as exc:
            raise ImportError(
                "Could not import vig_ti_224_gelu from Vision_GNN/vig.py. "
                f"Root cause: {exc}"
            ) from exc

        # Build ViG-Ti with the correct number of output classes.
        # We will strip the prediction head below and use a plain nn.Linear
        # as self.classifier, keeping the DecoupledModel contract.
        backbone = vig_ti_224_gelu(pretrained=False, num_classes=NUM_CLASSES[dataset])

        # --- Split backbone into base + classifier ---
        # DeepGCN.forward:
        #   x = stem(x) + pos_embed
        #   for block in backbone: x = block(x)
        #   x = AdaptiveAvgPool2d(1)          → (B, 192, 1, 1)
        #   return prediction(x).squeeze()    → (B, num_classes)
        #
        # We keep stem / pos_embed / backbone blocks as self.base and replace
        # prediction with nn.Identity so the base returns (B, 192) features.
        # self.classifier is a separate nn.Linear(192, num_classes).
        self._vig_stem = backbone.stem          # Stem module
        self._vig_pos_embed = backbone.pos_embed  # nn.Parameter (1, 192, 14, 14)
        self._vig_blocks = backbone.backbone    # nn.Sequential of 12 Grapher+FFN pairs
        self.n_blocks = backbone.n_blocks       # 12

        # classifier follows DecoupledModel convention
        self.classifier = nn.Linear(192, NUM_CLASSES[dataset])

        # self.base must be set for DecoupledModel.check_and_preprocess().
        # We create a lightweight nn.Module that groups the ViG feature extractor
        # so that buffer/BN handling in check_and_preprocess works correctly.
        class _ViGBase(nn.Module):
            def __init__(self, stem, pos_embed, blocks, n_blocks):
                super().__init__()
                self.stem = stem
                self.pos_embed = pos_embed  # registered as a Parameter via backbone
                self.blocks = blocks
                self._n_blocks = n_blocks

            def forward(self, x):
                import torch.nn.functional as F
                x = self.stem(x) + self.pos_embed
                for i in range(self._n_blocks):
                    x = self.blocks[i](x)
                x = F.adaptive_avg_pool2d(x, 1)
                return x.flatten(1)  # (B, 192)

        self.base = _ViGBase(
            self._vig_stem,
            self._vig_pos_embed,
            self._vig_blocks,
            self.n_blocks,
        )

    def forward(self, x):
        return self.classifier(self.base(x))

    def get_last_features(self, x, detach=True):
        func = (lambda t: t.detach().clone()) if detach else (lambda t: t)
        return func(self.base(x))


class VimTiny(DecoupledModel):
    def __init__(self, dataset: str, pretrained: bool):
        super().__init__()
        try:
            from mamba_ssm.models.vim import VisionMamba
        except ImportError:
            try:
                from vim.models_mamba import VisionMamba
            except ImportError:
                try:
                    from models_mamba import VisionMamba
                except ImportError:
                    raise ImportError(
                        "Vim-tiny requires 'mamba_ssm' or the official 'Vim' repo. "
                        "Please clone https://github.com/hustvl/Vim and add it to your PYTHONPATH."
                    )

        self.base = VisionMamba(
            patch_size=16, 
            embed_dim=192, 
            depth=12,
            num_classes=0, 
            img_size=224,
            channels=INPUT_CHANNELS[dataset],
            if_cls_token=True,
            use_middle_cls_token=True,
            final_pool_type='mean',
            if_abs_pos_embed=True,
            bimamba_type="v2",
        )
        
        self.classifier = nn.Linear(192, NUM_CLASSES[dataset])

    def forward(self, x):
        features = self.base(x)
        if isinstance(features, dict):
            features = features["logits"]
        return self.classifier(features)

    def get_last_features(self, x, detach=True):
        func = (lambda t: t.detach().clone()) if detach else (lambda t: t)
        return func(self.base(x))


MODELS = {
    "lenet5": LeNet5,
    "avgcnn": FedAvgCNN,
    "alex": AlexNet,
    "2nn": TwoNN,
    "squeeze0": partial(SqueezeNet, version="0"),
    "squeeze1": partial(SqueezeNet, version="1"),
    "res18": partial(ResNet, version="18"),
    "res34": partial(ResNet, version="34"),
    "res50": partial(ResNet, version="50"),
    "res101": partial(ResNet, version="101"),
    "res152": partial(ResNet, version="152"),
    "dense121": partial(DenseNet, version="121"),
    "dense161": partial(DenseNet, version="161"),
    "dense169": partial(DenseNet, version="169"),
    "dense201": partial(DenseNet, version="201"),
    "mobile2": partial(MobileNet, version="2"),
    "mobile3s": partial(MobileNet, version="3s"),
    "mobile3l": partial(MobileNet, version="3l"),
    "efficient0": partial(EfficientNet, version="0"),
    "efficient1": partial(EfficientNet, version="1"),
    "efficient2": partial(EfficientNet, version="2"),
    "efficient3": partial(EfficientNet, version="3"),
    "efficient4": partial(EfficientNet, version="4"),
    "efficient5": partial(EfficientNet, version="5"),
    "efficient6": partial(EfficientNet, version="6"),
    "efficient7": partial(EfficientNet, version="7"),
    "shuffle0_5": partial(ShuffleNet, version="0_5"),
    "shuffle1_0": partial(ShuffleNet, version="1_0"),
    "shuffle1_5": partial(ShuffleNet, version="1_5"),
    "shuffle2_0": partial(ShuffleNet, version="2_0"),
    "vgg11": partial(VGG, version="11"),
    "vgg13": partial(VGG, version="13"),
    "vgg16": partial(VGG, version="16"),
    "vgg19": partial(VGG, version="19"),
    "vit_tiny": ViTTiny,
    "vig_tiny": ViGTiny,
    "vim_tiny": VimTiny,
    "efficient0_gn": EfficientNetB0_GN,
    "efficient0_ln": EfficientNetB0_LN,
    "efficient1_gn": EfficientNetB1_GN,
    "efficient1_ln": EfficientNetB1_LN,
}