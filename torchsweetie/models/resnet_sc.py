from typing import Optional

import torch
from rich import print
from torch import Tensor, nn
from torchvision.models import resnet

from ..data import ClsDataPack
from ..utils import KEY_B, KEY_E, MODELS, URL_B, URL_E
from .resnet import ResNet

__all__ = [
    "resnext50_32x4d_sc",
    "ResNetSC",
]

_pretrained_weights = {
    "resnet18": resnet.ResNet18_Weights.DEFAULT,
    "resnet34": resnet.ResNet34_Weights.DEFAULT,
    "resnet50": resnet.ResNet50_Weights.DEFAULT,
    "resnet101": resnet.ResNet101_Weights.DEFAULT,
    "resnet152": resnet.ResNet152_Weights.DEFAULT,
    "resnext50_32x4d": resnet.ResNeXt50_32X4D_Weights.DEFAULT,
    "resnext101_32x8d": resnet.ResNeXt101_32X8D_Weights.DEFAULT,
    "resnext101_64x4d": resnet.ResNeXt101_64X4D_Weights.DEFAULT,
}

_resnet_models = {
    "resnet18": resnet.resnet18,
    "resnet34": resnet.resnet34,
    "resnet50": resnet.resnet50,
    "resnet101": resnet.resnet101,
    "resnet152": resnet.resnet152,
    "resnext50_32x4d": resnet.resnext50_32x4d,
    "resnext101_32x8d": resnet.resnext101_32x8d,
    "resnext101_64x4d": resnet.resnext101_64x4d,
}

_num_features = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    "resnext50_32x4d": 2048,
    "resnext101_32x8d": 2048,
    "resnext101_64x4d": 2048,
}


class ResNetSC(ResNet):
    def __init__(
        self, model: resnet.ResNet, se_features: int, num_features: int, num_classes: int
    ) -> None:
        super().__init__(model, num_features, num_classes)

        self.size_extractor = nn.Sequential(
            nn.Linear(2, se_features),
            nn.ReLU(True),
            nn.Linear(se_features, se_features),
        )

        self.feature_fuser = nn.Sequential(
            nn.Linear(num_features + se_features, num_features),
            nn.ReLU(True),
        )

        if num_classes != 0:
            if num_classes == model.fc.weight.shape[0]:
                self.fc = model.fc
            else:
                self.fc = nn.Linear(num_features, num_classes)

    def forward(self, data: ClsDataPack) -> Tensor:
        x = data.inputs

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

        # size: (B, 2)
        #   se: (2, S)
        #    x: (B, N)
        # x | size @ se = (B, N) | (B, 2) @ (2, S) = (B, N+S)
        x = torch.cat([x, self.size_extractor(data.ori_sizes)], 1)
        x = self.feature_fuser(x)  # (B, N+S) @ (N+S, N) = (B, N)
        x = self.fc(x)

        return x


def _init_model(
    model_name: str, se_features: int, num_classes: int, weights: Optional[str] = None
) -> ResNetSC:
    if weights == "torchvision":
        pretrained_weights = _pretrained_weights[model_name]
        print(
            f"Using {KEY_B}pretrained{KEY_E} weights",
            f"from {KEY_B}torchvision{KEY_E}({URL_B}{pretrained_weights.url}{URL_E})",
        )
        _model = _resnet_models[model_name](weights=pretrained_weights)
        model = ResNetSC(_model, se_features, _num_features[model_name], num_classes)
    else:
        _model = _resnet_models[model_name](num_classes=num_classes)
        model = ResNetSC(_model, se_features, _num_features[model_name], num_classes)
        if weights is not None:
            print(f"Loading weights from: {URL_B}{weights}{URL_E}")
            model.load_state_dict(torch.load(weights, "cpu"), False)

    return model


@MODELS.register()
def resnext50_32x4d_sc(
    se_features: int, num_classes: int, weights: Optional[str] = None
) -> ResNetSC:
    return _init_model("resnext50_32x4d", se_features, num_classes, weights)
