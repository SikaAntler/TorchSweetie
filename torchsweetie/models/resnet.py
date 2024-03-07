import torch.nn as nn
from omegaconf import DictConfig
from rich import print
from torchvision.models import Weights, resnet

from ..utils import MODELS, load_weights

__all__ = [
    "resnet18",
    "resnet34",
    "resnet34_fe",
    "resnet50",
    "resnet50_fe",
    "resnet50_32x4d",
]

_pretrained_weights = {
    "resnet18": resnet.ResNet18_Weights.DEFAULT,
    "resnet34": resnet.ResNet34_Weights.DEFAULT,
    "resnet34_fe": resnet.ResNet34_Weights.DEFAULT,
    "resnet50": resnet.ResNet50_Weights.DEFAULT,
    "resnet50_fe": resnet.ResNet50_Weights.DEFAULT,
    "resnet50_32x4d": resnet.ResNeXt50_32X4D_Weights.DEFAULT,
}

_resnet_models = {
    "resnet18": resnet.resnet18,
    "resnet34": resnet.resnet34,
    "resnet34_fe": resnet.resnet34,
    "resnet50": resnet.resnet50,
    "resnet50_fe": resnet.resnet50,
    "resnet50_32x4d": resnet.resnext50_32x4d,
}

_num_features = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet34_fe": 512,
    "resnet50": 2024,
    "resnet50_fe": 2024,
    "resnet50_32x4d": 2024,
}


class ResNetFE(nn.Module):
    def __init__(
        self,
        conv1: nn.Module,
        bn1: nn.Module,
        relu: nn.Module,
        maxpool: nn.Module,
        layer1: nn.Module,
        layer2: nn.Module,
        layer3: nn.Module,
        layer4: nn.Module,
        avgpool: nn.Module,
        num_features: int,
        remap: int = 0,
    ) -> None:
        super().__init__()

        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.maxpool = maxpool
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.avgpool = avgpool

        self.remap = None
        if remap != 0:
            self.remap = nn.Linear(num_features, remap)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)

        if self.remap is not None:
            x = self.remap(x)

        return x


def _init_model(cfg: DictConfig) -> resnet.ResNet:
    model_name = cfg.name
    weights = cfg.get("weights")
    num_classes = cfg.get("num_classes")
    # TODO: 如果用M个类别做的预训练权重给N个类别用，怎么处理？

    # if weights is None:
    #     model = _resnet_models[model_name](num_classes=cfg.num_classes)
    # elif weights == "pretrained":
    #     weights: Weights = _pretrained_weights[model_name]
    #     print(
    #         f"Using [bold magenta]pretrained weights[/bold magenta] ",
    #         f"from: [bold cyan]{weights.url}[/bold cyan]",
    #     )
    #     model = _resnet_models[model_name](weights=weights)
    #     if num_classes != 1000:
    #         model.fc = nn.Linear(2048, cfg.num_classes)
    # else:
    #     load_weights(model, cfg.weights)

    if weights == "pretrained":
        weights: Weights = _pretrained_weights[model_name]
        model = _resnet_models[model_name](weights=weights)
        print(
            f"Using [bold magenta]pretrained weights[/bold magenta]",
            f"from: [bold cyan]{weights.url}[/bold cyan]",
        )
        model = _resnet_models[model_name](weights=weights)
        if num_classes != 1000:
            model.fc = nn.Linear(_num_features[model_name], cfg.num_classes)
    else:
        model = _resnet_models[model_name](num_classes=num_classes)
        if weights is not None:
            load_weights(model, cfg.weights)

    return model


@MODELS.register
def resnet18(cfg: DictConfig) -> resnet.ResNet:
    return _init_model(cfg)


@MODELS.register
def resnet34(cfg: DictConfig) -> resnet.ResNet:
    return _init_model(cfg)


@MODELS.register
def resnet34_fe(cfg: DictConfig) -> nn.Module:
    cfg.num_classes = 1  # force to do this, since fe does not need fc layer
    model = _init_model(cfg)
    remap = cfg.get("remap", 0)
    return ResNetFE(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.avgpool,
        512,
        remap,
    )


@MODELS.register
def resnet50(cfg: DictConfig) -> resnet.ResNet:
    return _init_model(cfg)


@MODELS.register
def resnet50_fe(cfg: DictConfig) -> nn.Module:
    cfg.num_classes = 1  # force to do this, since fe does not need fc layer
    model = _init_model(cfg)
    remap = cfg.get("remap", 0)
    return ResNetFE(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.avgpool,
        2048,
        remap,
    )


@MODELS.register
def resnet50_32x4d(cfg: DictConfig) -> resnet.ResNet:
    return _init_model(cfg)
