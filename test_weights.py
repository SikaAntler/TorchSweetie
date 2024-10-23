from torch import nn

from torchsweetie.losses import *


def print_parameters(loss_fn: nn.Module):
    print(
        loss_fn._get_name(),
        len(list(loss_fn.parameters())),
        [(n, p.shape) for n, p in loss_fn.named_parameters()],
    )


loss1 = BalancedSoftmaxLoss("/ssdata0/thliang/chaohu-74/dist.csv")
print_parameters(loss1)

loss2 = BCELoss()
print_parameters(loss2)

loss3 = BCEWithLogitsLoss()
print_parameters(loss3)

loss4 = CenterLoss(512, 44, 0.05)
print_parameters(loss4)

loss5 = CrossEntropyLoss()
print_parameters(loss5)

loss6 = EffectiveNumberLoss("/ssdata0/thliang/chaohu-74/dist.csv", 0.99)
print_parameters(loss6)

loss7 = FocalLoss(0.5, 0.5)
print_parameters(loss7)

loss8 = LogitAdjustedLoss("/ssdata0/thliang/chaohu-74/dist.csv", 0.05)
print_parameters(loss8)

loss9 = NormalizedCenterLoss(512, 44, 0.05)
print_parameters(loss9)

loss10 = ReWeightCELoss("/ssdata0/thliang/chaohu-74/dist.csv")
print_parameters(loss10)

loss11 = TauNormalizedLoss(0.05)
print_parameters(loss11)
