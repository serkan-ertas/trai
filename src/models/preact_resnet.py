"""PreActResNet-18 for CIFAR-10 with embedded input normalization.

This is the canonical CIFAR variant (3x3 stride-1 stem, no max-pool) used by
Madry et al., TRADES, and the AMS paper. Normalization is the first child
module so the rest of the project can attack RAW [0,1] inputs and the ε-ball
remains in pixel space (see the project spec gotcha #10).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class _Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, x):
        return (x - self.mean) / self.std


class _PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(
                in_planes,
                planes * self.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = None

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        # Pre-activation convention: shortcut takes the post-BN-ReLU tensor
        # when projection is needed, otherwise it's identity on x.
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        out = out + shortcut
        return out


class PreActResNet18(nn.Module):
    """PreActResNet-18 for CIFAR-10 with embedded input normalization.

    Layout: normalize -> 3x3 conv stem (64) -> [2,2,2,2] PreAct blocks at
    widths [64, 128, 256, 512] -> final BN + ReLU -> global avg pool ->
    linear(num_classes).

    Inputs are raw [0,1] pixel tensors of shape (N, 3, 32, 32); outputs are
    raw logits of shape (N, num_classes).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.normalize = _Normalize(_CIFAR10_MEAN, _CIFAR10_STD)

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.bn_final = nn.BatchNorm2d(512 * _PreActBlock.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * _PreActBlock.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_PreActBlock(self.in_planes, planes, s))
            self.in_planes = planes * _PreActBlock.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.normalize(x)
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn_final(out), inplace=True)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def preact_resnet18(num_classes: int = 10) -> PreActResNet18:
    """Factory for the CIFAR-10 PreActResNet-18 used throughout this project."""
    return PreActResNet18(num_classes=num_classes)
