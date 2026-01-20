
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    """
    Section 3.1: Visual Feature Extraction Backbone
    Uses ResNet-50 and exposes multi-level feature maps.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        net = models.resnet50(pretrained=pretrained)
        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f2, f3, f4]  # multi-scale features
