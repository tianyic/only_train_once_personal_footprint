import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout, AdaptiveAvgPool2d, Conv2d, MaxPool2d
import torch.nn.functional as functional


class VGG16(nn.Module):
    def __init__(self, channels=[3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512], pooling_pos=[1, 3, 6, 9, 12], bn=False):
        # channels has 16 elms, the last 2 elms are for linear channels
        assert len(channels) == 16, 'len(channels)=%d'%(len(channels))
        super(VGG16, self).__init__()

        self.bn = bn
        self.num_classes = 10
        self.channels = channels
        self.features = self._make_features(channels, pooling_pos)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=channels[13], out_features=channels[14], bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=channels[14], out_features=channels[15], bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=channels[15], out_features=10, bias=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def _make_features(self, channels, pooling_pos):
        layers = []

        self.channels = channels
        self.pooling_pos = pooling_pos
        for i in range(len(channels) - 3):
            layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            if self.bn:
                layers.append(nn.BatchNorm2d(channels[i+1]))
            layers.append(nn.ReLU(inplace=True))
            for pos in pooling_pos:
                if i == pos:
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def get_config(self):
        return self.channels

def vgg16(cfg=[3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]):
    return VGG16(cfg)


def vgg16_bn(cfg=[3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]):
    return VGG16(cfg, bn=True)