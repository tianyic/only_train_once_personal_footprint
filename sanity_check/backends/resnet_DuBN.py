''' PyTorch implementation of ResNet taken from 
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
and used by 
https://github.com/TAMU-VITA/ATMC/blob/master/cifar/resnet/resnet.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_choices = ['M', 'A']

class DualBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(DualBatchNorm2d, self).__init__()
        self.bn = nn.ModuleList([nn.BatchNorm2d(num_features), nn.BatchNorm2d(num_features)])
        self.num_features = num_features
        self.ignore_model_profiling = True

        self.route = 'M' # route images to main BN or aux BN

    def forward(self, input):
        idx = BN_choices.index(self.route)

        idxn = 1 if idx == 0 else 0
        # print('Test', self.route, idx, idxn)
        
        y = self.bn[idx](input)
        
        ori_mean = self.bn[idxn].running_mean.clone()
        ori_std = self.bn[idxn].running_var.clone()

        yn = self.bn[idxn](torch.zeros_like(input)) * 0
        
        self.bn[idxn].running_mean.data.copy_(ori_mean.data)
        self.bn[idxn].running_var.data.copy_(ori_std.data)

        y = torch.add(y, yn)
        return y

class BasicBlock_DuBN(nn.Module):

    def __init__(self, in_planes, mid_planes, out_planes, stride=1):
        super(BasicBlock_DuBN, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = DualBatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = DualBatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                DualBatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        # print(out.size())
        return out


class ResNet_DuBN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, init_stride=1):
        super(ResNet_DuBN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=init_stride, padding=1, bias=False)
        self.bn1 = DualBatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18_DuBN(num_classes=10, init_stride=1):
    return ResNet_DuBN(BasicBlock_DuBN, [2, 2, 2, 2], num_classes=num_classes, init_stride=init_stride)


if __name__ == '__main__':
    from thop import profile
    net = ResNet18_DuBN()
    net.apply(lambda m: setattr(m, 'route', 'M'))
    print(net.bn1.route)
    x = torch.randn(1,3,32,32)
    flops, params = profile(net, inputs=(x, ))
    y = net(x)
    print(y.size())
    print('GFLOPS: %.4f, model size: %.4fMB' % (flops/1e9, params/1e6))

    net.apply(lambda m: setattr(m, 'route', 'A'))
    print(net.bn1.route)
    x = torch.randn(1,3,32,32)
    flops, params = profile(net, inputs=(x, ))
    y = net(x)
    print(y.size())
    print('GFLOPS: %.4f, model size: %.4fMB' % (flops/1e9, params/1e6))