''' PyTorch implementation of ResNet taken from 
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
and used by 
https://github.com/TAMU-VITA/ATMC/blob/master/cifar/resnet/resnet.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_DuBN import DualBatchNorm2d

class DuBIN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(DuBIN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = DualBatchNorm2d(planes - self.half)

    # def forward(self, x):
    #     split = torch.split(x, self.half, 1)
    #     # print(split[0].shape, split[1].shape, self.half)
    #     # print(self.IN, self.BN)
    #     out1 = self.IN(split[0].contiguous())
    #     out2 = self.BN(split[1].contiguous())
    #     out = torch.cat((out1, out2), 1)
    #     return out
    
    def forward(self, x):
        # print(x.shape)
        split = torch.split(x, self.half, 1)
        # print(split[0].contiguous().shape, split[1].contiguous().shape)
        if len(split) > 2:
            other = torch.cat(split[1:], dim=1)
        else:
            other = split[1]
        # print(split[0].contiguous().shape, other.contiguous().shape)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(other.contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class BasicBlock_DuBIN(nn.Module):

    def __init__(self, in_planes, mid_planes, out_planes, stride=1, ibn=None):
        super(BasicBlock_DuBIN, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if ibn == 'a':
            self.bn1 = DuBIN(mid_planes)
        else:
            self.bn1 = DualBatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = DualBatchNorm2d(out_planes)

        self.IN = nn.InstanceNorm2d(out_planes, affine=True) if ibn == 'b' else None

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
        if self.IN is not None:
            out = self.IN(out)
        out = F.relu(out)
        # print(out.size())
        return out


class BasicBlock_DuBIN_Dense(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, mid_planes, out_planes, conv_layer, stride=1, ibn=None):
        super(BasicBlock_DuBIN_Dense, self).__init__()
        self.conv1 = conv_layer(
            in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        if ibn == 'a':
            self.bn1 = DuBIN(mid_planes)
        else:
            self.bn1 = DualBatchNorm2d(mid_planes)
        self.conv2 = conv_layer(
            mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = DualBatchNorm2d(out_planes)

        self.IN = nn.InstanceNorm2d(out_planes, affine=True) if ibn == 'b' else None

        self.shortcut = nn.Sequential()
        if stride != 1 or stride == 1:
            # if stride != 1:
            #     print('stride != 1', stride)
            # else:
            #     print(in_planes, out_planes)
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                DualBatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        # print(self.conv1, self.conv2, self.shortcut)
        # print(out.shape, x.shape, self.shortcut(x).shape)
        out += self.shortcut(x)
        if self.IN is not None:
            out = self.IN(out)
        out = F.relu(out)
        # print(out.size())
        return out


class ResNet_DuBIN(nn.Module):
    def __init__(self, block, num_blocks, ibn_cfg=('a', 'a', 'a', None), num_classes=10, init_stride=1):
        '''
        For cifar (32*32) images, init_stride=1, num_classes=10/100;
        For Tiny ImageNet (64*64) images, init_stride=2, num_classes=200;
        See https://github.com/snu-mllab/PuzzleMix/blob/b7a795c1917a075a185aa7ea078bb1453636c2c7/models/preresnet.py#L65. 
        '''
        super(ResNet_DuBIN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=init_stride, padding=1, bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = DualBatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, ibn=ibn_cfg[3])
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, ibn=None):
        layers = []
        layers.append(block(self.in_planes, planes, planes, stride, None if ibn == 'b' else ibn))
        self.in_planes = planes

        for i in range(1,num_blocks):
            layers.append(block(self.in_planes, planes, planes, 1, None if (ibn == 'b' and i < num_blocks-1) else ibn))
            
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

def ResNet18_DuBIN(num_classes=10, init_stride=1, ibn_cfg=('a', 'a', 'a', None)):
    return ResNet_DuBIN(BasicBlock_DuBIN, [2, 2, 2, 2], ibn_cfg=ibn_cfg, num_classes=num_classes, init_stride=init_stride)

def ResNet18_DuBIN_b(num_classes=10, init_stride=1, ibn_cfg=('b', 'b', None, None)):
    return ResNet_DuBIN(BasicBlock_DuBIN, [2, 2, 2, 2], ibn_cfg=ibn_cfg, num_classes=num_classes, init_stride=init_stride)
    
def ResNet34_DuBIN(num_classes=10, init_stride=1, ibn_cfg=('a', 'a', 'a', None)):
    return ResNet_DuBIN(BasicBlock_DuBIN, [3,4,6,3], ibn_cfg=ibn_cfg, num_classes=num_classes, init_stride=init_stride)

def ResNet34_DuBIN_b(num_classes=10, init_stride=1, ibn_cfg=('b', 'b', None, None)):
    return ResNet_DuBIN(BasicBlock_DuBIN, [3,4,6,3], ibn_cfg=ibn_cfg, num_classes=num_classes, init_stride=init_stride)


class Purne_ResNet_DuBIN(nn.Module):
    def __init__(self, conv_layer, linear_layer, block, num_blocks, ch_list, \
                ibn_cfg=('a', 'a', 'a', None), num_classes=10):
        '''
        For cifar (32*32) images, init_stride=1, num_classes=10/100;
        For Tiny ImageNet (64*64) images, init_stride=2, num_classes=200;
        See https://github.com/snu-mllab/PuzzleMix/blob/b7a795c1917a075a185aa7ea078bb1453636c2c7/models/preresnet.py#L65. 
        '''
        super(Purne_ResNet_DuBIN, self).__init__()
        self.ch_list = ch_list
        self.in_planes = self.ch_list[0]
        self.conv_layer = conv_layer
        self.num_classes = num_classes
        self.conv1 = conv_layer(3, self.ch_list[1], kernel_size=3, stride=1, padding=1, bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(self.ch_list[1], affine=True)
        else:
            self.bn1 = DualBatchNorm2d(self.ch_list[1])

        self.layer_idx = 1
        self.layer1 = self._make_layer(block,  num_blocks[0], stride=1, ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block,  num_blocks[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block,  num_blocks[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block,  num_blocks[3], stride=2, ibn=ibn_cfg[3])
        self.linear = linear_layer(self.ch_list[-1], num_classes)

    def _make_layer(self, block, num_blocks, stride, ibn=None, prune_reg='weight', task_mode='prune'):
        layers = []
        layers.append(block(self.ch_list[self.layer_idx], self.ch_list[self.layer_idx + 1], \
                            self.ch_list[self.layer_idx + 2], stride, \
                            None if ibn == 'b' else ibn))
        self.layer_idx += 2

        for i in range(1,num_blocks):
            layers.append(block(self.ch_list[self.layer_idx], self.ch_list[self.layer_idx + 1], self.ch_list[self.layer_idx + 2], \
                            1, None if (ibn == 'b' and i < num_blocks-1) else ibn))
        self.layer_idx += 2
        return nn.Sequential(*layers)

    # Original channel shape: [3, 64, 64, 64, 64, 64, 128, 64, 128, 128, 128, 256, 128, 256, 256, 256, 512, 256, 512, 512, 512]
    # Strategy after pruning: [3, 25, 29, 12,  9, 46,  59, 46,  83,  37,  97, 133,  97, 190,  66, 199, 127, 199, 330,  43, 446]
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


# def ResNet18_DuBIN(num_classes=10, init_stride=1, ibn_cfg=('a', 'a', 'a', None)):
#     return ResNet_DuBIN(BasicBlock_DuBIN, [2, 2, 2, 2], ibn_cfg=ibn_cfg, num_classes=num_classes, init_stride=init_stride)

def Prune_ResNet18_DuBIN(conv_layer, linear_layer, init_type, ch_list, ibn_cfg=('a', 'a', 'a', None), **kwargs):
    return Purne_ResNet_DuBIN(conv_layer, linear_layer, BasicBlock_DuBIN, [2, 2, 2, 2], ch_list, ibn_cfg=ibn_cfg, **kwargs)



if __name__ == '__main__':
    from thop import profile
    # net = ResNet34_DuBIN() # GFLOPS: 1.1615, model size: 21.2821MB
    net = ResNet34_DuBIN_b() # GFLOPS: 1.1615, model size: 21.2821MB
    x = torch.randn(1,3,32,32)
    # net = ResNet34_DuBIN(num_classes=200, init_stride=2) # GFLOPS: 1.1615, model size: 21.3796MB
    # x = torch.randn(1,3,64,64)
    flops, params = profile(net, inputs=(x, ))
    y = net(x)
    print(y.size())
    print('GFLOPS: %.4f, model size: %.4fMB' % (flops/1e9, params/1e6))
