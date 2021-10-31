# Usage: python flops.py --backend resnet.resnet18 --dataset_name imagenet
import numpy as np
import argparse
import torch
import torch.nn as nn

from backend import Model

# ------------------------------------------------------------------ # 
class FConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(FConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.num_ops = 0

    def forward(self, x):
        output = super(FConv2d, self).forward(x)
        output_area = output.size(-1)*output.size(-2)
        filter_area = np.prod(self.kernel_size)
        assert (self.in_channels*self.out_channels*filter_area*output_area) % self.groups == 0 # default groups: 1
        self.num_ops += self.in_channels*self.out_channels*filter_area*output_area  // self.groups
        #print(self.in_channels, self.out_channels, self.kernel_size, ':', self.in_channels*self.out_channels*filter_area*output_area)
        return output

class FConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(FConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, groups, bias, dilation)
        self.num_ops = 0

    def forward(self, x):
        output = super(FConvTranspose2d, self).forward(x)
        output_area = output.size(-1)*output.size(-2)
        filter_area = np.prod(self.kernel_size)
        self.num_ops += self.in_channels*self.out_channels*filter_area*output_area
        return output



class FLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(FLinear, self).__init__(in_features, out_features, bias)
        self.num_ops = 0

    def forward(self, x):
        output = super(FLinear, self).forward(x)
        self.num_ops += self.in_features*self.out_features
        #print(self.in_features, self.out_features, ':',self.in_features*self.out_features)
        return output

def count_flops(model, reset=True):
    op_count = 0
    for name, m in model.named_modules():
        if hasattr(m, 'num_ops'):
            op_count += m.num_ops
            if reset: # count and reset to 0
                m.num_ops = 0

    return op_count



def compute_flops(dataset_name, backend, cfg):
    # replace all nn.Conv and nn.Linear layers with layers that count flops

    old_conv = nn.Conv2d
    old_convtranspose = nn.ConvTranspose2d
    old_linear = nn.Linear


    nn.Conv2d = FConv2d
    nn.ConvTranspose2d = FConvTranspose2d
    nn.Linear = FLinear

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset_name == 'imagenet':
        x = torch.randn(1, 3, 224, 224).to(device) # dummy inputs
    elif dataset_name == 'cifar100':
        x = torch.randn(1, 3, 32, 32).to(device) # dummy inputs
    elif dataset_name == 'cifar10':
        x = torch.randn(1, 3, 32, 32).to(device) # dummy inputs
    else:
        raise ValueError

    model = Model(backend, device, cfg=cfg)

    model = model.eval()
    params = sum([w.numel() for name, w in model.named_parameters()])
    with torch.no_grad():
        y_predicted = model.forward(x)
    flops = count_flops(model)
    #print('Params: %d | FLOPs: %d' % (params, flops))
    del model

    nn.Conv2d = old_conv
    nn.ConvTranspose2d = old_convtranspose
    nn.Linear = old_linear

    return params, flops


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, choices=['cifar10', 'cifar100', 'imagenet'], required=True)
    parser.add_argument('--bottlenecks', type=str, required=True)
    parser.add_argument('--backend', type=str, required=True)
    args = parser.parse_args()

    backend = args.backend
    dataset_name = args.dataset_name
    bottlenecks = list(map(int, args.bottlenecks.split(',')))

    #print(nn.Conv2d(3,3,3).num_ops)
    flops_list = compute_flops(backend, dataset_name)
    print(flops_list)
