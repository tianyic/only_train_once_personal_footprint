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

    def create_zig_params(self, opt):
        optimizer_grouped_parameters = list()

        type_1_groups = dict()
        type_3_groups = dict()
        type_4_groups = dict()


        type_3_groups_pointer = 0
        type_4_groups_pointer = 0

        for i, (name, param) in enumerate(self.named_parameters()):
            if 'features' in name:
                type_4_groups_pointer = int(i//2)

                if type_4_groups_pointer not in type_4_groups:
                    type_4_groups[type_4_groups_pointer] = dict()
                    type_4_groups[type_4_groups_pointer]['params'] = list()
                    type_4_groups[type_4_groups_pointer]['shapes'] = list()
                    type_4_groups[type_4_groups_pointer]['names'] = list()
                    type_4_groups[type_4_groups_pointer]['epsilon'] = opt['epsilon'][4]
                    type_4_groups[type_4_groups_pointer]['upper_group_sparsity'] = opt['upper_group_sparsity'][4]
                    type_4_groups[type_4_groups_pointer]['group_type'] = 4
                type_4_groups[type_4_groups_pointer]['params'].append(param)
                type_4_groups[type_4_groups_pointer]['shapes'].append(param.shape)
                type_4_groups[type_4_groups_pointer]['names'].append(name)
            elif ('classifier.0' in name) or ('classifier.3' in name):
                if type_3_groups_pointer not in type_3_groups:
                    type_3_groups[type_3_groups_pointer] = dict()
                    type_3_groups[type_3_groups_pointer]['params'] = list()
                    type_3_groups[type_3_groups_pointer]['shapes'] = list()
                    type_3_groups[type_3_groups_pointer]['names'] = list()
                    type_3_groups[type_3_groups_pointer]['epsilon'] = opt['epsilon'][3]
                    type_3_groups[type_3_groups_pointer]['upper_group_sparsity'] = opt['upper_group_sparsity'][3]
                    type_3_groups[type_3_groups_pointer]['group_type'] = 3
                type_3_groups[type_3_groups_pointer]['params'].append(param)
                type_3_groups[type_3_groups_pointer]['shapes'].append(param.shape)
                type_3_groups[type_3_groups_pointer]['names'].append(name)
                if 'bias' in name:
                    type_3_groups_pointer += 1

            elif 'classifier.6' in name: # last classifier does not need to be updated
                if len(type_1_groups) == 0:
                    type_1_groups['params'] = list()
                    type_1_groups['shapes'] = list()
                    type_1_groups['names'] = list()
                    type_1_groups['epsilon'] = opt['epsilon'][1]
                    type_1_groups['upper_group_sparsity'] = opt['upper_group_sparsity'][1]
                    type_1_groups['group_type'] = 1
                type_1_groups['params'].append(param)
                type_1_groups['shapes'].append(param.shape)
                type_1_groups['names'].append(name)
            else:
                raise ValueError

        optimizer_grouped_parameters.append(type_1_groups)
        optimizer_grouped_parameters += list(type_3_groups.values())
        optimizer_grouped_parameters += list(type_4_groups.values())
        
        self.optimizer_grouped_parameters = optimizer_grouped_parameters
        return optimizer_grouped_parameters

    def is_conv_weights(self, shape):
        return len(shape) == 4

    def is_linear_weights(self, shape, num_classes):
        if len(shape) == 2 and shape[0] != num_classes:
            return True
        else:
            return False



    def prune(self):
        channels = self.channels
        prune_channels = [3]
        non_zero_idxes = list()

        total_zero_kernels = 0
        total_num_kernels = 0

        biased = True
        bn = self.bn
        params = [param for param in self.parameters()]

        for i, param in enumerate(params):
            if self.is_conv_weights(param.shape):
                if biased is False:
                    num_kernels, channel, height, width = param.shape
                    total_num_kernels += num_kernels

                    flatten_param = param.view(num_kernels, -1)
                    norm_kernels = torch.norm(flatten_param, dim=1)

                elif biased is True and bn is False:
                    num_kernels, channel, height, width = param.shape
                    bias_param = params[i+1]
                    flatten_conv_param = param.view(num_kernels, -1)
                    flatten_bias_param = bias_param.unsqueeze(1)
                    flatten_param = torch.cat((flatten_conv_param, flatten_bias_param), dim = 1)
                    norm_kernels = torch.norm(flatten_param, dim=1)
                elif biased is True and bn is True:
                    num_kernels, channel, height, width = param.shape
                    bias_param = params[i+1]
                    bn_weight_param = params[i+2]
                    bn_bias_param = params[i+3]
                    flatten_conv_param = param.view(num_kernels, -1)
                    flatten_bias_param = bias_param.unsqueeze(1)
                    flatten_bn_weight_param = bn_weight_param.unsqueeze(1)
                    flatten_bn_bias_param = bn_bias_param.unsqueeze(1)
                    flatten_param = torch.cat((flatten_conv_param, flatten_bias_param, flatten_bn_weight_param, flatten_bn_bias_param), dim = 1)
                    norm_kernels = torch.norm(flatten_param, dim=1)
                num_zero_kernels = torch.sum(norm_kernels == 0).detach().tolist()
                total_zero_kernels += num_zero_kernels
                
                total_num_kernels += num_kernels
                non_zero_idx = norm_kernels != 0
                prune_channels.append(num_kernels - num_zero_kernels)
                non_zero_idxes.append(non_zero_idx)
            elif self.is_linear_weights(param.shape, num_classes=self.num_classes):
                if biased is True:
                    out_channels, in_channels = param.shape
                    bias_param = params[i+1]
                    flatten_linear_param = param
                    flatten_bias_param = bias_param.unsqueeze(1)
                    flatten_param = torch.cat((flatten_linear_param, flatten_bias_param), dim = 1)
                    norm_kernels = torch.norm(flatten_param, dim=1)
                else:
                    raise ValueError
                num_zero_kernels = torch.sum(norm_kernels == 0).detach().tolist()
                total_zero_kernels += num_zero_kernels
                
                total_num_kernels += num_kernels
                non_zero_idx = norm_kernels != 0
                prune_channels.append(num_kernels - num_zero_kernels)
                non_zero_idxes.append(non_zero_idx)

        prune_model = VGG16(channels=prune_channels, bn=self.bn)


        gs_param_names = [p_name for p_name in self.state_dict()]
        prune_param_names = [p_name for p_name in prune_model.state_dict()]

        conv_idx = 0
        visited = set()    
        for i, (gs_p_name, prune_p_name) in enumerate(zip(gs_param_names, prune_param_names)):
            gs_param = self.state_dict()[gs_p_name]
            prune_param = prune_model.state_dict()[prune_p_name]
            if i in visited:
                continue
            if self.is_conv_weights(gs_param.shape) and self.is_conv_weights(prune_param.shape):
                if biased is False:
                    if conv_idx == 0:
                        prune_param.data.copy_(gs_param.data[non_zero_idxes[conv_idx], ...])
                    elif conv_idx < len(non_zero_idxes):
                        prune_param.data.copy_(gs_param.data[non_zero_idxes[conv_idx], ...][:, non_zero_idxes[conv_idx - 1], ...])
                    else:
                        prune_param.data.copy_(gs_param.data[:, non_zero_idxes[conv_idx - 1], ...])
                    conv_idx += 1
                elif biased is True and bn is False:
                    gs_bias_param = self.state_dict()[gs_param_names[i+1]]
                    prune_bias_param = prune_model.state_dict()[prune_param_names[i+1]]
                    if conv_idx == 0:
                        prune_param.data.copy_(gs_param.data[non_zero_idxes[conv_idx], ...])
                        prune_bias_param.data.copy_(gs_bias_param.data[non_zero_idxes[conv_idx], ...])
                    elif conv_idx < len(non_zero_idxes):
                        prune_param.data.copy_(gs_param.data[non_zero_idxes[conv_idx], ...][:, non_zero_idxes[conv_idx - 1], ...])
                        prune_bias_param.data.copy_(gs_bias_param.data[non_zero_idxes[conv_idx], ...])
                    else:
                        prune_param.data.copy_(gs_param.data[:, non_zero_idxes[conv_idx - 1], ...])
                        prune_bias_param.data.copy_(gs_bias_param.data)
                    conv_idx += 1
                    visited.add(i+1)
                elif biased is True and bn is True:
                    gs_bias_param = self.state_dict()[gs_param_names[i+1]]
                    gs_bn_weight_param = self.state_dict()[gs_param_names[i+2]]
                    gs_bn_bias_param = self.state_dict()[gs_param_names[i+3]]
                    gs_bn_mean_param = self.state_dict()[gs_param_names[i+4]]
                    gs_bn_var_param = self.state_dict()[gs_param_names[i+5]]
                    prune_bias_param = prune_model.state_dict()[prune_param_names[i+1]]
                    prune_bn_weight_param = prune_model.state_dict()[prune_param_names[i+2]]
                    prune_bn_bias_param = prune_model.state_dict()[prune_param_names[i+3]]
                    prune_bn_mean_param = prune_model.state_dict()[prune_param_names[i+4]]
                    prune_bn_var_param = prune_model.state_dict()[prune_param_names[i+5]]
                    if conv_idx == 0:
                        prune_param.data.copy_(gs_param.data[non_zero_idxes[conv_idx], ...])
                        prune_bias_param.data.copy_(gs_bias_param.data[non_zero_idxes[conv_idx], ...])
                        prune_bn_weight_param.data.copy_(gs_bn_weight_param.data[non_zero_idxes[conv_idx], ...])
                        prune_bn_bias_param.data.copy_(gs_bn_bias_param.data[non_zero_idxes[conv_idx], ...])
                        prune_bn_mean_param.data.copy_(gs_bn_mean_param.data[non_zero_idxes[conv_idx], ...])
                        prune_bn_var_param.data.copy_(gs_bn_var_param.data[non_zero_idxes[conv_idx], ...])
                    elif conv_idx < len(non_zero_idxes):
                        prune_param.data.copy_(gs_param.data[non_zero_idxes[conv_idx], ...][:, non_zero_idxes[conv_idx - 1], ...])
                        prune_bias_param.data.copy_(gs_bias_param.data[non_zero_idxes[conv_idx], ...])
                        prune_bn_weight_param.data.copy_(gs_bn_weight_param.data[non_zero_idxes[conv_idx], ...])
                        prune_bn_bias_param.data.copy_(gs_bn_bias_param.data[non_zero_idxes[conv_idx], ...])
                        prune_bn_mean_param.data.copy_(gs_bn_mean_param.data[non_zero_idxes[conv_idx], ...])
                        prune_bn_var_param.data.copy_(gs_bn_var_param.data[non_zero_idxes[conv_idx], ...])
                    else:
                        prune_param.data.copy_(gs_param.data[:, non_zero_idxes[conv_idx - 1], ...])
                        prune_bias_param.data.copy_(gs_bias_param.data)
                        prune_bn_weight_param.data.copy_(gs_bn_weight_param.data)
                        prune_bn_bias_param.data.copy_(gs_bn_bias_param.data)
                        prune_bn_mean_param.data.copy_(gs_bn_mean_param.data)
                        prune_bn_var_param.data.copy_(gs_bn_var_param.data)

                    conv_idx += 1
                    visited.add(i+1)
                    visited.add(i+2)
                    visited.add(i+3)
                    visited.add(i+4)
                    visited.add(i+5)
            # The last linear needs to be assigned
            elif self.is_linear_weights(gs_param.shape, num_classes=-1) and self.is_linear_weights(prune_param.shape, num_classes=-1):
                if biased is True:
                    gs_bias_param = self.state_dict()[gs_param_names[i+1]]
                    prune_bias_param = prune_model.state_dict()[prune_param_names[i+1]]
                    if conv_idx == 0:
                        prune_param.data.copy_(gs_param.data[non_zero_idxes[conv_idx], ...])
                        prune_bias_param.data.copy_(gs_bias_param.data[non_zero_idxes[conv_idx], ...])
                    elif conv_idx < len(non_zero_idxes):
                        prune_param.data.copy_(gs_param.data[non_zero_idxes[conv_idx], ...][:, non_zero_idxes[conv_idx - 1], ...])
                        prune_bias_param.data.copy_(gs_bias_param.data[non_zero_idxes[conv_idx], ...])
                    else:
                        prune_param.data.copy_(gs_param.data[:, non_zero_idxes[conv_idx - 1], ...])
                        prune_bias_param.data.copy_(gs_bias_param.data)

                    conv_idx += 1
                    visited.add(i+1)
                else:
                    raise ValueError


            else:
                prune_param.data.copy_(gs_param.data)
            
            visited.add(i)
        return prune_model, prune_channels


def vgg16(cfg=[3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]):
    return VGG16(cfg)


def vgg16_bn(cfg=[3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]):
    return VGG16(cfg, bn=True)

