import math
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import copy

__all__ = ['ResNet', 'resnet50']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, name=None, btn_config=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        if btn_config:
            if isinstance(btn_config[0], list):
                channels = btn_config[0]
                transit_channel = btn_config[1]
            else:
                channels = btn_config

            self.conv1 = conv1x1(inplanes, channels[0])
            self.bn1 = norm_layer(channels[0])
            self.conv2 = conv3x3(channels[0], channels[1], stride, groups)
            self.bn2 = norm_layer(channels[1])
            self.conv3 = conv1x1(channels[1], channels[2])
            self.bn3 = norm_layer(channels[2])
        else:
            self.conv1 = conv1x1(inplanes, width)
            self.bn1 = norm_layer(width)
            self.conv2 = conv3x3(width, width, stride, groups)
            self.bn2 = norm_layer(width)
            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
            self.cfg = [width, width]
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x) # 1x1, stride=1
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # 3x3
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) # 1x1, stride=1
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, cfg=None):
        super(ResNet, self).__init__()
        self.num_block = layers
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.bottlenecks = []
        self.cfg = cfg

        if self.cfg is None:
            self.inplanes = 64
            self.groups = groups
            self.base_width = width_per_group
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2) 
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            assert len(self.num_block) + 1 == len(cfg)
            for i in range(len(self.num_block)):
                assert self.num_block[i] == len(cfg[i+1])
            self.inplanes = cfg[0]
            self.groups = groups
            self.base_width = width_per_group
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, cfg[1][-1][-1], layers[0], layer_id=0)
            self.layer2 = self._make_layer(block, cfg[2][-1][-1], layers[1], stride=2, layer_id=1) 
            self.layer3 = self._make_layer(block, cfg[3][-1][-1], layers[2], stride=2, layer_id=2)
            self.layer4 = self._make_layer(block, cfg[4][-1][-1], layers[3], stride=2, layer_id=3)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(cfg[4][-1][-1], num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, layer_id=-1):
        norm_layer = self._norm_layer
        downsample = None


        if self.cfg:
            transit_channel = self.cfg[layer_id + 1][0][1]
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, transit_channel, stride),
                    norm_layer(transit_channel),
                )

        else:
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
                )

        layers = []
        if self.cfg:
            btn = block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, norm_layer, btn_config=self.cfg[layer_id + 1][0])
        else:
            btn = block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, norm_layer)
        self.bottlenecks.append(btn)
        layers.append(btn)

        if self.cfg:
            self.inplanes = planes
        else:
            self.inplanes = planes * block.expansion
        for block_id in range(1, blocks):
            if self.cfg:
                btn = block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, 
                                    norm_layer=norm_layer, btn_config=self.cfg[layer_id + 1][block_id])
            else:
                btn = block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, 
                                    norm_layer=norm_layer)

            self.bottlenecks.append(btn)
            layers.append(btn)

        return nn.Sequential(*layers)

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
        x = torch.flatten(x, 1)
        x = self.fc(x) 

        return x


    def compress_cfg(self, density_threshold=0.0):
        cfg = []
        for btn in self.bottlenecks:
            cfg.append(btn.compress_cfg(density_threshold))

        return cfg


    def get_config(self):
        cfg = [-1, [[[], -1],], [[[], -1],], [[[], -1],], [[[],-1],]]
        for idx, (name, param) in enumerate(self.named_parameters()):
            if len(param.shape) == 1: # ignore bn
                continue
            channel = param.shape[0]
            if idx == 0:
                cfg[0] = channel
                continue

            for i in range(len(self.num_block)):
                if idx < (3 + sum(self.num_block[:(i+1)]) * 9 + 3 * (i+1)):
                    #if 'shortcut' in name:
                    if 'downsample' in name:
                        cfg[i+1][0][1] = channel
                    elif cfg[i+1][0][1] == -1: # haven't met shortcut yet
                        cfg[i+1][0][0].append(channel)
                    #elif '0.weight' in name:
                    elif 'conv1.weight' in name:
                        cfg[i+1].append([channel])
                    else:
                        cfg[i+1][-1].append(channel)
                    break
        return cfg
    def create_zig_params(self, opt):
        optimizer_grouped_parameters = list()

        type_4_groups_name_regs = [
            'conv1.weight',
            'bn1.weight',
            'bn1.bias',
            'conv2.weight',
            'bn2.weight',
            'bn2.bias',
        ]
        type_1_groups = dict()
        type_4_groups = dict()
        type_4_groups2 = dict()

        type_4_groups_pointer = 0

        for i, (name, param) in enumerate(self.named_parameters()):
            is_type_4_group = False
            if i > 2:
                for reg in type_4_groups_name_regs:
                    if reg in name:
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
                        if 'bias' in name:
                            type_4_groups_pointer += 1
                        is_type_4_group = True
                        break
            if not is_type_4_group:
                if len(type_1_groups) == 0:
                    type_1_groups['params'] = list()
                    type_1_groups['shapes'] = list()
                    type_1_groups['names'] = list()
                    type_1_groups['group_type'] = 1
                type_1_groups['params'].append(param)
                type_1_groups['shapes'].append(param.shape)
                type_1_groups['names'].append(name)         

        optimizer_grouped_parameters.append(type_1_groups)
        optimizer_grouped_parameters += list(type_4_groups.values())                      
        self.optimizer_grouped_parameters = optimizer_grouped_parameters
        return optimizer_grouped_parameters

    def get_config_location(self, idx, cfg):
        assert idx % 3 == 0 and idx <= 156
        idx = idx // 3
        location = []
        pointer = 0
        for i1, c1 in enumerate(cfg):
            if isinstance(c1, list):
                for i2, c2 in enumerate(c1):
                    if isinstance(c2, list):
                        for i3, c3 in enumerate(c2):
                            if isinstance(c3, list):
                                for i4, c4 in enumerate(c3):
                                    if isinstance(c4, list):
                                        raise ValueError
                                    else:
                                        if idx == pointer:
                                            return [i1, i2, i3, i4]
                                        pointer += 1
                            else:
                                if idx == pointer:
                                    return [i1, i2, i3]
                                pointer += 1
                    else:
                        if idx == pointer:
                            return [i1, i2]
                        pointer += 1
            else:
                if idx == pointer:
                    return [i1]
                pointer += 1
        raise ValueError

    def get_group(self, idx, group_indices=[[9, 12, 21, 30], [39, 42, 51, 60, 69], [78, 81, 90, 99, 108, 117, 126], [135, 138, 147, 156]]): # resnet50_cifar
        for g_id, group_idx in enumerate(group_indices):
            if idx in group_idx:
                group = group_idx
                return group, g_id
        return None, None
    
    def is_conv_weights(self, shape):
        return len(shape) == 4

    def is_linear_weights(self, shape, num_classes):
        if len(shape) == 2 and shape[0] != num_classes:
            return True
        else:
            return False

    def prune(self):
        cfg = self.get_config()
        prune_cfg = copy.deepcopy(cfg)
        non_zero_idxes = [None for i in range(156 // 3 + 1)]

        total_zero_kernels = 0
        total_num_kernels = 0

        params = [param for param in self.parameters()]
        names = [name for name, param in self.named_parameters()]

        visited_set = set()
        for i, (name, param) in enumerate(zip(names, params)):
            if len(param.shape) != 4:
                continue

            if i in visited_set:
                continue
            group, group_id = self.get_group(i)

            if group is not None:
                l_x = []
                num_kernels, _0, _1, _2 = params[group[0]].shape
                for idx in group:
                    l_x.append(params[idx].view(num_kernels, -1))
                    l_x.append(params[idx+1].unsqueeze(1))
                    l_x.append(params[idx+2].unsqueeze(1))
                    visited_set.add(idx)
                    visited_set.add(idx+1)
                    visited_set.add(idx+2)
                flatten_param = torch.cat(l_x, dim = 1)

                num_zero_kernels = 0
                non_zero_idx = torch.norm(flatten_param, dim=1).bool()
                for k in range(flatten_param.shape[0]):
                    if torch.sum(flatten_param[k] == 0) == flatten_param[k].numel(): # all zeros
                        num_zero_kernels += 1
                        non_zero_idx[k] = False
                    else:
                        non_zero_idx[k] = True

                total_num_kernels += num_kernels
                total_zero_kernels += num_zero_kernels
                prune_channel = num_kernels - num_zero_kernels

                prune_cfg[group_id + 1][0][0][-1] = prune_channel
                for c_id in range(len(prune_cfg[group_id + 1])):
                    prune_cfg[group_id + 1][c_id][-1] = prune_channel
                    
                for idx in group:
                    non_zero_idxes[idx // 3] = non_zero_idx

            else:
                num_kernels, channel, height, width = param.shape
                bn_weight_param = params[i+1]
                bn_bias_param = params[i+2]
                visited_set.add(i)
                visited_set.add(i+1)
                visited_set.add(i+2)
                flatten_conv_param = param.view(num_kernels, -1)
                flatten_bn_weight_param = bn_weight_param.unsqueeze(1)
                flatten_bn_bias_param = bn_bias_param.unsqueeze(1)
                flatten_param = torch.cat((flatten_conv_param, flatten_bn_weight_param, flatten_bn_bias_param), dim = 1)

                num_zero_kernels = 0
                non_zero_idx = torch.norm(flatten_param, dim=1).bool()
                for k in range(flatten_param.shape[0]):
                    if torch.sum(flatten_param[k] == 0) == flatten_param[k].numel(): # all zeros
                        num_zero_kernels += 1
                        non_zero_idx[k] = False
                    else:
                        non_zero_idx[k] = True

                total_zero_kernels += num_zero_kernels
                total_num_kernels += num_kernels

                location = self.get_config_location(i, cfg)
                if len(location) == 1:
                    prune_cfg[location[0]] = num_kernels - num_zero_kernels
                elif len(location) == 3:
                    prune_cfg[location[0]][location[1]][location[2]] = num_kernels - num_zero_kernels
                elif len(location) == 4:
                    prune_cfg[location[0]][location[1]][location[2]][location[3]] = num_kernels - num_zero_kernels
                else:
                    raise ValueError
                non_zero_idxes[i // 3] = non_zero_idx

        prune_model = resnet50(cfg=prune_cfg)

        gs_param_names = [p_name for p_name in self.state_dict()]
        gs_param_names_trainable = [p_name for p_name, _  in self.named_parameters()]
        prune_param_names = [p_name for p_name in prune_model.state_dict()]
        prune_param_names_trainable = [p_name for p_name, _ in prune_model.named_parameters()]

        conv_idx = 0
        visited = set()    
        for i, (gs_p_name, prune_p_name) in enumerate(zip(gs_param_names, prune_param_names)):
            # print(i, gs_p_name, prune_p_name)
            gs_param = self.state_dict()[gs_p_name]
            prune_param = prune_model.state_dict()[prune_p_name]
            if i in visited:
                continue

            if self.is_conv_weights(gs_param.shape) and self.is_conv_weights(prune_param.shape):
                idx = gs_param_names_trainable.index(gs_p_name)
                group, group_id = self.get_group(idx)

                if prune_param.shape[0] == 1 or prune_param.shape[1] == 1:
                    continue
                
                gs_bn_weight_param = self.state_dict()[gs_param_names[i+1]]
                gs_bn_bias_param = self.state_dict()[gs_param_names[i+2]]
                gs_bn_mean_param = self.state_dict()[gs_param_names[i+3]]
                gs_bn_var_param = self.state_dict()[gs_param_names[i+4]]
                prune_bn_weight_param = prune_model.state_dict()[prune_param_names[i+1]]
                prune_bn_bias_param = prune_model.state_dict()[prune_param_names[i+2]]
                prune_bn_mean_param = prune_model.state_dict()[prune_param_names[i+3]]
                prune_bn_var_param = prune_model.state_dict()[prune_param_names[i+4]]

                if conv_idx == 0:
                    prune_param.data.copy_(gs_param.data[non_zero_idxes[conv_idx], ...])
                    prune_bn_weight_param.data.copy_(gs_bn_weight_param.data[non_zero_idxes[conv_idx], ...])
                    prune_bn_bias_param.data.copy_(gs_bn_bias_param.data[non_zero_idxes[conv_idx], ...])
                    prune_bn_mean_param.data.copy_(gs_bn_mean_param.data[non_zero_idxes[conv_idx], ...])
                    prune_bn_var_param.data.copy_(gs_bn_var_param.data[non_zero_idxes[conv_idx], ...])
                elif conv_idx < len(non_zero_idxes):
                    if conv_idx in [4]: # [3, 4, 7, 10]
                        prune_param.data.copy_(gs_param.data[non_zero_idxes[conv_idx], ...][:, non_zero_idxes[0], ...])
                    elif conv_idx in [14]: # [13, 14, 17, 20, 23]
                        prune_param.data.copy_(gs_param.data[non_zero_idxes[conv_idx], ...][:, non_zero_idxes[3], ...])
                    elif conv_idx in [27]: # [26, 27, 30, 33, 36, 39, 42]
                        prune_param.data.copy_(gs_param.data[non_zero_idxes[conv_idx], ...][:, non_zero_idxes[13], ...])
                    elif conv_idx in [46]: # [45, 46, 49, 52]
                        prune_param.data.copy_(gs_param.data[non_zero_idxes[conv_idx], ...][:, non_zero_idxes[26], ...])
                    else:
                        prune_param.data.copy_(gs_param.data[non_zero_idxes[conv_idx], ...][:, non_zero_idxes[conv_idx - 1], ...])
                    prune_bn_weight_param.data.copy_(gs_bn_weight_param.data[non_zero_idxes[conv_idx], ...])
                    prune_bn_bias_param.data.copy_(gs_bn_bias_param.data[non_zero_idxes[conv_idx], ...])
                    prune_bn_mean_param.data.copy_(gs_bn_mean_param.data[non_zero_idxes[conv_idx], ...])
                    prune_bn_var_param.data.copy_(gs_bn_var_param.data[non_zero_idxes[conv_idx], ...])

                conv_idx += 1
                visited.add(i+1)
                visited.add(i+2)
                visited.add(i+3)
                visited.add(i+4)
            # The last linear needs to be assigned
            elif self.is_linear_weights(gs_param.shape, num_classes=-1) and self.is_linear_weights(prune_param.shape, num_classes=-1):
                gs_bias_param = self.state_dict()[gs_param_names[i+1]]
                prune_bias_param = prune_model.state_dict()[prune_param_names[i+1]]
                prune_param.data.copy_(gs_param.data[:, non_zero_idxes[-1], ...])
                prune_bias_param.data.copy_(gs_bias_param.data)

                conv_idx += 1
                visited.add(i+1)
            else:
                if gs_param.data.shape != prune_param.data.shape: # dummy residual function
                    conv_idx += 1
                else:
                    prune_param.data.copy_(gs_param.data)
            visited.add(i)
        return prune_model, prune_model.get_config()

def _resnet(arch, block, layers, pretrained, last_model_path='', **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch])
        print(f"load model parameters from: {model_urls[arch]}")
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, last_model_path='', **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, last_model_path, 
                   **kwargs)

