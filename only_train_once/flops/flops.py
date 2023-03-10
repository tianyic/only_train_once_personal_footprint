import numpy as np
import torch
import torch.nn as nn
import copy

MILLION = 1e6

def compute_linear_flops(node, compressed=False):
    if not node.input_shape or not node.output_shape:
        return 0
    output_shape = node.output_shape
    input_shape = node.input_shape[0]
    if compressed:
        pruned_shape = node.pruned_shape
        input_shape[1] = pruned_shape[1] if pruned_shape[1] >= 0 else input_shape[1]
        output_shape[1] = pruned_shape[0] if pruned_shape[0] >= 0 else output_shape[1]
    input_features = input_shape[1]
    output_features = output_shape[1]
    num_ops = input_features * output_features / float(MILLION)
    return num_ops

def compute_conv_flops(node, compressed=False):
    if not node.input_shape or not node.output_shape:
        return 0
    output_shape = copy.deepcopy(node.output_shape) # B * C * H * W
    input_shape = copy.deepcopy(node.input_shape[0]) # B * C * H * W
    if compressed:
        pruned_shape = node.pruned_shape
        input_shape[1] = pruned_shape[1] if pruned_shape[1] >= 0 else input_shape[1]
        output_shape[1] = pruned_shape[0] if pruned_shape[0] >= 0 else output_shape[1]
    output_area = output_shape[-1] * output_shape[-2]
    filter_area = np.prod(node.op_params["kernel_shape"])
    output_channel = output_shape[1]
    input_channel = input_shape[1]
    group = node.op_params["group"]
    num_ops = float(input_channel*output_channel*filter_area*output_area  // group) / float(MILLION)
    return num_ops

def compute_flops(graph, compressed=False):
    flops_break_down = dict()
    flops_break_down['total'] = 0
    for cc in graph.connected_components.values():
        flops_break_down[cc.id] = dict()
        flops_break_down[cc.id]['flops'] = 0
        flops_break_down[cc.id]['num_groups'] = cc.num_groups
        for node in cc.nodes.values():
            if node.is_conv():
                cur_flops = compute_conv_flops(node, compressed)
                flops_break_down[cc.id]['flops'] += cur_flops
                flops_break_down['total'] += cur_flops
            if node.is_linear():
                cur_flops = compute_linear_flops(node, compressed)
                flops_break_down[cc.id]['flops'] += cur_flops
                flops_break_down['total'] += cur_flops
                
    for cc in graph.connected_components.values():
        flops_break_down[cc.id]['flops'] /= flops_break_down['total']
    return flops_break_down