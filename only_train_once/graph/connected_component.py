import torch
import numpy as np
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from assets.group_types import GROUP_TYPE

class ConnectedComponent:
    def __init__(self):
        self.nodes = dict()
        self.params = set()
        self.type = 1
        self.epsilon = 0.0
        self.upper_group_sparsity = 0.0
        self.num_zero_groups = 0
        self.zero_groups_idxes = list()
        self.non_zero_group_idxes = list()
        self.onnx_params = set()
        self.num_groups = 0
        self.auxiliary_ccs = list()
        
    def __repr__(self):
        return f"Id: {self.id}; Group type: {self.type}; Node ids: {self.nodes.values()}; \
        Param: {self.params}; OnnxParam: {self.onnx_params}; Num groups: {self.num_groups}, Num zero groups: {self.num_zero_groups}"

    @property
    def id(self):
        return "_".join([node.id for node in self.nodes.values()])

    def add_node(self, node):
        self.nodes[node.id] = node
    
    def add_nodes(self, nodes):
        for node in nodes:
            if node.id not in self.nodes:
                self.nodes[node.id] = node
    
    def merge(self, other_cc):
        for node_id in other_cc.nodes:
            if node_id not in self.nodes:
                self.nodes[node_id] = other_cc.nodes[node_id]

    def set_properties(self, output_nodes, opt=None):
        self.type = GROUP_TYPE["default"]
        for node in self.nodes.values():
            if node.is_concat(axis=1):
                self.type = GROUP_TYPE["auxilary"]
            if node.op.name == "conv":
                self.type = GROUP_TYPE["conv"]
            elif node.op.name == "linear" or node.op.name == "gemm":
                self.type = GROUP_TYPE["linear"]
            elif node.op.name == "multi-head-linear":
                self.type = GROUP_TYPE["multi-head-linear"]
            self.params |= set(node.params)
        self.type = GROUP_TYPE["default"] if (self.contain_output_nodes(output_nodes) \
                    or self.contain_non_zero_invariant_nodes()) else self.type
        self.epsilon = opt["optimizer"]["epsilon"][self.type] if opt is not None else 0.0
        self.upper_group_sparsity = opt["optimizer"]["upper_group_sparsity"][self.type] if opt is not None else 1.0

    def get_zero_groups(self):
        xs = []
        for param in self.params:
            if len(param.data.shape) == 1:
                xs.append(param.data.unsqueeze(1))
            elif len(param.data.shape) == 4: # conv layer
                xs.append(param.data.view(param.data.shape[0], -1))
            else:
                xs.append(param.data)        
        flatten_x = torch.cat(xs, dim = 1)

        norm_groups = torch.norm(flatten_x, dim=1)
        zero_groups_idxes = norm_groups == 0
        nonzero_groups_idxes = norm_groups != 0
        self.num_zero_groups = int(torch.sum(zero_groups_idxes).item())
        self.zero_groups_idxes = np.arange(0, self.num_groups)[zero_groups_idxes]
        self.non_zero_group_idxes = np.arange(0, self.num_groups)[nonzero_groups_idxes]

    def contain_concat_node(self, axis=1):  
        for node in self.nodes.values():
            if node.is_concat(axis=axis):
                return True
        return False
    
    def is_auxilary(self):
        if self.contain_concat_node(axis=1):
            self.type = GROUP_TYPE["auxilary"]
            return True
        else:
            return False
    
    def get_concat_node(self):
        # One auxiliary connected component has at most one concat joint node 
        # based on the algorithm
        if not self.is_auxilary():
            return None
        else:
            for node in self.nodes.values():
                if node.is_concat(axis=1):
                    return node
            return None
    
    def contain_output_nodes(self, output_nodes):
        for node in self.nodes.values():
            if node.id in output_nodes:
                return True
        return False

    def contain_non_zero_invariant_nodes(self):
        for node in self.nodes.values():
            if not node.is_zero_invariant():
                return True
        return False