from abc import ABC, abstractclassmethod
import torch
from only_train_once.transform import tensor_transformation, TensorTransform
import numpy as np

class BasicNodeGroup(ABC):
    def __init__(self, is_prunable=True):
        self.nodes = dict()
        self.output_nodes = dict()
        self.is_prunable = is_prunable
        self.pruning_redundant_idxes = list()
        self.pruning_important_idxes = list()
        self.is_auxiliary = False
        self.extra_param_group_attrs = dict()

    def __repr__(self):
        return f"Id: {self.id}, is_prunable: {self.is_prunable}, nodes: {self.nodes}"
    
    def num_nodes(self):
        return len(self.nodes)
    
    @property
    def id(self):
        return "_".join([node.id for node in self.nodes.values()])
        
    def add_node(self, node):
        if node.id not in self.nodes:
            self.nodes[node.id] = node

    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)

    def contain_some_nodes(self, nodes):
        for node in nodes:
            if self.contain_node(node):
                return True
        return False
        
    def contain_node(self, node):
        return True if node.id in self.nodes else False
    
    def remove_node(self, node):
        if node.id in self.nodes:
            del self.nodes[node.id]
    
    @abstractclassmethod
    def get_param_groups(self):
        raise NotImplementedError

    @property
    def param_names(self):
        return self.get_param_names()
    
    def get_param_names(self):    
        param_names = list()
        for node in self:
            if len(node.param_names) == 0:
                continue
            param_names.extend(node.param_names)
        return param_names

    def __iter__(self):
        self._iter_idx = 0
        self._node_ids = list(self.nodes.keys())
        return self
    
    def __next__(self):
        if self._iter_idx < self.num_nodes():
            node = self.nodes[self._node_ids[self._iter_idx]]
            self._iter_idx += 1
            return node
        raise StopIteration
    
    def set_output_nodes(self, graph):
        for node in self.nodes.values():
            is_out_node = True
            for node_out in graph.outgoing(node):
                if node_out.id in self.nodes:
                    is_out_node = False
            if is_out_node:
                self.output_nodes[node.id] = node 

    def get_node_ids(self, skip_output_node=False):
        return set(self.nodes.keys()) if not skip_output_node \
            else set(self.nodes.keys()).difference(self.output_nodes.keys())

    def merge(self, node_group):
        self.add_nodes(node_group.nodes.values())
        
class NodeGroup(BasicNodeGroup):
    def __init__(self, is_prunable=True):
        super().__init__(is_prunable)

    @property
    def num_groups(self):
        num_groups = 1
        for node in self:
            if len(node.param_names) == 0:
                continue
            if not node.op:
                continue
            num_groups = max(num_groups, node.op.num_groups)
        return num_groups

    def get_modules(self):
        modules = set()
        for node in self:
            if not node.op:
                continue
            if hasattr(node.op, 'module'):
                modules.add(node.op.module)
        return modules
    
    def get_param_groups(self):
        ng_param_group = dict()
        ng_param_group['id'] = self.id
        ng_param_group['num_groups'] = self.num_groups
        ng_param_group['is_prunable'] = self.is_prunable
        ng_param_group['is_auxiliary'] = self.is_auxiliary
        ng_param_group['p_names'] = list()
        ng_param_group['params'] = list()
        ng_param_group['op_names'] = list()
        ng_param_group['p_transform'] = list()
        ng_param_group['auxiliary_ngs'] = list()
        
        basic_attrs = ['op_names', 'p_names', 'params', 'p_transform']
        for node in self:
            if len(node.param_names) == 0 or not node.op:
                continue
            node_param_groups = node.op.get_param_groups(param_names=node.param_names)
            ng_param_group['op_names'].extend([node_param_groups['op']] * len(node_param_groups['params']))
            ng_param_group['p_names'].extend(node_param_groups['p_names'])
            ng_param_group['params'].extend(node_param_groups['params'])
            ng_param_group['p_transform'].extend(node_param_groups['p_transform'])
            for attr in node_param_groups:
                if attr not in basic_attrs:
                    ng_param_group[attr] = node_param_groups[attr]
        
        for attr in self.extra_param_group_attrs:
            if attr not in ng_param_group:
                ng_param_group[attr] = self.extra_param_group_attrs[attr]
        return ng_param_group

    def set_pruning_redundant_idxes(self):
        param_groups = self.get_param_groups()
        if len(param_groups['params']) == 0 and not self.is_auxiliary:
            self.pruning_important_idxes, self.pruning_redundant_idxes = list(), list()
            return 
        elif len(param_groups['params']) > 0 and not self.is_auxiliary:
            norm_group = None
            for (param, p_transform) in zip(param_groups['params'], param_groups['p_transform']):
                if p_transform == TensorTransform.NO_PRUNE:
                    continue
                
                param_transform = None
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    param_transform = tensor_transformation(param, p_transform, param_groups['num_groups'], param_groups['num_heads'])
                else:
                    param_transform = tensor_transformation(param, p_transform, param_groups['num_groups'])

                if norm_group == None:
                    norm_group = torch.norm(param_transform, dim=1) ** 2
                else:
                    norm_group += torch.norm(param_transform, dim=1) ** 2
            if norm_group is None:
                self.pruning_important_idxes, self.pruning_redundant_idxes = list(), list()
                return
            norm_group = torch.sqrt(norm_group)
            norm_group = norm_group.cpu()

            self.pruning_important_idxes = np.arange(self.num_groups)[norm_group != 0]
            self.pruning_redundant_idxes = np.arange(self.num_groups)[norm_group == 0]
    
            if hasattr(self, 'overwrite_p_transform'):
                if self.overwrite_p_transform == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD and 'head_dim' in param_groups:
                    head_dim = param_groups['head_dim']
                    refined_pruning_important_idxes = list()
                    for i in self.pruning_important_idxes:
                        refined_pruning_important_idxes.extend([h + i * head_dim for h in range(head_dim)])
                    self.pruning_important_idxes = np.array(refined_pruning_important_idxes)
                    refined_pruning_redundant_idxes = list()
                    for i in self.pruning_redundant_idxes:
                        refined_pruning_redundant_idxes.extend([h + i * head_dim for h in range(head_dim)])
                    self.pruning_redundant_idxes = np.array(refined_pruning_redundant_idxes)
        elif self.is_auxiliary:
            pruning_redundant_idxes = list()
            offset = 0
            for dependent_node_group in self.dependent_node_groups:
                
                if len(dependent_node_group.pruning_redundant_idxes) == 0:
                    offset += dependent_node_group.num_groups
                    continue
                pruning_redundant_idxes.append(dependent_node_group.pruning_redundant_idxes + offset)
                offset += dependent_node_group.num_groups

            if len(pruning_redundant_idxes) > 0:
                self.pruning_redundant_idxes = np.concatenate(pruning_redundant_idxes)
            else:
                self.pruning_redundant_idxes = list()
            self.pruning_important_idxes = list()

    def prune_out_dim(self, global_skip_modules=set()):
        local_skip_modules=set()    
        for node in self.nodes.values():
            if not node.op:
                continue
            if hasattr(node.op, 'prune_out_dim'):
                if node.op.module not in local_skip_modules and node.op.module not in global_skip_modules:
                    node.op.prune_out_dim(pruned_idxes=self.pruning_redundant_idxes, param_names=node.param_names)
                    local_skip_modules.add(node.op.module)
                elif node.op.module is None and type(node.op).__name__ == 'ParamOTO':
                    # ParamOperator does not have module
                    node.op.prune_out_dim(pruned_idxes=self.pruning_redundant_idxes, param_names=node.param_names)
                    local_skip_modules.add(node.op.param)
                elif self.contain_lora():
                    node.op.prune_out_dim(pruned_idxes=self.pruning_redundant_idxes, param_names=node.param_names)
                    
    def contain_lora(self):
        for node in self:
            if len(node.param_names) == 0 or not node.op:
                continue
            for param_name in node.param_names:
                if 'lora_B' in param_name or 'lora_embedding_B' in param_name:
                    self.scaling = node.op.lora_scaling
                    return True
        return False
    
    def contain_stem_op(self):
        is_stem = False
        for node in self:
            if not node.op:
                continue
            if node.op.is_stem:
                is_stem = True
        return is_stem

    def contain_concat(self, axis=1):
        for node in self:
            if node.is_concat(axis=axis):
                return True
        return False

    def get_concat_nodes(self, axis=1):
        concat_nodes = list()
        for node in self:
            if node.is_concat(axis=axis):
                concat_nodes.append(node)
        return concat_nodes
        
    def set_auxiliary(self):
        if self.contain_concat(axis=1):
            self.is_auxiliary = True
            return True
        else:
            self.is_auxiliary = False
            return False
        
class NodeGroupComposedOp(BasicNodeGroup):
    """
    NodeGroupComposedOp refers to the node group for a composed operator
    """
    def __init__(self, is_prunable=True, op=None):
        super().__init__(is_prunable)
        self.op = op

    def get_modules(self):
        modules = set()
        if not self.op:
            return modules
        elif hasattr(self.op, 'module'):
            modules.add(self.op.module)
        return modules
                        
    def set_node_equivalence(self):
        self.node_cluster_by_leaf_module = dict()
        self.node_id_to_leaf_module_id = dict()
        for node in self:
            if len(node.param_names) == 0:
                continue
            for leaf_module in self.op.leaf_modules.values():
                if set(node.param_names).issubset(set(leaf_module.param_names)):
                    if leaf_module.id not in self.node_cluster_by_leaf_module:
                        self.node_cluster_by_leaf_module[leaf_module.id] = list()
                    self.node_cluster_by_leaf_module[leaf_module.id].append(node)
                    self.node_id_to_leaf_module_id[node.id] = leaf_module.id

    def set_auxiliary(self):
        # TODO: to implemented.
        self.is_auxiliary = False

    def set_output_nodes(self, graph):
        for node in self.nodes.values():
            is_out_node = True
            for node_out in graph.outgoing(node):
                if node_out.id in self.nodes:
                    is_out_node = False 
            if is_out_node:
                self.output_nodes[node.id] = node 
        
        new_node_outs = set()
        for node_out in self.output_nodes.values():
            if node_out.id not in self.node_id_to_leaf_module_id:
                continue
            leaf_module_id = self.node_id_to_leaf_module_id[node_out.id]
            node_cluster = self.node_cluster_by_leaf_module[leaf_module_id]
            for node in node_cluster:
                if node.id not in self.output_nodes:
                    new_node_outs.add(node)

        # needs to include the incoming stems for the direct output_node
        visited = dict.fromkeys(self.nodes, False)
        def dfs_helper(node, graph, path):
            if node.is_stem():
                for node_new in path:
                    if node_new.id not in self.output_nodes:
                        new_node_outs.add(node_new)
                return 
            for node_in in graph.incoming(node):
                if node_in.id in self.nodes and not visited[node_in.id]:
                    visited[node_in.id] = True
                    path.append(node_in)
                    dfs_helper(node_in, graph, path)

        for out_node in self.output_nodes.values():
            dfs_helper(out_node, graph, [])

        for node in new_node_outs:
            self.output_nodes[node.id] = node 

        self.out_param_names = list()
        for out_node in self.output_nodes.values():
            self.out_param_names.extend(out_node.param_names)
        # Set op out_param_names for composed op
        self.op.out_param_names = self.out_param_names

    def get_param_groups(self):
        ng_param_group = dict()
        ng_param_group['id'] = self.id
        ng_param_group['num_groups'] = self.num_groups
        ng_param_group['is_prunable'] = self.is_prunable
        ng_param_group['is_auxiliary'] = self.is_auxiliary
        ng_param_group['p_names'] = list()
        ng_param_group['params'] = list()
        ng_param_group['op_names'] = list()
        ng_param_group['p_transform'] = list()
        ng_param_group['auxiliary_ngs'] = list()
        
        # Skip the output node params, which should depend on other node groups
        op_param_group = self.op.get_param_groups(skip_output_node=True)
        basic_attrs = ['op_names', 'p_names', 'params', 'p_transform']
        ng_param_group['op_names'].extend([op_param_group['op']] * len(op_param_group['params']))
        ng_param_group['p_names'].extend(op_param_group['p_names'])
        ng_param_group['params'].extend(op_param_group['params'])
        ng_param_group['p_transform'].extend(op_param_group['p_transform'])
        
        for attr in op_param_group:
            if attr not in basic_attrs:
                ng_param_group[attr] = op_param_group[attr]

        for attr in self.extra_param_group_attrs:
            if attr not in ng_param_group:
                ng_param_group[attr] = self.extra_param_group_attrs[attr]
        return ng_param_group
    
    @property
    def num_groups(self):
        return self.op.num_groups

    def set_pruning_redundant_idxes(self):
        param_groups = self.get_param_groups()
        if len(param_groups['params']) == 0 and not self.is_auxiliary:
            self.pruning_important_idxes, self.pruning_redundant_idxes = list(), list()
            return 
        elif len(param_groups['params']) > 0 and not self.is_auxiliary:
            norm_group = None
            for (p_name, param, p_transform) in zip(param_groups['p_names'], param_groups['params'], param_groups['p_transform']):
                # Skip lora_A or lora_embedding_A if any
                if 'lora_A' in p_name or 'lora_embedding_A' in p_name:
                    continue
                param_transform = None
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    param_transform = tensor_transformation(param, p_transform, param_groups['num_groups'], param_groups['num_heads'])
                else:
                    param_transform = tensor_transformation(param, p_transform, param_groups['num_groups'])

                if norm_group == None:
                    norm_group = torch.norm(param_transform, dim=1) ** 2
                else:
                    norm_group += torch.norm(param_transform, dim=1) ** 2
            norm_group = torch.sqrt(norm_group)
            norm_group = norm_group.cpu()

            self.pruning_important_idxes = np.arange(self.num_groups)[norm_group != 0]
            self.pruning_redundant_idxes = np.arange(self.num_groups)[norm_group == 0]
                                        
    def prune_out_dim(self, **kwargs):
        if hasattr(self.op, 'prune_out_dim'):
            self.op.prune_out_dim(pruned_idxes=self.pruning_redundant_idxes, skip_output_node=True)
            for node in self:
                if node.id in self.output_nodes:
                    continue
                node.pruned_status['out_dim'] = True

    def contain_lora(self):
        for node in self:
            if len(node.param_names) == 0 or not node.op:
                continue
            for param_name in node.param_names:
                if 'lora_B' in param_name:
                    self.scaling = node.op.lora_scaling
                    return True
        return False

    def contain_stem_op(self):
        return self.op.is_stem
    
    def set_auxilary(self):
        pass