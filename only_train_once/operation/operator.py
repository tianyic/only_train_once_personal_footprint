import torch
import torch.nn as nn
from only_train_once.transform import TensorTransform
from abc import ABC, abstractclassmethod

class BasicOperator(ABC):
    def __init__(self, id=None, _type=None, cfg_params=dict()):
        self.id = id
        self._type = _type
        self.cfg_params = cfg_params
        # Stem operator can transform the primary dim of input tensor
        self.is_stem = False
        self.pruned_status = {
            'out_dim': False,
            'in_dim': False
        }
    
    @abstractclassmethod
    def get_param_groups(self):
        raise NotImplementedError
    
    def prune_param_and_grad(self, param, preserved_idxes, dim=0):
        pruned_param = torch.nn.Parameter(torch.index_select(param, dim, torch.LongTensor(preserved_idxes).to(param.device)))
        if param.grad is not None:
            pruned_param.grad = torch.index_select(param.grad, dim, torch.LongTensor(preserved_idxes).to(param.device))
        return pruned_param.to(param.device)

class Operator(BasicOperator):
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().__init__(id, _type, cfg_params)
        self.module = module
        self.leaf_modules = dict()

        self.set_leaf_modules()
        self.set_param_names()
        self.name_to_param = dict()
        for name, param in self.named_parameters():
            self.name_to_param[self.id+'.'+name] = param
        self.num_groups = 1
        # Is basic module or not
        self.is_basic = True
        self.p_transform = TensorTransform.BASIC
        
    def __eq__(self, name):
        return self.name == name

    def __repr__(self) -> str:
        return self._full_info()

    def _get_module_type(self, module):
        return type(module).__name__
    
    def _full_info(self):
        return "Id: {id}, Type: {type}, Leaf Modules: {leaf_module_keys}, Param Names: {param_names}".format(
            id=self.id, type=self._type, leaf_module_keys=" ".join(list(self.leaf_modules.keys())), param_names=" ".join(self.param_names) 
        )

    def set_leaf_modules(self):
        if not self.module:
            return 
        def dfs_helper(module, module_name, composed_op):
            module_type = self._get_module_type(module)
            if module_type in COMPOSED_MODULES:
                composed_op = COMPOSED_MODULES[module_type](
                    id = module_name,
                    _type = module_type,
                    module = module)
                self.leaf_modules[composed_op.id] = composed_op
                return 
            
            if next(module.named_children(), None) is None:
                if module_type in BASIC_MODULES:
                    self.leaf_modules[module_name] = BASIC_MODULES[module_type](
                        id = module_name, 
                        _type = module_type,
                        module = module)
                return 

            for name, module_child in module.named_children():
                dfs_helper(module_child, module_name + '.' + name if module_name != '' else name, composed_op)

        if next(self.module.named_children(), None) is None:
            self.leaf_modules[self.id] = self
        else:        
            for name, module_child in self.module.named_children():
                dfs_helper(module_child, self.id + '.' + name if self.id != '' else name, None)

    def set_param_names(self):
        self.param_names = list()
        if not self.module:
            return 
        for name, _ in self.module.named_parameters():
            self.param_names.append(self.id + '.' + name)

    def named_parameters(self):
        return self.module.named_parameters() if self.module else []

    def get_param_groups(self, param_names=list(), **kwargs):
        param_groups = dict()
        param_groups['op'] = self._type
        param_groups['p_names'] = list()
        param_groups['params'] = list()
        param_groups['p_transform'] = list()
        for p_name in param_names:
            param = self.name_to_param[p_name]
            if not param.requires_grad:
                continue
            param_groups['p_names'].append(p_name)
            param_groups['params'].append(param)
            param_groups['p_transform'].append(self.p_transform)
        return param_groups
            
    def set_num_groups(self):
        self.num_groups = 1
        for param_name in self.name_to_param:
            param = self.name_to_param[param_name]
            self.num_groups = max(self.num_groups, param.shape[0])

    def compute_flops(self, input_shape):
        return 0

    @property
    def num_params(self):
        num_params = 0
        for param_name in self.name_to_param:
            param = self.name_to_param[param_name]
            num_params += param.numel()
        return num_params

class ParamOTO(Operator):
    '''
    Operator for the tensor parameters in torch yet not formed in nn.Module 
    '''
    def __init__(self, id=None, _type=None, cfg_params=dict(), param_name=None, param=None):
        super().__init__(id, _type, cfg_params)
        self.is_stem = False
        self.param_name = param_name
        self.param = param
        
    def get_param_groups(self, **kwargs):
        param_groups = dict()
        param_groups['op'] = self._type
        param_groups['p_names'] = [self.param_name]
        param_groups['params'] = [self.param]
        param_groups['p_transform'] = [self.p_transform]
        return param_groups

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        preserved_idxes = list(set(range(self.param.shape[0])) - set(pruned_idxes))
        preserved_idxes.sort()
        self.param = self.prune_param_and_grad(self.param, preserved_idxes, 0)
        
class Conv2dOTO(Operator):
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().__init__(id, _type, cfg_params, module)
        self.is_stem = True
        self.set_num_groups()
    
    def get_param_groups(self, param_names=list()):
        param_groups = dict()
        param_groups['op'] = 'conv2d'
        param_groups['p_names'] = list()
        param_groups['params'] = list()
        param_groups['p_transform'] = list()
        for p_name in param_names:
            param_groups['p_names'].append(p_name)
            param_groups['params'].append(self.name_to_param[p_name])
            param_groups['p_transform'].append(self.p_transform)
        return param_groups

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        # TODO: for generic group conv support, we currently only support a special perhaps the most common
        # case, that is groups = out_channels
        if self.module.groups == self.module.out_channels:
            self.module.groups = self.module.out_channels - len(pruned_idxes)
        preserved_idxes = list(set(range(self.module.out_channels)) - set(pruned_idxes))
        preserved_idxes.sort()
        self.module.out_channels = self.module.out_channels - len(pruned_idxes)
        self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)
        if self.module.bias is not None:
            self.module.bias = self.prune_param_and_grad(self.module.bias, preserved_idxes, 0)

    def prune_in_dim(self, pruned_idxes=list(), **kwargs):
        preserved_idxes = list(set(range(self.module.in_channels)) - set(pruned_idxes))
        preserved_idxes.sort()
        # TODO: for generic group conv support, 
        # We currently consider a special case, see zig.py for more details
        if self.module.groups == self.module.out_channels and self.module.groups > 1:
            return 
        if self.module.weight.shape[1] <= len(preserved_idxes):
            return
        self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 1)
        self.module.in_channels = self.module.in_channels - len(pruned_idxes)

    def compute_flops(self, input_tensor_shape):
        # Only consider multiplication
        batch_size, _, height_in, width_in = input_tensor_shape
        stride_h, stride_w = self.cfg_params['strides']
        kernel_h, kernel_w = self.cfg_params['kernel_shape']
        if 'pads' in self.cfg_params:
            height_in = height_in + self.cfg_params['pads'][2] * 2
            width_in = width_in + self.cfg_params['pads'][3] * 2

        sliding_times_h = (height_in - kernel_h + stride_h) // (stride_h)
        sliding_times_w = (width_in - kernel_w + stride_w) // (stride_w)

        flops = batch_size * kernel_h * kernel_w * sliding_times_h * sliding_times_w \
                * self.module.in_channels * self.module.out_channels
        if 'group' in self.cfg_params:
            flops /= self.cfg_params['group']
        return flops
    
class ConvTranspose2dOTO(Operator):
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().__init__(id, _type, cfg_params, module)
        self.is_stem = True
        self.set_num_groups()
        self.p_transform = TensorTransform.TRANSPOSE
        
    def set_num_groups(self):
        self.num_groups = 1
        for param_name in self.name_to_param:
            param = self.name_to_param[param_name]
            if param_name.endswith('.weight'):
                self.num_groups = max(self.num_groups, param.shape[1])
            elif param_name.endswith('.bias'):
                self.num_groups = max(self.num_groups, param.shape[0])
            
    def get_param_groups(self, param_names=[]):
        param_groups = dict()
        param_groups['op'] = 'convtranspose2d'
        param_groups['p_names'] = list()
        param_groups['params'] = list()
        param_groups['p_transform'] = list()
        for p_name in param_names:
            param_groups['p_names'].append(p_name)
            param_groups['params'].append(self.name_to_param[p_name])
            param_groups['p_transform'].append(self.p_transform)
        return param_groups

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        preserved_idxes = list(set(range(self.module.out_channels)) - set(pruned_idxes))
        preserved_idxes.sort()
        self.module.out_channels = self.module.out_channels - len(pruned_idxes)

        if not self.module.transposed:
            self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)
        else:
            self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 1)
        
        if self.module.bias is not None:
            self.module.bias = self.prune_param_and_grad(self.module.bias, preserved_idxes, 0)
        
    def prune_in_dim(self, pruned_idxes=list(), **kwargs):
        preserved_idxes = list(set(range(self.module.in_channels)) - set(pruned_idxes))
        preserved_idxes.sort()
        self.module.in_channels = self.module.in_channels - len(pruned_idxes)        
        if not self.module.transposed:
            self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 1)
        else:
            self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)
            
class BatchNormOTO(Operator):
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().__init__(id, _type, cfg_params, module)
        self.is_stem = False
        self.set_num_groups()
        self.p_transform = TensorTransform.ACCESSORY
        
    def get_param_groups(self, param_names=[]):
        param_groups = dict()
        param_groups['op'] = 'batchnorm'
        param_groups['p_names'] = list()
        param_groups['params'] = list()
        param_groups['p_transform'] = list()
        for p_name in param_names:
            if p_name in self.name_to_param:
                param_groups['p_names'].append(p_name)
                param_groups['params'].append(self.name_to_param[p_name])
                param_groups['p_transform'].append(self.p_transform)
        return param_groups

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        preserved_idxes = list(set(range(self.module.num_features)) - set(pruned_idxes))
        preserved_idxes.sort()
        self.module.num_features = self.module.num_features - len(pruned_idxes)
        self.module.running_mean = self.module.running_mean.data[preserved_idxes]
        self.module.running_var = self.module.running_var.data[preserved_idxes]
        if self.module.affine:
            self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)
            self.module.bias = self.prune_param_and_grad(self.module.bias, preserved_idxes, 0)

class InstanceNormOTO(Operator):
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().__init__(id, _type, cfg_params, module)
        self.is_stem = False
        self.set_num_groups()
        self.p_transform = TensorTransform.ACCESSORY

    def get_param_groups(self, param_names=[]):
        param_groups = dict()
        param_groups['op'] = 'instantnorm'
        param_groups['p_names'] = list()
        param_groups['params'] = list()
        param_groups['p_transform'] = list()
        for p_name in param_names:
            param_groups['p_names'].append(p_name)
            param_groups['params'].append(self.name_to_param[p_name])
            param_groups['p_transform'].append(self.p_transform)
        return param_groups

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        preserved_idxes = list(set(range(self.module.num_features)) - set(pruned_idxes))
        preserved_idxes.sort()
        self.module.num_features = self.module.num_features - len(pruned_idxes)
        if self.module.affine:
            self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)
            self.module.bias = self.prune_param_and_grad(self.module.bias, preserved_idxes, 0)

class GroupNormOTO(Operator):
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().__init__(id, _type, cfg_params, module)
        self.is_stem = False
        self.set_num_groups()
        self.num_heads = module.num_groups
        self.head_dim = self.num_groups // self.num_heads
        self.num_groups = module.num_groups
        self.p_transform = TensorTransform.MULTIHEAD_NUMHEAD_SPREAD
        
    def get_param_groups(self, param_names=list()):
        param_groups = dict()
        param_groups['op'] = 'groupnorm'
        param_groups['p_names'] = list()
        param_groups['params'] = list()
        param_groups['p_transform'] = list()
        param_groups['num_groups'] = self.num_groups
        param_groups['num_heads'] = self.num_heads
        param_groups['head_dim'] = self.head_dim
        for p_name in param_names:
            param_groups['p_names'].append(p_name)
            param_groups['params'].append(self.name_to_param[p_name])
            param_groups['p_transform'].append(self.p_transform)
        return param_groups

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        preserved_idxes = list(set(range(self.module.num_channels)) - set(pruned_idxes))
        preserved_idxes.sort()
        self.module.num_channels = self.module.num_channels - len(pruned_idxes)
        self.module.num_groups = self.module.num_groups - len(pruned_idxes) // self.head_dim
        if self.module.affine:
            self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)
            self.module.bias = self.prune_param_and_grad(self.module.bias, preserved_idxes, 0)

class LinearOTO(Operator):
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().__init__(id, _type, cfg_params, module)
        self.is_stem = True
        self.set_num_groups()
    
    def get_param_groups(self, param_names=list()):
        param_groups = dict()
        param_groups['op'] = 'linear'
        param_groups['p_names'] = list()
        param_groups['params'] = list()
        param_groups['p_transform'] = list()
        target_param_names = param_names if len(param_names) > 0 else self.name_to_param.keys()
        for p_name in target_param_names:
            param_groups['p_names'].append(p_name)
            param_groups['params'].append(self.name_to_param[p_name])
            param_groups['p_transform'].append(self.p_transform)
        return param_groups

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        preserved_idxes = list(set(range(self.module.out_features)) - set(pruned_idxes))
        preserved_idxes.sort()
        self.module.out_features = self.module.out_features - len(pruned_idxes)
        self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)
        if self.module.bias is not None:
            self.module.bias = self.prune_param_and_grad(self.module.bias, preserved_idxes, 0)

    def prune_in_dim(self, pruned_idxes=list(), **kwargs):
        preserved_idxes = list(set(range(self.module.in_features)) - set(pruned_idxes))
        preserved_idxes.sort()
        self.module.in_features = self.module.in_features - len(pruned_idxes)
        self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 1)

    def compute_flops(self, input_tensor_shape):
        # Only consider multiplication
        # The input_tensor_shape for linear is [*, in_features]
        flops = 1
        for dim in input_tensor_shape:
            flops *= dim
        flops *= self.module.out_features
        return flops

class LoraLinearOTO(Operator):
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().__init__(id, _type, cfg_params, module)
        self.set_num_groups()
        self.ori_in_features = self.module.in_features
        self.ori_out_features = self.module.out_features
        self.lora_scaling = module.scaling
        self.is_stem = True
        self.is_basic = False

    def get_param_groups(self, param_names=list(), skip_output_node=False, **kwargs):
        param_groups = dict()
        param_groups['op'] = 'lora_linear'
        param_groups['p_names'] = list()
        param_groups['params'] = list()
        param_groups['p_transform'] = list()
        if hasattr(self, 'lora_scaling'):
            param_groups['lora_scaling'] = self.lora_scaling
        target_param_names = param_names if len(param_names) > 0 else self.name_to_param.keys()
        for p_name in target_param_names:
            param = self.name_to_param[p_name]
            if not skip_output_node or (skip_output_node and 'lora_A' in p_name):
                param_groups['p_names'].append(p_name)
                param_groups['params'].append(param)
                if 'lora_A' in p_name:          
                    param_groups['p_transform'].append(TensorTransform.NO_PRUNE)
                else:
                    param_groups['p_transform'].append(TensorTransform.BASIC)
        return param_groups
        
        
    def prune_out_dim(self, pruned_idxes=list(), param_names=list(), skip_output_node=False, **kwargs):  
        # If param_names are provided, pruned by names.
        target_param_names = param_names if len(param_names) > 0 else self.name_to_param
        for param_name in target_param_names:
            preserved_idxes = list(set(range(self.num_groups)) - set(pruned_idxes))
            preserved_idxes.sort()        
            if 'lora_A' not in param_name and 'lora_B' not in param_name:
                if param_name.endswith('.weight'):
                    self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)
                    self.name_to_param[param_name] = self.module.weight
                elif param_name.endswith('.bias'):
                    self.module.bias = self.prune_param_and_grad(self.module.bias, preserved_idxes, 0)
                    self.name_to_param[param_name] = self.module.bias
                self.module.out_features = len(preserved_idxes)
            elif 'lora_B' in param_name:
                for module in self.module.lora_B.values():
                    module.weight = self.prune_param_and_grad(module.weight, preserved_idxes, 0)
                    self.name_to_param[param_name] = module.weight
                    module.out_features = len(preserved_idxes)
                self.module.out_features = len(preserved_idxes)

    def prune_in_dim(self, pruned_idxes=list(), param_names=list(), verbose=False, **kwargs):
        for param_name in param_names:
            preserved_idxes = list(set(range(self.ori_in_features)) - set(pruned_idxes))
            preserved_idxes.sort()
            # weight
            if 'lora_A' not in param_name and 'lora_B' not in param_name:
                if param_name.endswith('.weight'):
                    self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 1)
                    self.name_to_param[param_name] = self.module.weight
                    self.module.in_features = len(preserved_idxes)
            # lora_B
            elif 'lora_B' in param_name:
                pass
            elif 'lora_A' in param_name:
                for module in self.module.lora_A.values():
                    module.weight = self.prune_param_and_grad(module.weight, preserved_idxes, 1)
                    self.name_to_param[param_name] = module.weight
                    module.in_features = len(preserved_idxes)
                self.module.in_features = len(preserved_idxes)

class LoraEmbeddingOTO(Operator):
    '''
    model.model.embed_tokens.weight torch.Size([32000, 4096])
    model.model.embed_tokens.lora_embedding_A.default torch.Size([8, 32000])
    model.model.embed_tokens.lora_embedding_B.default torch.Size([4096, 8])
    '''
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().__init__(id, _type, cfg_params, module)
        self.num_groups = self.module.embedding_dim
        self.is_stem = True
        self.is_basic = False
        self.lora_scaling = module.scaling

    def get_param_groups(self, param_names=list(), skip_output_node=False, **kwargs):
        param_groups = dict()
        param_groups['op'] = 'lora_embedding'
        param_groups['p_names'] = list()
        param_groups['params'] = list()
        param_groups['p_transform'] = list()
        if hasattr(self, 'lora_scaling'):
            param_groups['lora_scaling'] = self.lora_scaling
        target_param_names = param_names if len(param_names) > 0 else self.name_to_param.keys()
        for p_name in target_param_names:
            param = self.name_to_param[p_name]
            if not skip_output_node or (skip_output_node and 'lora_embedding_A' in p_name):
                param_groups['p_names'].append(p_name)
                param_groups['params'].append(param)
                if 'lora_embedding_A' in p_name:          
                    param_groups['p_transform'].append(TensorTransform.NO_PRUNE)
                elif 'lora_embedding_B' in p_name:
                    param_groups['p_transform'].append(TensorTransform.BASIC)
                else:
                    param_groups['p_transform'].append(TensorTransform.TRANSPOSE)
        return param_groups
    
    def prune_out_dim(self, pruned_idxes=list(), param_names=list(), skip_output_node=False, **kwargs):  
        preserved_idxes = list(set(range(self.num_groups)) - set(pruned_idxes))
        preserved_idxes.sort()
        # If param_names are provided, pruned by names.
        target_param_names = param_names if len(param_names) > 0 else self.name_to_param
        for param_name in target_param_names:   
            if 'lora_embedding_A' not in param_name and 'lora_embedding_B' not in param_name:
                if param_name.endswith('.weight'):
                    self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 1)
                    self.name_to_param[param_name] = self.module.weight
                self.module.embedding_dim = len(preserved_idxes)
            elif 'lora_embedding_B' in param_name:
                for module_name in self.module.lora_embedding_B:
                    module_param = self.module.lora_embedding_B[module_name]
                    self.module.lora_embedding_B[module_name] = self.prune_param_and_grad(module_param, preserved_idxes, 0)
                    self.name_to_param[param_name] = module_param
                self.module.embedding_dim = len(preserved_idxes)

class EmbeddingOTO(Operator):
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().__init__(id, _type, cfg_params, module)
        self.num_groups = self.module.embedding_dim
        self.is_transpose = True
        self.is_stem = False
        self.is_basic = True
        self.p_transform = TensorTransform.TRANSPOSE
        
    def get_param_groups(self, **kwargs):
        param_groups = dict()
        param_groups['op'] = 'embedding'
        param_groups['p_names'] = list()
        param_groups['params'] = list()
        param_groups['p_transform'] = list()
        for p_name in self.name_to_param:
            param_groups['p_names'].append(p_name)
            param_groups['params'].append(self.name_to_param[p_name])
            param_groups['p_transform'].append(self.p_transform)
        return param_groups

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        preserved_idxes = list(set(range(self.module.embedding_dim)) - set(pruned_idxes))
        preserved_idxes.sort()
        self.module.embedding_dim = self.module.embedding_dim - len(pruned_idxes)
        self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 1)
    
class LayerNormOTO(Operator):
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().__init__(id, _type, cfg_params, module)
        self.set_num_groups()
        self.is_stem = False
        self.is_basic = False # is basic module
            
    def get_param_groups(self, **kwargs):
        param_groups = dict()
        param_groups['op'] = 'layernorm'
        param_groups['p_names'] = list()
        param_groups['params'] = list()
        param_groups['p_transform'] = list()
        for p_name in self.name_to_param:
            param_groups['p_names'].append(p_name)
            param_groups['params'].append(self.name_to_param[p_name])
            param_groups['p_transform'].append(self.p_transform)
        return param_groups

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        preserved_idxes = list(set(range(self.num_groups)) - set(pruned_idxes))
        preserved_idxes.sort()
        self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)
        if hasattr(self.module, 'bias'):
            self.module.bias = self.prune_param_and_grad(self.module.bias, preserved_idxes, 0)
        if hasattr(self.module, 'normalized_shape'):
            self.module.normalized_shape = tuple((len(preserved_idxes),))

class BaseMultiHeadAttentionOTO(Operator):
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().__init__(id, _type, cfg_params, module)
        self.is_stem = True
        self.is_basic = False
        self.out_key = 'attn_w'
        self.op_name = 'multi_head_attention'
        self.prune_mode = 'num_head'

        assert self.prune_mode in ['head_dim', 'num_head'], "Prune_mode of mha must be either head_dim or num_head."
        
        # find the scaling parameters if loralinear
        for leaf_module in self.leaf_modules.values():
            if type(leaf_module).__name__ == 'LoraLinearOTO':
                if hasattr(leaf_module, 'lora_scaling'):
                    self.lora_scaling = leaf_module.lora_scaling
        self.set_attributes()
        
    def set_attributes(self):
        if hasattr(self.module, 'n_heads'):
            self.num_heads = self.module.n_heads
        elif hasattr(self.module, 'num_heads'):
            self.num_heads = self.module.num_heads
        self.head_dim = self.module.head_dim
        if self.prune_mode == 'head_dim':
            self.num_groups = self.head_dim
        elif self.prune_mode == 'num_head':
            self.num_groups = self.num_heads
        self.hidden_size = self.module.hidden_size
        self.num_group_divisible = 2
    
    def get_param_groups(self, param_names=list(), skip_output_node=False, **kwargs):
        param_groups = dict()
        param_groups['op'] = self.op_name
        param_groups['p_names'] = list()
        param_groups['params'] = list()
        param_groups['p_transform'] = list()
        param_groups['num_heads'] = self.num_heads
        param_groups['head_dim'] = self.head_dim
        if hasattr(self, 'lora_scaling'):
            param_groups['lora_scaling'] = self.lora_scaling
        target_param_names = param_names if len(param_names) > 0 else self.name_to_param.keys()
        for p_name in target_param_names:
            param = self.name_to_param[p_name]
            if self.out_key in p_name and not skip_output_node:
                param_groups['p_names'].append(p_name)
                param_groups['params'].append(param)
                if 'lora_A' in p_name:
                    param_groups['p_transform'].append(TensorTransform.NO_PRUNE)
                else:
                    param_groups['p_transform'].append(TensorTransform.BASIC)
            elif self.out_key not in p_name:
                param_groups['p_names'].append(p_name)
                param_groups['params'].append(param)
                if 'lora_A' in p_name:
                    param_groups['p_transform'].append(TensorTransform.NO_PRUNE)
                elif self.prune_mode == 'head_dim':
                    param_groups['p_transform'].append(TensorTransform.MULTIHEAD_HEADDIM)
                elif self.prune_mode == 'num_head':
                    param_groups['p_transform'].append(TensorTransform.MULTIHEAD_NUMHEAD)
        return param_groups
    
    def prune_out_dim_head_dim(self, pruned_idxes=list(), param_names=list(), skip_output_node=True, **kwargs):
        visited_modules = set()
        if len(param_names) > 0:
            for param_name in param_names:
                for module_name in self.leaf_modules:
                    if not param_name.startswith(module_name):
                        continue
                    leaf_op = self.leaf_modules[module_name]
                    if module_name not in visited_modules:
                        leaf_op.prune_out_dim(pruned_idxes, param_names=[param_name])
                    visited_modules.add(module_name)
        elif len(param_names) == 0 and skip_output_node:
            preserved_idxes = list(set(range(self.num_groups)) - set(pruned_idxes))
            preserved_idxes.sort()
            # Prune over k, q, v weights
            self.module.head_dim = self.module.head_dim - len(pruned_idxes)
            for module_name in self.leaf_modules:
                if self.out_key in module_name:
                    continue
                leaf_op = self.leaf_modules[module_name]
                expand_pruned_idxes = list()
                for h in range(self.num_heads):
                    expand_pruned_idxes.extend([i + h * self.head_dim for i in pruned_idxes])
                leaf_op.prune_out_dim(expand_pruned_idxes)

    def prune_out_dim_num_head(self, pruned_idxes=list(), param_names=list(), skip_output_node=True, **kwargs):
        visited_modules = set()
        if len(param_names) > 0:
            for param_name in param_names:
                for module_name in self.leaf_modules:
                    if not param_name.startswith(module_name):
                        continue
                    leaf_op = self.leaf_modules[module_name]
                    if module_name not in visited_modules:
                        leaf_op.prune_out_dim(pruned_idxes, param_names=[param_name])
                    visited_modules.add(module_name)
        elif len(param_names) == 0 and skip_output_node:
            preserved_idxes = list(set(range(self.num_groups)) - set(pruned_idxes))
            preserved_idxes.sort()
            # Prune over k, q, v weights
            if hasattr(self.module, 'n_heads'):
                self.module.n_heads = self.num_groups - len(pruned_idxes)
            elif hasattr(self.module, 'num_heads'):
                self.module.num_heads = self.num_groups - len(pruned_idxes)
                
            for module_name in self.leaf_modules:
                if self.out_key in module_name:
                    continue
                leaf_op = self.leaf_modules[module_name]
                expand_pruned_idxes = list()
                for i in pruned_idxes:
                    for h in range(self.head_dim):
                        expand_pruned_idxes.append(h + i * self.head_dim)
                leaf_op.prune_out_dim(expand_pruned_idxes)
                
    def prune_out_dim(self, pruned_idxes=list(), param_names=list(), skip_output_node=True, **kwargs):
        if self.prune_mode == 'head_dim':       
            self.prune_out_dim_head_dim(pruned_idxes, param_names, skip_output_node, **kwargs)
        elif self.prune_mode == 'num_head':
            self.prune_out_dim_num_head(pruned_idxes, param_names, skip_output_node, **kwargs)
        
    def prune_in_dim(self, pruned_idxes=list(), param_names=list(), **kwargs):
        visited_modules = set()
        for param_name in param_names:
            for module_name in self.leaf_modules:
                if not param_name.startswith(module_name):
                    continue
                leaf_op = self.leaf_modules[module_name]
                if not hasattr(leaf_op, 'prune_in_dim'):
                    continue
                if module_name not in visited_modules:
                    leaf_op.prune_in_dim(pruned_idxes, param_names=[param_name])
                visited_modules.add(module_name)

class LlamaAttentionOTO(BaseMultiHeadAttentionOTO):
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().__init__(id, _type, cfg_params, module)
        self.out_key = 'o_proj'
        self.op_name = 'llama_attention'

    def prune_out_dim_head_dim(self, pruned_idxes=list(), param_names=list(), skip_output_node=True, **kwargs):        
        if len(param_names) > 0:
            visited_modules = set()
            for param_name in param_names:
                for module_name in self.leaf_modules:
                    if not param_name.startswith(module_name):
                        continue
                    leaf_op = self.leaf_modules[module_name]
                    if module_name not in visited_modules:
                        leaf_op.prune_out_dim(pruned_idxes, param_names=[param_name])
                    visited_modules.add(module_name)
        elif len(param_names) == 0 and skip_output_node:
            preserved_idxes = list(set(range(self.module.head_dim)) - set(pruned_idxes))
            preserved_idxes.sort()
            # Prune over k, q, v weights
            self.module.head_dim = self.module.head_dim - len(pruned_idxes)          
            self.module.reset_rotary_emb() 
            
            for module_name in self.leaf_modules:
                if self.out_key in module_name:
                    continue
                leaf_op = self.leaf_modules[module_name]
                expand_pruned_idxes = list()
                for h in range(self.num_heads):
                    expand_pruned_idxes.extend([i + h * self.head_dim for i in pruned_idxes])
                leaf_op.prune_out_dim(expand_pruned_idxes)
                
class BertAttentionOTO(BaseMultiHeadAttentionOTO):
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().__init__(id, _type, cfg_params, module)
        self.out_key = 'output'
        self.op_name = 'bert_attention'
        
    def set_attributes(self):
        self.num_heads = self.module.self.num_attention_heads
        self.head_dim = self.module.self.attention_head_size
        if self.prune_mode == 'head_dim':
            self.num_groups = self.head_dim
        elif self.prune_mode == 'num_head':
            self.num_groups = self.num_heads
        self.hidden_size = self.num_heads * self.head_dim
        self.num_group_divisible = 2

    def prune_out_dim_head_dim(self, pruned_idxes=list(), param_names=list(), skip_output_node=True, **kwargs):        
        if len(param_names) > 0:
            visited_modules = set()
            for param_name in param_names:
                for module_name in self.leaf_modules:
                    if not param_name.startswith(module_name):
                        continue
                    leaf_op = self.leaf_modules[module_name]
                    if module_name not in visited_modules:
                        leaf_op.prune_out_dim(pruned_idxes, param_names=[param_name])
                    visited_modules.add(module_name)
        elif len(param_names) == 0 and skip_output_node:
            preserved_idxes = list(set(range(self.num_groups)) - set(pruned_idxes))
            preserved_idxes.sort()
            # Prune over k, q, v weights
            self.module.self.attention_head_size = self.head_dim - len(pruned_idxes)
            self.module.self.all_head_size = self.module.self.num_attention_heads * self.module.self.attention_head_size
            
            for module_name in self.leaf_modules:
                if self.out_key in module_name:
                    continue
                leaf_op = self.leaf_modules[module_name]
                expand_pruned_idxes = list()
                for h in range(self.num_heads):
                    expand_pruned_idxes.extend([i + h * self.head_dim for i in pruned_idxes])
                leaf_op.prune_out_dim(expand_pruned_idxes)

    def prune_out_dim_num_head(self, pruned_idxes=list(), param_names=list(), skip_output_node=True, **kwargs):
        visited_modules = set()
        if len(param_names) > 0:
            for param_name in param_names:
                for module_name in self.leaf_modules:
                    if not param_name.startswith(module_name):
                        continue
                    leaf_op = self.leaf_modules[module_name]
                    if module_name not in visited_modules:
                        leaf_op.prune_out_dim(pruned_idxes, param_names=[param_name])
                    visited_modules.add(module_name)
        elif len(param_names) == 0 and skip_output_node:
            preserved_idxes = list(set(range(self.num_groups)) - set(pruned_idxes))
            preserved_idxes.sort()
            # Prune over k, q, v weights
            self.module.self.num_attention_heads = self.num_groups - len(pruned_idxes)
            self.module.self.all_head_size = self.module.self.num_attention_heads * self.module.self.attention_head_size
            for module_name in self.leaf_modules:
                if self.out_key in module_name:
                    continue
                leaf_op = self.leaf_modules[module_name]
                expand_pruned_idxes = list()
                for i in pruned_idxes.tolist():
                    for h in range(self.head_dim):
                        expand_pruned_idxes.append(h + i * self.head_dim)
                leaf_op.prune_out_dim(expand_pruned_idxes)
                
class PhiAttentionOTO(BaseMultiHeadAttentionOTO):
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().__init__(id, _type, cfg_params, module)
        self.is_stem = True
        self.is_basic = False
        self.out_key = 'out_proj'
        self.op_name = 'phi_attention'
        self.set_attributes()

    def set_attributes(self):
        self.num_heads = self.module.n_head
        self.head_dim = self.module.head_dim
        if self.prune_mode == 'head_dim':
            self.num_groups = self.head_dim
        elif self.prune_mode == 'num_head':
            self.num_groups = self.num_heads

    def get_param_groups(self, param_names=list(), skip_output_node=False, **kwargs):
        param_groups = dict()
        param_groups['op'] = self.op_name
        param_groups['p_names'] = list()
        param_groups['params'] = list()
        param_groups['p_transform'] = list()
        param_groups['num_heads'] = self.num_heads
        param_groups['head_dim'] = self.head_dim
        if hasattr(self, 'lora_scaling'):
            param_groups['lora_scaling'] = self.lora_scaling
        target_param_names = param_names if len(param_names) > 0 else self.name_to_param.keys()
        for p_name in target_param_names:
            param = self.name_to_param[p_name]
            if self.out_key in p_name and not skip_output_node:
                param_groups['p_names'].append(p_name)
                param_groups['params'].append(param)
                if 'lora_A' in p_name:
                    param_groups['p_transform'].append(TensorTransform.NO_PRUNE)
                else:
                    param_groups['p_transform'].append(TensorTransform.BASIC)
            
        return param_groups

    def prune_out_dim_head_dim(self, pruned_idxes=list(), param_names=list(), skip_output_node=True, **kwargs):  
        visited_modules = set()
        if len(param_names) > 0:
            for param_name in param_names:
                for module_name in self.leaf_modules:
                    if not param_name.startswith(module_name):
                        continue
                    leaf_op = self.leaf_modules[module_name]
                    if module_name not in visited_modules:
                        leaf_op.prune_out_dim(pruned_idxes, param_names=[param_name])
                    visited_modules.add(module_name)
        elif len(param_names) == 0 and skip_output_node:
            preserved_idxes = list(set(range(self.module.head_dim)) - set(pruned_idxes))
            preserved_idxes.sort()
            # Prune over k, q, v weights
            self.module.head_dim = self.module.head_dim - len(pruned_idxes)
            for module_name in self.leaf_modules:
                if self.out_key in module_name:
                    continue
                leaf_op = self.leaf_modules[module_name]
                expand_pruned_idxes = list()
                for h in range(self.num_heads):
                    expand_pruned_idxes.extend([i + h * self.head_dim for i in pruned_idxes])
                leaf_op.prune_out_dim(expand_pruned_idxes)

class PReLUOTO(Operator):
    def init(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().init(id, _type, cfg_params, module)
        self.is_stem = False
        self.set_num_groups()
        self.p_transform = TensorTransform.ACCESSORY
        
    def get_param_groups(self, param_names=[]):
        param_groups = dict()
        param_groups['op'] = 'prelu'
        param_groups['p_names'] = list()
        param_groups['params'] = list()
        param_groups['p_transform'] = list()
        for p_name in param_names:
            if p_name in self.name_to_param:
                param_groups['p_names'].append(p_name)
                param_groups['params'].append(self.name_to_param[p_name])
                param_groups['p_transform'].append(self.p_transform)
        return param_groups

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        preserved_idxes = list(set(range(self.module.num_parameters)) - set(pruned_idxes))
        preserved_idxes.sort()
        self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)
        self.module.num_parameters = self.module.num_parameters - len(pruned_idxes)

BASIC_MODULES = {
    'ConvTranspose2d': ConvTranspose2dOTO,
    'Conv2d': Conv2dOTO,
    'ModulatedConv2d': Conv2dOTO, # For stagelightv2
    'EqualLinear': LinearOTO, # For stagelightv2
    'Linear': LinearOTO,
    'BatchNorm2d': BatchNormOTO,
    'InstanceNorm2d': InstanceNormOTO,
    'GroupNorm': GroupNormOTO,
    'Embedding': EmbeddingOTO,
    
    'LlamaRMSNorm': LayerNormOTO,
    'LayerNorm': LayerNormOTO,
    
    'PReLU': PReLUOTO,
}

# Composed modules must contain at least two nodes with trainable variables
COMPOSED_MODULES = {
    'LlamaAttention': LlamaAttentionOTO,
    'SelfAttention': BaseMultiHeadAttentionOTO,
    'BertAttention': BertAttentionOTO,
    'PhiMHA': PhiAttentionOTO,
    'LoraLinear': LoraLinearOTO,
    'LoraEmbedding': LoraEmbeddingOTO,
}

# Unsupported e=yet or unprunable Operators
# If one node group contains these operator, mark them as 
UNPRUNABLE_BASIC_OPERATORS = [
    'depthtospace',
]
UNPRUNABLE_COMPOSED_OPERATORS = [
    'LoraLinearOTO',
    'LoraEmbeddingOTO'
]