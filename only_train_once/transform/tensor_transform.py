from enum import IntEnum

class TensorTransform(IntEnum):
    NO_UPDATE = 0
    NO_PRUNE = 1
    BASIC = 2
    ACCESSORY = 3
    MULTIHEAD_HEADDIM = 4 # Only affects the tensor itself
    MULTIHEAD_NUMHEAD = 5 # Only affects the tensor itself
    REVERSE_MULTIHEAD_HEADDIM = 6 # Only affects the tensor itself
    REVERSE_MULTIHEAD_NUMHEAD = 7 # Only affects the tensor itself
    AUXILIARY = 8
    TRANSPOSE = 9
    MULTIHEAD_NUMHEAD_SPREAD = 10 # Affect other nodes in the same node group
    REVERSE_MULTIHEAD_NUMHEAD_SPREAD = 11 # Affect other nodes in the same node group

    TOTAL = 12

def is_spread_transformation(transformation_type):
    if transformation_type == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD:
        return True
    elif transformation_type == TensorTransform.REVERSE_MULTIHEAD_NUMHEAD_SPREAD:
        return True
    else:
        return False

SPREAD_TRANSFORM_MAP = {
    TensorTransform.MULTIHEAD_NUMHEAD_SPREAD: TensorTransform.MULTIHEAD_NUMHEAD
}

def tensor_transformation(tensor, transformation_type, num_groups=1, num_heads=1, head_dim=1):
    if transformation_type == TensorTransform.NO_UPDATE or \
       transformation_type == TensorTransform.NO_PRUNE:
        return tensor 
    elif transformation_type == TensorTransform.BASIC:
        return basic_transformation(tensor, num_groups)
    elif transformation_type == TensorTransform.ACCESSORY:
        return basic_transformation(tensor, num_groups)
    elif transformation_type == TensorTransform.MULTIHEAD_HEADDIM:
        return multihead_headdim_transformation(tensor, num_groups, num_heads)
    elif transformation_type == TensorTransform.MULTIHEAD_NUMHEAD:
        return multihead_numhead_transformation(tensor, num_groups)
    elif transformation_type == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD:
        return multihead_numhead_transformation(tensor, num_groups)
    elif transformation_type == TensorTransform.REVERSE_MULTIHEAD_HEADDIM:
        return reverse_multihead_headdim_transformation(tensor, num_groups, num_heads)
    elif transformation_type == TensorTransform.REVERSE_MULTIHEAD_NUMHEAD:
        return reverse_multihead_numhead_transformation(tensor, num_groups, head_dim)
    elif transformation_type == TensorTransform.TRANSPOSE:
        return transpose_transformation(tensor, num_groups)
    
def basic_transformation(tensor, num_groups=1):
    return tensor.view(num_groups, -1)

def multihead_headdim_transformation(tensor, num_groups=1, num_heads=1):
    return tensor.view(num_heads, num_groups, -1).permute(1, 0, 2).contiguous().view(num_groups, -1)

def multihead_numhead_transformation(tensor, num_groups=1):
    return tensor.view(num_groups, -1)

def reverse_multihead_headdim_transformation(tensor, num_groups=1, num_heads=1):
    if tensor.numel() >= num_groups * num_heads:
        return tensor.view(num_groups, num_heads, -1).permute(1, 0, 2).contiguous().view(num_heads * num_groups, -1)
    else:
        if len(tensor.shape) == 1:
            return tensor.unsqueeze(1).repeat(1, num_heads).view(num_groups, num_heads, -1).permute(1, 0, 2).contiguous()\
                    .view(num_heads * num_groups, -1).squeeze()
        else:
            return tensor

def reverse_multihead_numhead_transformation(tensor, num_groups=1, head_dim=1):
    if tensor.numel() >= num_groups * head_dim:
        raise NotImplementedError
    else:
        if len(tensor.shape) == 1:
            return tensor.unsqueeze(1).repeat(1, head_dim).view(num_groups * head_dim, -1).squeeze()
        else:
            return tensor
        
def transpose_transformation(tensor, num_groups=1):
    if len(tensor.shape) == 1:
        return tensor.view(num_groups, -1)
    elif len(tensor.shape) == 2:
        return tensor.permute(1, 0).contiguous().view(num_groups, -1)
    elif len(tensor.shape) == 4:
        return tensor.permute(1, 0, 2, 3).contiguous().view(num_groups, -1)