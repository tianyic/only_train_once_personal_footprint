from enum import IntEnum

class TensorTransform(IntEnum):
    NO_UPDATE = 0
    NO_PRUNE = 1
    BASIC = 2
    ACCESSORY = 3
    MULTIHEAD = 4
    REVERSE_MULTIHEAD = 5
    AUXILIARY = 6
    TRANSPOSE = 7

    TOTAL = 8
    
def tensor_transformation(tensor, transformation_type, num_groups=1, num_heads=1):
    if transformation_type == TensorTransform.NO_UPDATE or \
       transformation_type == TensorTransform.NO_PRUNE:
        return tensor 
    elif transformation_type == TensorTransform.BASIC:
        return basic_transformation(tensor, num_groups)
    elif transformation_type == TensorTransform.ACCESSORY:
        return basic_transformation(tensor, num_groups)
    elif transformation_type == TensorTransform.MULTIHEAD:
        return multihead_transformation(tensor, num_groups, num_heads)
    elif transformation_type == TensorTransform.REVERSE_MULTIHEAD:
        return reverse_multihead_transformation(tensor, num_groups, num_heads)
    elif transformation_type == TensorTransform.TRANSPOSE:
        return transpose_transformation(tensor, num_groups)
    
def basic_transformation(tensor, num_groups=1):
    return tensor.view(num_groups, -1)

def multihead_transformation(tensor, num_groups=1, num_heads=1):
    return tensor.view(num_heads, num_groups, -1).permute(1, 0, 2).contiguous().view(num_groups, -1)

def reverse_multihead_transformation(tensor, num_groups=1, num_heads=1):
    if tensor.numel() >= num_groups * num_heads:
        return tensor.view(num_groups, num_heads, -1).permute(1, 0, 2).contiguous().view(num_heads * num_groups, -1)
    else:
        if len(tensor.shape) == 1:
            return tensor.unsqueeze(1).repeat(1, num_heads).view(num_groups, num_heads, -1).permute(1, 0, 2).contiguous()\
                    .view(num_heads * num_groups, -1).squeeze()
        else:
            return tensor

def transpose_transformation(tensor, num_groups=1):
    if len(tensor.shape) == 1:
        return tensor.view(num_groups, -1)
    elif len(tensor.shape) == 2:
        return tensor.permute(1, 0).contiguous().view(num_groups, -1)
    elif len(tensor.shape) == 4:
        return tensor.permute(1, 0, 2, 3).contiguous().view(num_groups, -1)

