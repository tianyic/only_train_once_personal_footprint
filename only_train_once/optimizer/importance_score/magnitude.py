import torch
from only_train_once.transform import tensor_transformation, TensorTransform

def importance_score_by_magnitude_dhspg(param_group):
    norm_group = None
    for param, p_transform in zip(param_group['params'], param_group['p_transform']):
        param_transform = None
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])
        if norm_group == None:
            norm_group = torch.norm(param_transform, dim=1) ** 2
        else:
            norm_group += torch.norm(param_transform, dim=1) ** 2
    param_group['importance_scores']['magnitude'] = torch.sqrt(norm_group)

def importance_score_by_avg_magnitude_dhspg(param_group):
    norm_group = None
    group_sizes = 0
    for param, p_transform in zip(param_group['params'], param_group['p_transform']):
        param_transform = None
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])
        if norm_group == None:
            norm_group = torch.norm(param_transform, dim=1) ** 2
        else:
            norm_group += torch.norm(param_transform, dim=1) ** 2
        group_sizes += param_transform.shape[1]
    param_group['importance_scores']['avg_magnitude'] = torch.sqrt(norm_group) / float(group_sizes + 1e-6)

def importance_score_by_magnitude_lhspg(param_group):
    norm_group = None
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        if 'lora_A' in p_name or 'lora_B' in p_name:
            continue
        param_transform = None
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])
        if norm_group == None:
            norm_group = torch.norm(param_transform, dim=1) ** 2
        else:
            norm_group += torch.norm(param_transform, dim=1) ** 2
    param_group['importance_scores']['magnitude'] = torch.sqrt(norm_group)

def importance_score_by_avg_magnitude_lhspg(param_group):
    norm_group = None
    group_sizes = 0
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        if 'lora_A' in p_name or 'lora_B' in p_name:
            continue
        param_transform = None
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])
        if norm_group == None:
            norm_group = torch.norm(param_transform, dim=1) ** 2
        else:
            norm_group += torch.norm(param_transform, dim=1) ** 2
        group_sizes += param_group['num_groups']
    param_group['importance_scores']['avg_magnitude'] = torch.sqrt(norm_group) / float(group_sizes + 1e-6)