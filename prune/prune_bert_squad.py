import os
import sys
import argparse
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from backend.bert import BertConfig, BertForQuestionAnswering
import torch
from torch import nn
from pytorch_transformers import BertTokenizer
from utils.utils_squad import create_group_params_config, evaluate
import copy

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True, type=str) 
    parser.add_argument('--eval_data', required=True, type=str) 
    return parser.parse_args()

def get_layer_idx(idx, infos):
    for info in infos:
        if idx in info[0:2]:
            return info[-1]
    return None

def get_xs(params):
    xs = []
    for p in params:
        if len(p.data.shape) == 1:
            xs.append(p.data.unsqueeze(1))
        else:
            xs.append(p.data)
    return xs    

def reorg_multi_head(x, num_groups, num_heads):
    return x.view(num_heads, num_groups, -1).permute(1, 0, 2).contiguous().view(num_groups, -1)

def get_pruned_configs(model, model_config):
    # configure the pruned_model configs
    pruned_model_configs = [copy.deepcopy(model_config) for _ in range(model_config.num_hidden_layers)]

    layers_info = [[2, 14, 0], [3, 15, 1], [4, 16, 2], [5, 17, 3], [6, 18, 4], [7, 19, 5], [8, 20, 6], [9, 21, 7], [10, 22, 8], [11, 23, 9],\
        [12, 24, 10], [13, 25, 11]]
    optimizer_grouped_parameters = create_group_params_config(model, [0.0] * 5, [0.0] * 5)
    non_zero_idxes = dict()
    for i, group in enumerate(optimizer_grouped_parameters):
        if group['group_type'] == 0 or group['group_type'] == 1:
            continue
        layer_idx = get_layer_idx(i, layers_info)
        layer_config = pruned_model_configs[layer_idx]
        if group['group_type'] == 2:
            xs = get_xs(group['params'])
            flatten_x = torch.cat(xs, dim = 1)
            num_groups = flatten_x.shape[0] // group['num_heads']
            flatten_x = reorg_multi_head(flatten_x, num_groups, group['num_heads'])

            flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
            nonzero_group_idxes = flatten_x_norm != 0.0
            num_nonzero_groups = torch.sum(nonzero_group_idxes)
            layer_config.q_dim = int(num_nonzero_groups) * group['num_heads']
            layer_config.k_dim = int(num_nonzero_groups) * group['num_heads']
            for name in group['names']:
                non_zero_idxes[name] = [nonzero_group_idxes, group['group_type'], layer_config]

        elif group['group_type'] == 3:
            xs = get_xs(group['params'])
            flatten_x = torch.cat(xs, dim = 1)
            num_groups = flatten_x.shape[0]
            flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
            nonzero_group_idxes = flatten_x_norm != 0.0
            num_nonzero_groups = torch.sum(nonzero_group_idxes)    
            layer_config.intermediate_size = int(num_nonzero_groups)       
            for name in group['names']:
                non_zero_idxes[name] = [nonzero_group_idxes, group['group_type'], layer_config]
    return pruned_model_configs, non_zero_idxes

def assign_weights(pruned_model, model, pruned_model_configs, non_zero_idxes):
    gs_param_names = [p_name for p_name in model.state_dict()]
    pruned_param_names = [p_name for p_name in pruned_model.state_dict()]
    output_dense_weights_idx = [17, 33, 49, 65, 81, 97, 113, 129, 145, 161, 177, 193]
    for i, (gs_p_name, pruned_p_name) in enumerate(zip(gs_param_names, pruned_param_names)):
        gs_param = model.state_dict()[gs_p_name]
        pruned_param = pruned_model.state_dict()[pruned_p_name]

        if i in output_dense_weights_idx:
            target_name = gs_param_names[i-1]
            nonzero_group_idxes, group_type, pruned_model_config = non_zero_idxes[target_name]
            pruned_param.data.copy_(gs_param.data[:, nonzero_group_idxes])
        else:
            if pruned_p_name not in non_zero_idxes:
                pruned_param.data.copy_(gs_param.data)  
            else:
                nonzero_group_idxes, group_type, pruned_model_config = non_zero_idxes[pruned_p_name]
                if group_type == 2:
                    if len(gs_param.shape) == 1:
                        gs_param = gs_param.view(model_config.num_attention_heads, \
                            model_config.hidden_size // model_config.num_attention_heads)
                        pruned_param = pruned_param.view(pruned_model_config.num_attention_heads, \
                            pruned_model_config.q_dim // pruned_model_config.num_attention_heads)
                        pruned_param.data.copy_(gs_param[:, nonzero_group_idxes])
                        pruned_param = pruned_param.view(pruned_model_config.q_dim)
                    elif len(gs_param.shape) == 2:
                        gs_param = gs_param.view(model_config.num_attention_heads, \
                            model_config.hidden_size // model_config.num_attention_heads,\
                            model_config.hidden_size)
                        pruned_param = pruned_param.view(pruned_model_config.num_attention_heads, \
                            pruned_model_config.q_dim // pruned_model_config.num_attention_heads,\
                            pruned_model_config.hidden_size)
                        pruned_param.data.copy_(gs_param[:, nonzero_group_idxes, :])
                        pruned_param = pruned_param.view(pruned_model_config.q_dim, pruned_model_config.hidden_size)
                elif group_type == 3:
                    pruned_param.data.copy_(gs_param.data[nonzero_group_idxes, ...])

def get_num_params(model, excluded_embedding=True):
    num_params = 0
    for i, (name, param) in enumerate(model.named_parameters()):
        if excluded_embedding and "embedding" in name:
            continue
        num_params += param.shape.numel()
    return num_params

if __name__ == "__main__":
    args = ParseArgs()

    checkpoint_dir = args.checkpoint_dir
    eval_data_input_file = args.eval_data
    config_name = "config.json"
    model_name = "pytorch_model.bin"

    max_seq_length = 384
    doc_stride = 128
    max_query_length = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Construct Original Model 
    print("Construct original full group sparse model")
    model_config = BertConfig.from_json_file(os.path.join(checkpoint_dir, config_name))
    model = BertForQuestionAnswering(model_config)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, model_name), map_location=device))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    print("Get pruned model config")
    pruned_model_configs, non_zero_idxes = get_pruned_configs(model, model_config)
    pruned_model = BertForQuestionAnswering(pruned_model_configs)

    print("Assign weights from full model to pruned model")
    assign_weights(pruned_model, model, pruned_model_configs, non_zero_idxes)

    print("Save pruned model to: ", os.path.join(args.checkpoint_dir, 'pruned_model.bin'))
    torch.save(pruned_model, os.path.join(args.checkpoint_dir, 'pruned_model.bin'))

    num_params_model = get_num_params(model)
    num_params_pruned_model = get_num_params(pruned_model)
    print("Original full model number of parameters:", num_params_model)
    print("Pruned model number of parameters:", num_params_pruned_model)
    print("Ratio: ", num_params_pruned_model / num_params_model)

    print("Evaludate full group sparse model:")
    model.to(device)
    evaluate(model, eval_data_input_file, checkpoint_dir, tokenizer, max_seq_length, doc_stride, max_query_length)

    print("Evaludate pruned model:")
    pruned_model.to(device)
    evaluate(pruned_model, eval_data_input_file, checkpoint_dir, tokenizer, max_seq_length, doc_stride, max_query_length)
