import torch
import os

def automated_pruning_compression(oto_graph, model, merge_lora_to_base, unmerge_lora_to_base, export_huggingface_format, export_float16, \
                          full_group_sparse_model_dir, compressed_model_dir, save_full_group_sparse_model, ckpt_format):
    
    full_group_spase_model_name = None
    compressed_model_name = None
    model_name_prefix =  (model.name if hasattr(model, 'name') else type(model).__name__)
    if ckpt_format == 'torch':
        full_group_spase_model_name = model_name_prefix + "_full_group_sparse.pt"
        compressed_model_name = model_name_prefix + "_compressed.pt"
    elif ckpt_format == 'onnx':
        full_group_spase_model_name = model_name_prefix + "_full_group_sparse.onnx"
        compressed_model_name = model_name_prefix + "_compressed.onnx"
    full_group_sparse_model_path = os.path.join(full_group_sparse_model_dir, full_group_spase_model_name)
    compressed_model_path = os.path.join(compressed_model_dir, compressed_model_name)

    if export_huggingface_format:
        full_group_sparse_model_dir = os.path.join(full_group_sparse_model_dir, 'huggingface_format_full')
        compressed_model_dir = os.path.join(compressed_model_dir, 'huggingface_format_compressed')
        full_group_sparse_model_path = full_group_sparse_model_dir
        compressed_model_path = compressed_model_dir
        
    os.makedirs(full_group_sparse_model_dir, exist_ok=True)
    os.makedirs(compressed_model_dir, exist_ok=True)
    
    if export_float16:
        model.half()
    
    if save_full_group_sparse_model:
        if export_huggingface_format:
            model.save_pretrained(full_group_sparse_model_path)
        elif ckpt_format == 'torch':
            torch.save(model, full_group_sparse_model_path)   
        elif ckpt_format == 'onnx':
            torch.onnx.export(
                model,
                oto_graph.dummy_input,
                full_group_sparse_model_path)
    oto_graph.set_pruning_redundant_idxes()

    # First pass conduct out-channel pruning
    pruned_out_dim_modules = set()
    for node_group in oto_graph.node_groups.values():
        if not node_group.is_prunable and not node_group.is_auxiliary:
            continue
        node_group.prune_out_dim(global_skip_modules=pruned_out_dim_modules)
        pruned_out_dim_modules = pruned_out_dim_modules.union(node_group.get_modules())

    # Second pass conduct in-channel pruning
    def find_incoming_node_group_stem_node(graph, node, src_ng, visited, incoming_node_groups, incoming_stem_node_ids):
        if src_ng.id not in node.node_group_ids and not src_ng.contain_node(node):
            incoming_node_groups.update(node.node_group_ids)
            return 
        visited[node.id] = True
        for node_in in graph.incoming(node):
            if node_in.is_stem():
                incoming_stem_node_ids.add(node_in)
                return     
            if not visited[node_in.id]:                    
                find_incoming_node_group_stem_node(graph, node_in, src_ng, visited, incoming_node_groups, incoming_stem_node_ids)
    
    pruned_in_dim_modules = set()

    verbose = False
    for node_group in oto_graph.node_groups.values():
        for node in node_group.nodes.values():
            if node.pruned_status['in_dim']:
                continue
            
            if node.op.module in pruned_in_dim_modules:
                continue

            if not hasattr(node.op, 'prune_in_dim'):
                continue

            incoming_node_groups = set()
            incoming_stem_nodes = set()
            
            find_incoming_node_group_stem_node(oto_graph, node, node_group, oto_graph.visited_dict(), \
                                               incoming_node_groups, incoming_stem_nodes)
                
            in_channel_pruned_idxes = None
            if len(incoming_stem_nodes) > 0:
                incoming_stem_node = next(iter(incoming_stem_nodes))
                incoming_ng = oto_graph.node_groups[incoming_stem_node.node_group_ids[0]]
                in_channel_pruned_idxes = incoming_ng.pruning_redundant_idxes
            elif len(incoming_node_groups) > 0:
                incoming_ng_id = None
                for ng_id in incoming_node_groups:
                    ng = oto_graph.node_groups[ng_id]
                    if ng.is_prunable or ng.is_auxiliary:
                        incoming_ng_id = ng_id
                    elif not ng.is_prunable and len(ng.param_names) > 0:
                        incoming_ng_id = None
                        break
                if incoming_ng_id is None:
                    continue
                incoming_ng = oto_graph.node_groups[incoming_ng_id]
                in_channel_pruned_idxes = incoming_ng.pruning_redundant_idxes

            if in_channel_pruned_idxes is None:
                continue
            
            if hasattr(incoming_ng, 'op'):
                num_heads = 1
                head_dim = 1
                if hasattr(incoming_ng.op, 'num_heads'):
                    num_heads = incoming_ng.op.num_heads
                if hasattr(incoming_ng.op, 'head_dim'):
                    head_dim = incoming_ng.op.head_dim
                if num_heads > 1 and head_dim > 1:
                    in_channel_pruned_idxes = list()
                    for h in range(num_heads):
                        in_channel_pruned_idxes.extend([i + h * head_dim for i in incoming_ng.pruning_redundant_idxes])
                
            # To tackle reshape as flatten operator followed by linear operator
            node_in = oto_graph.incoming(node)[0]
            if node_in.op_name == 'flatten' and node.op_name == 'linear':
                expand_time = node.op.module.in_features // incoming_ng.get_num_groups()
                in_channel_pruned_idxes_refined = list()
                for idx in in_channel_pruned_idxes:
                    in_channel_pruned_idxes_refined.extend([i + idx * expand_time for i in range(expand_time)])
                in_channel_pruned_idxes = in_channel_pruned_idxes_refined
            
            if not node.pruned_status['in_dim']:
                node.op.prune_in_dim(pruned_idxes=in_channel_pruned_idxes, param_names=node.param_names, verbose=verbose)
                node.pruned_status['in_dim'] = True
                # Skip composed node group since such groups may contain multiple nodes correspond to the same module 
                if node.op.is_basic and not node_group.contain_lora():
                    pruned_in_dim_modules.add(node.op.module)
                
    if merge_lora_to_base:
        if hasattr(model, 'merge_and_unload'):
            model = model.merge_and_unload()

    if unmerge_lora_to_base:
        if hasattr(model, 'unmerge_and_unload'):
            model = model.unmerge_and_unload()
            
    if export_huggingface_format:
        model.save_pretrained(compressed_model_path)
    elif ckpt_format == 'torch':
        torch.save(model, compressed_model_path)
    elif ckpt_format == 'onnx':
        torch.onnx.export(
            model,
            oto_graph.dummy_input,
            compressed_model_path)
    return compressed_model_path, full_group_sparse_model_path 