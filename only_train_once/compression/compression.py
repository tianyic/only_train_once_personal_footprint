from time import sleep
import torch
import os

def automated_compression(oto_graph, model, dummy_input, compressed_model_path, dynamic_axes=[False, dict()]):
    compressed_model_path = './' if compressed_model_path is None else compressed_model_path
    full_group_sparse_model_path = os.path.join(compressed_model_path, (model.name if hasattr(model, 'name') else type(model).__name__) + "_full_group_sparse.onnx" )
    compressed_model_path = os.path.join(compressed_model_path, (model.name if hasattr(model, 'name') else type(model).__name__) + "_compressed.onnx" )
    if dynamic_axes[0]:
        torch.onnx.export(
            model, 
            dummy_input, 
            full_group_sparse_model_path, 
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes[1]
        )
    else:
        torch.onnx.export(
            model, 
            dummy_input, 
            full_group_sparse_model_path)  

    import onnx
    max_try_count = 10
    try_count = 0
    onnx_model = None
    while True:
        sleep(1)
        if os.path.isfile(full_group_sparse_model_path):
            onnx_model = onnx.load(full_group_sparse_model_path)
            break
        try_count += 1
        if try_count >= max_try_count:
            break
    
    onnx.checker.check_model(onnx_model)
    onnx_graph = onnx_model.graph
    
    oto_graph.set_zero_groups()
    assign_onnx_tensors_on_oto(oto_graph, onnx_graph)

    for param in oto_graph.params_onnx:
        if param not in oto_graph.params_to_nodes:
            oto_graph.params_to_nodes[param] = list()
        for node in oto_graph.nodes.values():
            if param in node.params:
                oto_graph.params_to_nodes[param].append(node)
    
    oto_graph.compress()

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, compressed_model_path)

    return onnx_model, compressed_model_path, full_group_sparse_model_path


def assign_onnx_tensors_on_oto(oto_graph, onnx_graph):
    # print("assign_onnx_tensors_on_oto")
    remaining_tensors = list()
    for i, tensor in enumerate(onnx_graph.initializer):
        # print(f"Tensor Name: {tensor.name}, Data Type: {tensor.data_type}, Shape: {tensor.dims}")
        matched = False
        for cc in oto_graph.connected_components.values():
            if tensor.name in cc.params:
                cc.onnx_params.add(tensor.name)
                oto_graph.params_onnx[tensor.name] = tensor  
                matched = True
        if not matched:
            remaining_tensors.append(tensor)

    conv_param_count = 0
    for tensor in remaining_tensors:
        if "Conv" not in tensor.name and "conv" not in tensor.name:
            continue
        # print(f"Tensor Name: {tensor.name}, Data Type: {tensor.data_type}, Shape: {tensor.dims}")
        idx = conv_param_count // 2
        conv_param_count += 1
        conv_bn_fuse = oto_graph.fused_conv_bns[idx]
        cc_id = conv_bn_fuse[0].cc_id
        cc = oto_graph.connected_components[cc_id]
        cc.onnx_params.add(tensor.name)
        oto_graph.params_onnx[tensor.name] = tensor
        
        for node in conv_bn_fuse:
            if node.is_conv():
                node.params.append(tensor.name)
