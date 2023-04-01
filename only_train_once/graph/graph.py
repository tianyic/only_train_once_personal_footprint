import imp
import torch
import numpy as np

from .connected_component import GROUP_TYPE
from .node import Node
from assets.theme import THEMES
from operation.operator import OP_DICT, Operator
from transform.onnx_graph_transform import FRAMEWORK_TRANSFORMS, CONV_BN_FUSE
from distutils.version import LooseVersion

class Graph():
    """Tracks nodes and edges of a directed graph and supports basic operations on them."""
    def __init__(self, model=None, dummy_input=None):
        print("graph constructor")
        self.nodes = dict()
        self.edges = list()
        self.connected_components = dict()
        self.output_nodes = list()
        self.paths = list()
        self.theme = THEMES["basic"]
        # parameters of neural network model with names 
        self.params_grad = dict()
        self.params_no_grad = dict()
        self.params_names = list()
        self.params_onnx = dict()
        self.params_to_nodes = dict()
        self.inputs = dict()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params_grad[name] = param
        
        for name in model.state_dict():
            self.params_names.append(name)
            if name not in self.params_grad:
                self.params_no_grad[name] = model.state_dict()[name]

        if model:
            assert dummy_input is not None, "Dummy_input args must be provided for Pytorch models."
            model = model.eval()
            self.build(model, dummy_input)

            # Apply Transforms
            for t in FRAMEWORK_TRANSFORMS:
                t.apply(self)
            CONV_BN_FUSE.apply(self)

    def id(self, node):
        """Returns a unique node identifier. If the node has an id
        attribute (preferred), it's used. Otherwise, the hash() is returned."""
        return node.id if hasattr(node, "id") else hash(node)

    def add_node(self, node):
        id = self.id(node)
        if len(self.nodes) == 0:
            self.root_id = id
        self.nodes[id] = node

    def root(self):
        try:
            return self.nodes[self.root_id]
        except:
            return None

    def add_edge(self, node1, node2, label=None):
        # If the edge is already present, don't add it again.
        # TODO: If an edge exists with a different label, still don't add it again.
        edge = (self.id(node1), self.id(node2), label)
        if edge not in self.edges:
            self.edges.append(edge)

    def add_edge_by_id(self, vid1, vid2, label=None):
        self.edges.append((vid1, vid2, label))

    def outgoing(self, node):
        """Returns nodes connecting out of the given node (or list of nodes)."""
        nodes = node if isinstance(node, list) else [node]
        node_ids = [self.id(n) for n in nodes]
        # Find edges outgoing from this group but not incoming to it
        outgoing = [self[e[1]] for e in self.edges
                    if e[0] in node_ids and e[1] not in node_ids]
        return outgoing

    def incoming(self, node):
        """Returns nodes connecting to the given node (or list of nodes)."""
        nodes = node if isinstance(node, list) else [node]
        node_ids = [self.id(n) for n in nodes]
        # Find edges incoming to this group but not outgoing from it
        incoming = [self[e[0]] for e in self.edges
                    if e[1] in node_ids and e[0] not in node_ids]
        return incoming

    def siblings(self, node):
        """Returns all nodes that share the same parent (incoming node) with
        the given node, including the node itself.
        """
        incoming = self.incoming(node)
        # TODO: Not handling the case of multiple incoming nodes yet
        if len(incoming) == 1:
            incoming = incoming[0]
            siblings = self.outgoing(incoming)
            return siblings
        else:
            return [node]

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self.nodes.get(k) for k in key]
        else:
            return self.nodes.get(key)

    def remove(self, nodes):
        """Remove a node and its edges."""
        nodes = nodes if isinstance(nodes, list) else [nodes]
        for node in nodes:
            k = self.id(node)
            self.edges = list(filter(lambda e: e[0] != k and e[1] != k, self.edges))
            del self.nodes[k]
    
    def set_connected_components(self, connected_components):
        for cc in connected_components:
            self.connected_components[cc.id] = cc
            for node in cc.nodes.values():
                node.connected_components[cc.id] = cc

    def print(self):
        for node_id in self.nodes:
            print(node_id, self.nodes[node_id])

        for edge in self.edges:
            print(edge)

        for tensor_key in self.tensors:
            print(tensor_key, self.tensors[tensor_key])

    def _parse_tensors_info(self, state_dict, torch_graph_str):
        """Use hack to parse tensor info, should be better option for doing it"""
        prefix_str = "graph"
        assert torch_graph_str.startswith(prefix_str), "Invalid graph str to be parsed"

        tensors_str = get_str_inside_parenthesis(torch_graph_str, prefix_str = prefix_str)
        tensors_str_list = [s.strip() for s in tensors_str.split('%')][1:]

        self.params_id_to_name = dict()

        num_inputs = len(tensors_str_list) - len(state_dict)
        cur_input = 0
        cur_param = 0
        for i, tensor_str in enumerate(tensors_str_list):
            tensor_str_split = [s.strip() for s in tensor_str.split(":")]
            tensor_id = tensor_str_split[0]
            tensor_info = [s.strip() for s in tensor_str_split[1].split(",")]
            required_grad = True if [s for s in tensor_info if "requires_grad" in s][0].split('=')[1] == '1' else False
            tensor_type = "input" if i < num_inputs else "params"
            if tensor_type == "input":
                self.inputs[cur_input] = (tensor_id, tensor_str_split)
                cur_input += 1
            elif tensor_type == "params":
                self.params_id_to_name[int(tensor_id)] = self.params_names[cur_param]
                cur_param += 1

    def get_output_shape(self, torch_node_str):
        str_to_processed = torch_node_str.split(':')[1].strip()
        output_str = get_str_inside_parenthesis(str_to_processed, prefix_str = "Float")
        if output_str is None:
            return None
        output_str_splits = output_str.split(',')
        output_shapes = []
        for item in output_str_splits:
            item = item.strip()
            if item.isnumeric():
                output_shapes.append(int(item))
            else:
                break
        return output_shapes

    def build(self, model, dummy_input=None):
        # Run the Pytorch graph to get a trace and generate a graph from it
        trace_graph, _ = torch.jit._get_trace_graph(model, dummy_input)
        torch_graph = None
        if LooseVersion(torch.__version__) >= LooseVersion('1.9.0') and \
           LooseVersion(torch.__version__) <= LooseVersion('1.11.10'):
            torch_graph = torch.onnx._optimize_trace(trace_graph, torch.onnx.OperatorExportTypes.ONNX)
        elif LooseVersion(torch.__version__) >= LooseVersion('1.13.0'):
            torch_graph = torch.onnx._optimize_graph(trace_graph, torch.onnx.OperatorExportTypes.ONNX)
        else:
            raise "Torch {} is not supported because of some bug in _optimize_trace.".format(torch.__version__)

        # TODO: Should be better way to get the information of tensors from torch_graph
        self._parse_tensors_info(model.state_dict(), str(torch_graph))

        # Loop through nodes from torch_graph to build graph for OTO
        for torch_node in torch_graph.nodes():
            # Get Operation
            op_name = torch_node.kind().split("::")[-1].lower()
            # Operation Parameters
            op_params = {k: getattr(torch_node, torch_node.kindOf(k))(k) for k in torch_node.attributeNames()}
            op = None
            if op_name in OP_DICT:
                op= OP_DICT[op_name]
            else:
                op = Operator(name=op_name, params=op_params)
                OP_DICT[op_name] = op

            # Inputs/outputs
            inputs = [i.unique() for i in torch_node.inputs()]
            outputs = [o.unique() for o in torch_node.outputs()]

            params = [self.params_id_to_name[i] for i in inputs if i in self.params_id_to_name]
            # Only consider non-params inputs
            inputs = [i for i in inputs if i not in self.params_id_to_name]
 
            output_shape = self.get_output_shape(str(torch_node))

            # Add nodes
            node = Node(id=pytorch_id(torch_node), op=op, op_params=op_params, \
                        params=params, inputs=inputs, outputs=outputs, output_shape=output_shape)

            self.add_node(node)
            # Add edges
            for target_torch_node in torch_graph.nodes():
                target_inputs = [i.unique() for i in target_torch_node.inputs()]
                if set(outputs) & set(target_inputs):
                    self.add_edge_by_id(pytorch_id(torch_node), pytorch_id(target_torch_node))

        for node in self.nodes.values():
            if len(self.outgoing(node)) == 0:
                self.output_nodes.append(node.id)

        for i, node in enumerate(self.nodes.values()):
            if len(node.inputs) == 0:
                continue
            nodes_in = self.incoming(node)
            if len(nodes_in) == 0:
                for in_id in node.inputs:
                    in_id = in_id[4:] #skip out- prefix
                    input_tensor_info = self.inputs[int(in_id)]
                    tensor_shape_str = get_str_inside_parenthesis(input_tensor_info[1][1], prefix_str='Float')
                    input_shape = []
                    if tensor_shape_str:
                        tensor_shape_str_splits = tensor_shape_str.split(',')
                        for item in tensor_shape_str_splits:
                            item = item.strip()
                            if item.isnumeric():
                                input_shape.append(int(item))
                            else:
                                break
                    node.input_shape.append(input_shape)
            else:
                for node_in in nodes_in:
                    node.input_shape.append(node_in.output_shape)

    def compress(self):
        import onnx
        from onnx import numpy_helper
        # Each CC prunes row-wisely based on non-zero groups.
        for cc in self.connected_components.values():
            for node in cc.nodes.values():
                node.pruned_shape = [-1] * 2
                if len(cc.non_zero_group_idxes) > 0:
                    node.pruned_shape[0] = len(cc.non_zero_group_idxes)
                    if hasattr(cc, 'raw_num_groups'):
                        if cc.raw_num_groups != cc.num_groups:
                            node.pruned_shape[0] = len(cc.raw_non_zero_group_idxes)
            if cc.type <= GROUP_TYPE["auxilary"]:
                continue
            for name in cc.onnx_params:
                numpy_param = onnx.numpy_helper.to_array(self.params_onnx[name])
                if cc.raw_num_groups == cc.num_groups:
                    pruned_onnx_param = numpy_param[cc.non_zero_group_idxes, ...]
                else:
                    pruned_onnx_param = numpy_param[cc.raw_non_zero_group_idxes, ...]

                onnx_param = onnx.helper.make_tensor(
                        name=name,
                        data_type=onnx.TensorProto.FLOAT,
                        dims=pruned_onnx_param.shape,
                        vals=pruned_onnx_param.flatten().tolist())
                self.params_onnx[name].CopyFrom(onnx_param)

        def dfs_helper(graph, node, cc_id, visited, incoming_cc_ids):
            if node.cc_id != cc_id:
                incoming_cc_ids.add(node.cc_id)
                return 
            visited[node.id] = True
            for node_in in graph.incoming(node):
                if not visited[node_in.id]:                    
                    dfs_helper(graph, node_in, cc_id, visited, incoming_cc_ids)
        
        # Each node prunes redundancy depending on incoming or dependent CC.
        for cc in self.connected_components.values():
            if cc.is_auxilary():
                # print(cc)
                for name in cc.onnx_params:
                    # print(name)
                    node = self.params_to_nodes[name][0]
                    numpy_param = onnx.numpy_helper.to_array(self.params_onnx[name])
                    pruned_onnx_param = numpy_param[cc.non_zero_group_idxes, ...]
                    onnx_param = onnx.helper.make_tensor(
                            name=name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=pruned_onnx_param.shape,
                            vals=pruned_onnx_param.flatten().tolist())
                    self.params_onnx[name].CopyFrom(onnx_param)                     

            for name in cc.onnx_params:
                # TODO: Weight sharing, i.e., one param has multiple nodes
                node = self.params_to_nodes[name][0]
                numpy_param = onnx.numpy_helper.to_array(self.params_onnx[name])
                visited = {}
                for node_id in self.nodes:
                    visited[node_id] = False
                incoming_cc_ids = set()
                dfs_helper(self, node, node.cc_id, visited, incoming_cc_ids)
                # If no-incoming CCs, then the node is directly connected regardless stem node to the input
                # there is no need to adjust parameters
                if len(incoming_cc_ids) == 0:
                    continue
                else:
                    incoming_cc = self.connected_components[next(iter(incoming_cc_ids))]
                    if incoming_cc.type <= GROUP_TYPE['default']:
                        continue
                    if len(numpy_param.shape) >= 2 and (node.is_conv() or node.is_linear()):
                        node.pruned_shape[1] = len(incoming_cc.non_zero_group_idxes)
                        pruned_onnx_param = numpy_param[:, incoming_cc.non_zero_group_idxes, ...]
                        onnx_param = onnx.helper.make_tensor(
                            name=name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=pruned_onnx_param.shape,
                            vals=pruned_onnx_param.flatten().tolist())
                        self.params_onnx[name].CopyFrom(onnx_param)

    def search(self, pattern):
        """Searches the graph for a sub-graph that matches the given pattern
        and returns the first match it finds.
        """
        for node in self.nodes.values():
            match, following = pattern.match(self, node)
            if match:
                return match, following
        return [], None

    def sequence_id(self):
        from random import getrandbits
        return getrandbits(64)

    def replace(self, nodes, node):
        """Replace nodes with node. Edges incoming to nodes[0] are connected to
        the new node, and nodes outgoing from nodes[-1] become outgoing from
        the new node."""
        nodes = nodes if isinstance(nodes, list) else [nodes]
        # Is the new node part of the replace nodes (i.e. want to collapse
        # a group of nodes into one of them)?
        collapse = self.id(node) in self.nodes
        # Add new node and edges
        if not collapse:
            self.add_node(node)
        for in_node in self.incoming(nodes):
            # TODO: check specifically for output_shape is not generic. Consider refactoring.
            self.add_edge(in_node, node, in_node.output_shape if hasattr(in_node, "output_shape") else None)
        for out_node in self.outgoing(nodes):
            self.add_edge(node, out_node, node.output_shape if hasattr(node, "output_shape") else None)
        # Remove the old nodes
        for n in nodes:
            if collapse and n == node:
                continue
            self.remove(n)
    
    def set_zigs(self, opt=None):
        self.auxilary_ccs = dict()
        # First pass to set up stem ccs
        for cc_id in self.connected_components:
            cc = self.connected_components[cc_id]
            cc.set_properties(self.output_nodes, opt)
            if not cc.is_auxilary():
                # If cc is stem but no params_grad, then all params are not required grad
                shapes_grad_params = [self.params_grad[name].shape[0] for name in cc.params if name in self.params_grad]
                if len(shapes_grad_params) > 0:
                    cc.num_groups = max(shapes_grad_params)
                else:
                    shapes_nograd_params = [self.params_no_grad[name].shape[0] for name in cc.params if name in self.params_no_grad]
                    cc.num_groups = max(shapes_nograd_params) if len(shapes_nograd_params) > 0 else 0
                    cc.type = GROUP_TYPE['no-update']
            else:
                self.auxilary_ccs[cc_id] = cc
                cc.dependent_stem_ccs = list()

        # Second pass to tackle auxiliary connected components
        visited = dict()
        for cc_id in self.auxilary_ccs:
            visited[cc_id] = False
        
        def dfs_helper(graph, cc, dependent_stem_ccs):
            if not cc.is_auxilary():
                dependent_stem_ccs.append(cc.id)
                return 
            elif visited[cc.id]:
                dependent_stem_ccs.extend(cc.dependent_stem_ccs)
                return 

            # if cc is auxiliary, get its concat node if any
            concat_node = cc.get_concat_node()
            if concat_node is None:
                return 

            for node_in_id in concat_node.inputs:
                node_in = graph.nodes[node_in_id]
                cc_in = graph.connected_components[node_in.cc_id]
                dfs_helper(self, cc_in, dependent_stem_ccs)

        for auxilary_cc in self.auxilary_ccs.values():
            if visited[auxilary_cc.id] is True:
                continue
            dfs_helper(self, auxilary_cc, auxilary_cc.dependent_stem_ccs)
            visited[auxilary_cc.id] = True
    
        # Third pass for tackling connection between stem cc and auxiliary ccs
        for cc in self.connected_components.values():
            if cc.is_auxilary():
                if len(cc.params) == 0:
                    continue
                offset = 0
                for depend_cc_id in cc.dependent_stem_ccs:
                    depend_cc = self.connected_components[depend_cc_id]
                    depend_cc.auxiliary_ccs.append((cc.id, offset))
                    offset += depend_cc.num_groups

    def random_set_zero_groups(self, group_sparsity=0.5):
        for cc in self.connected_components.values():
            if cc.is_auxilary():
                continue
            # if cc.type is not stem, or default with no group sparsity penalize, skip
            if cc.type <= GROUP_TYPE["auxilary"]:
                continue
            num_zero_groups = np.random.randint(1, cc.num_groups - 1)
            # num_zero_groups = max(max(1, int((group_sparsity + np.random.random()) * cc.num_groups - 1)), cc.num_groups - 1)
            # num_zero_groups = int(group_sparsity * cc.num_groups - 1)
            zero_group_idxes = np.random.choice(list(range(0, cc.num_groups - 1)), num_zero_groups, replace=False)
            cc_params_objs = [self.params_grad[name] if name in self.params_grad else self.params_no_grad[name] for name in cc.params]
            for p in cc_params_objs:
                p.data[zero_group_idxes, ...] = 0.0
            
            for (auxiliary_cc_id, offset) in cc.auxiliary_ccs:
                auxiliary_cc = self.connected_components[auxiliary_cc_id]
                auxiliary_cc_params_objs = [self.params_grad[name] if name in self.params_grad else self.params_no_grad[name] for name in auxiliary_cc.params]
                for p in auxiliary_cc_params_objs:
                    p.data[zero_group_idxes + offset, ...] = 0.0

    def set_zero_groups(self):
        for cc in self.connected_components.values():
            if cc.is_auxilary():
                continue
            cc_params_objs = [self.params_grad[name] if name in self.params_grad else self.params_no_grad[name] for name in cc.params]
            xs = []
            cc.raw_num_groups = cc.num_groups
            for param in cc_params_objs:
                cc.raw_num_groups = param.shape[0]     
                xs.append(param.data.view(cc.num_groups, -1))

            for (auxiliary_cc_id, offset) in cc.auxiliary_ccs:
                auxiliary_cc = self.connected_components[auxiliary_cc_id]
                auxiliary_cc_params_objs = [(name, self.params_grad[name]) if name in self.params_grad else (name, self.params_no_grad[name]) for name in auxiliary_cc.params]
                for name, param in auxiliary_cc_params_objs:
                    if len(param.data.shape) == 1:
                        xs.append(param.data[offset:offset + cc.num_groups,...].unsqueeze(1))
                    else:
                        xs.append(param.data[offset:offset + cc.num_groups,...].view(cc.num_groups, -1))
            if len(xs) == 0:
                continue            
            flatten_x = torch.cat(xs, dim = 1)
            norm_groups = torch.norm(flatten_x, dim=1)
            zero_groups_idxes = norm_groups == 0
            nonzero_groups_idxes = norm_groups != 0
            cc.num_zero_groups = int(torch.sum(zero_groups_idxes).item())
            cc.zero_groups_idxes = np.arange(0, cc.num_groups)[zero_groups_idxes.cpu()]
            cc.non_zero_group_idxes = np.arange(0, cc.num_groups)[nonzero_groups_idxes.cpu()]

            if cc.raw_num_groups != cc.num_groups:
                repeat_time = cc.raw_num_groups // cc.num_groups
                zero_groups_idxes = zero_groups_idxes.repeat_interleave(repeat_time)
                nonzero_groups_idxes = nonzero_groups_idxes.repeat_interleave(repeat_time)
            cc.raw_zero_groups_idxes = np.arange(0, cc.raw_num_groups)[zero_groups_idxes.cpu()]
            cc.raw_non_zero_group_idxes = np.arange(0, cc.raw_num_groups)[nonzero_groups_idxes.cpu()]

        for cc in self.connected_components.values():
            if cc.is_auxilary():
                non_zero_group_idxes = []
                offset = 0
                for dependent_stem_cc_id in cc.dependent_stem_ccs:
                    dependent_stem_cc = self.connected_components[dependent_stem_cc_id]
                    non_zero_group_idxes.append(dependent_stem_cc.non_zero_group_idxes + offset)
                    offset += dependent_stem_cc.num_groups
                cc.non_zero_group_idxes = np.concatenate(non_zero_group_idxes)

    def build_dot(self, verbose=True, vertical=False):
        """
        Generate a GraphViz Dot graph.
        If verbose, then draw more detailed info as well as groups.
        Returns a GraphViz Digraph object.
        """
        from graphviz import Digraph
        import random

        dot = Digraph()
        dot.attr("graph", 
                bgcolor=self.theme["background_color"],
                color=self.theme["outline_color"],
                fontsize=self.theme["font_size"],
                fontcolor=self.theme["font_color"],
                fontname=self.theme["font_name"],
                margin=self.theme["margin"],
                rankdir="TB" if vertical else "LR",
                pad=self.theme["padding"])

        dot.attr("edge", style="solid", 
                color=self.theme["outline_color"],
                fontsize=self.theme["font_size"],
                fontcolor=self.theme["font_color"],
                fontname=self.theme["font_name"])

        # Build GraphViz Digraph
        if not verbose or len(self.connected_components) == 0:
            dot.attr("node", shape="box", 
                    style="filled", margin="0,0",
                    fillcolor=self.theme["fill_color"],
                    color=self.theme["outline_color"],
                    fontsize=self.theme["font_size"],
                    fontcolor=self.theme["font_color"],
                    fontname=self.theme["font_name"])

            for node in self.nodes.values():
                label = "<tr><td cellpadding='6'>{}</td></tr>".format(node.title)
                label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"
                dot.node(str(node.id), label)
        else:
            for cc in self.connected_components.values():
                random_number = random.randint(0,16777215)
                hex_number = str(hex(random_number))
                color ='#'+ hex_number[2:]
                dot.attr("node", shape="box", 
                        style="filled", margin="0,0",
                        fillcolor=color,
                        color=color,
                        fontsize=self.theme["font_size"],
                        fontcolor="#FFFFFF",
                        fontname=self.theme["font_name"])
                for node in cc.nodes.values():
                    label = "<tr><td cellpadding='6'>{}</td></tr>".format(node.title)
                    if node.id:
                        label += "<tr><td>{}</td></tr>".format(node.id)
                    label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"
                    dot.node(str(node.id), label)                        
        for a, b, label in self.edges:
            if isinstance(label, (list, tuple)):
                label = "x".join([str(l or "?") for l in label])
            dot.edge(str(a), str(b), label)
        return dot    
        
    def params_groups(self, epsilon=[]):            
        param_groups = dict()
        
        for cc in self.connected_components.values():
            cc_param_groups = dict()
            cc.params_grad = cc.params & self.params_grad.keys()
            cc_param_groups['cc_id'] = cc.id
            cc_param_groups['params'] = [self.params_grad[name] for name in cc.params_grad] 
            cc_param_groups['group_type'] = cc.type
            if isinstance(epsilon, list):
                cc_param_groups['epsilon'] = cc.epsilon if len(epsilon) == 0 else epsilon[cc.type]
            else:
                cc_param_groups['epsilon'] = epsilon
            cc_param_groups['upper_group_sparsity'] = cc.upper_group_sparsity
            cc_param_groups['names'] = cc.params_grad
            cc_param_groups['shapes'] = [self.params_grad[name].shape for name in cc.params_grad] 
            cc_param_groups['auxiliary_ccs'] = list()
            if len(cc_param_groups['params']) > 0:
                cc_param_groups['num_groups'] = cc.num_groups
                param_groups[cc.id] = cc_param_groups

        # Second pass for tackling auxliary cc
        for cc in self.connected_components.values():
            if cc.is_auxilary():
                if len(cc.params) == 0:
                    continue
                offset = 0
                for depend_cc_id in cc.dependent_stem_ccs:
                    depend_cc = self.connected_components[depend_cc_id]
                    depend_cc_param_groups = param_groups[depend_cc.id]
                    depend_cc_param_groups['auxiliary_ccs'].append((cc.id, offset))
                    offset += depend_cc.num_groups
        param_groups = dict(sorted(param_groups.items(), key=lambda kv:(kv[0], kv[1])))
        return param_groups.values()
    
def pytorch_id(node):
    """Returns a unique ID for a node."""
    return "out-" + "-".join(["{}".format(o.unique()) for o in node.outputs()])

def get_str_inside_parenthesis(str_to_processed, prefix_str=None):
    if not str_to_processed.startswith(prefix_str):
        return None
    stack = []
    start_idx = len(prefix_str) + 1
    end_idx = -1 
    for c in str_to_processed:
        if c == '(':
            stack.append(c)
        elif c == ')':
            stack.pop()
        end_idx += 1
        if len(stack) == 0 and end_idx > len(prefix_str):
            break
    return str_to_processed[start_idx : end_idx] 
