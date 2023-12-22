import torch
from torch import _C
import numpy as np
from distutils.version import LooseVersion
from collections import defaultdict
from only_train_once.operation import COMPOSED_MODULES, BASIC_MODULES, Operator, ParamOTO
from only_train_once.assets import THEMES
from .node import Node
from .node_group import NodeGroupComposedOp, NodeGroup
from only_train_once.transform.graph_transform import FRAMEWORK_TRANSFORMS
from only_train_once.transform import TensorTransform
import warnings

if LooseVersion(torch.__version__) >= LooseVersion('1.13.0'):
    from torch.onnx._globals import GLOBALS
    #tested basd on 1.13 default value 14 does not support gridsample op in onnx
    GLOBALS.export_onnx_opset_version = 16 

class Graph:
    """Tracks nodes and edges of a directed graph and supports basic operations on them."""
    def __init__(self, model=None, dummy_input=None, trace_onnx=True, skip_patterns=None, strict_out_nodes=False):
        print("OTO graph constructor")
        self.inputs = dict()
        self.nodes = dict()
        self.edges = list()
        self.node_groups = dict()
        self.output_nodes = dict()
        self.input_nodes = dict()
        self.dummy_input = dummy_input

        self.params_grad = dict()
        self.params_no_grad = dict()
        self.param_names = list()
        self.trace_onnx = trace_onnx
        self.root_module = None
        
        self.theme = THEMES["basic"]
        self.skip_patterns = []
        if skip_patterns is not None:
            if isinstance(skip_patterns, str):
                self.skip_patterns = [skip_patterns]
            elif isinstance(skip_patterns, list):
                try:
                    assert(all([isinstance(a,str) for a in skip_patterns]))
                except:
                    raise ValueError("skip_patterns only supports string or list of strings")
                self.skip_patterns = skip_patterns
            else:
                raise ValueError("skip_patterns only supports string or list of strings")
        
        # If True, only consider the nodes without outgoing nodes as out_nodes
        self.strict_out_nodes = strict_out_nodes
        if not model:
            return 

        self._model = model
        self.set_param_grad_no_grad(self._model)

        assert dummy_input is not None, "Dummy_input args must be provided for Pytorch models."
        model = model.eval()
        self.build(model, dummy_input)
        if len(self.skip_patterns) > 0:
            self.remove_patterns(self.skip_patterns)
        # Apply Transforms
        for t in FRAMEWORK_TRANSFORMS:
            t.apply(self)
            
    def build(self, model, dummy_input):
        print("graph build")
        trace_graph = self._get_trace_graph(model, dummy_input, optimized_onnx=self.trace_onnx)
        self._parse_modules(model)

        # TODO: Should be better way to get the information of tensors from torch_graph
        self._parse_tensors_info(model.state_dict(), str(trace_graph))

        torch_nodes_by_inputs = defaultdict(set)
        torch_nodes_by_outputs = defaultdict(set)
        
        for torch_node in trace_graph.nodes():
            # Get Operation
            op_name = torch_node.kind().split("::")[-1].lower().replace('_', '')
            # Operation Parameters
            op_cfg_params = {k: getattr(torch_node, torch_node.kindOf(k))(k) for k in torch_node.attributeNames()}

            # Inputs/outputs
            inputs = [i.unique() for i in torch_node.inputs()]
            outputs = [o.unique() for o in torch_node.outputs()]

            param_names = [self.param_id_to_name[i] for i in inputs if i in self.param_id_to_name]

            op = None
            if len(param_names) > 0:
                if param_names[0] in self.param_name_to_operator:
                    op = self.param_name_to_operator[param_names[0]]
                elif len(param_names) == 1 and param_names[0] in self.params_grad:
                    op = ParamOTO(_type=op_name, cfg_params=op_cfg_params, param_name=param_names[0],\
                                  param=self.params_grad[param_names[0]])
                else:
                    op = Operator(_type=op_name, cfg_params=op_cfg_params)
                    for param_name in param_names:
                        op.name_to_param[param_name] = self.params_grad[param_name] if param_name in self.params_grad \
                                                       else self.params_no_grad[param_name]
            else:
                op = Operator(_type=op_name, cfg_params=op_cfg_params)
                
            # Note that op_name may not equals to op.id if belongs to the composed operator. 
            node = Node(id=self.torch_node_id(torch_node), op_name=op_name, op=op, inputs=inputs, outputs=outputs, param_names=param_names)
            if op.id in self.op_name_to_node_group_comp_op:
                self.op_name_to_node_group_comp_op[op.id].add_node(node)
            
            self.add_node(node)

            # Add edges
            for output in outputs:
                torch_nodes_by_outputs[output].add(torch_node)
                for target_torch_node in torch_nodes_by_inputs[output]:
                    self.add_edge_by_id(self.torch_node_id(torch_node), self.torch_node_id(target_torch_node))
            for input in inputs:
                torch_nodes_by_inputs[input].add(torch_node)
                for target_torch_node in torch_nodes_by_outputs[input]:
                    self.add_edge_by_id(self.torch_node_id(target_torch_node), self.torch_node_id(torch_node))            
        for op_id in self.op_name_to_node_group_comp_op:
            node_group = self.op_name_to_node_group_comp_op[op_id]
            self.node_groups[node_group.id] = node_group

        out_ids = set()
        for node in self.nodes.values():
            if len(self.outgoing(node)) == 0:
                out_ids.add(node.id)
        if not self.strict_out_nodes:
            for out in trace_graph.outputs():
                out_id = str(out).split()[0]
                out_ids.add('node-' + out_id)

        # Set up nodes that are directly connected to the output and the input
        for node in self.nodes.values():
            if node.id in out_ids:
                self.output_nodes[node.id] = node
            if len(set(node.inputs).intersection(set(self.inputs.keys()))) > 0:
                self.input_nodes[node.id] = node
        
        # Add dummy input and output node
        dummy_input_node = Node(id='dummy_input', op_name='dummy_input')
        dummy_output_node = Node(id='dummy_output', op_name='dummy_output')
        self.add_node(dummy_input_node)
        self.add_node(dummy_output_node)
        for input_node in self.input_nodes.values():
            self.add_edge(dummy_input_node, input_node)
        for output_node in self.output_nodes.values():
            self.add_edge(output_node, dummy_output_node)
            
        if self.trace_onnx:
            self.replace_eligible_matmul_as_linear()
            self.remove_isolated_nodes()
        
    def remove_patterns(self, skip_patterns):
        # Warning: This method does not gurantee the validity of the graph. Users should be careful when using the method.
        # remove path patterns in dfs order
        # pattern expected to be in the form of a path "a->b->c", any part that matches this pattern will be removed.
        # e.g. Given a model "input->conv->bn->conv->bn->conv->output" and remove "conv->bn->conv" will result only 
        # "input" and "output" nodes left and disconnected.
        
        warnings.warn("This method does not gurantee the validity of the graph. Users should be careful when using this method.")
        all_remove_nodes = []
        for pattern in skip_patterns:
            nodes_path_to_remove = self._find_remove_pattern(pattern)
            all_remove_nodes.append(nodes_path_to_remove)
            
        all_remove_nodes_unique = []
        for pattern in all_remove_nodes:
            for found_path in pattern:
                for node in found_path:
                    all_remove_nodes_unique.append(node)
        all_remove_nodes_unique = list(set(all_remove_nodes_unique))
        for node_to_remove in all_remove_nodes_unique:
            self.nodes.pop(node_to_remove)
            if node_to_remove in self.input_nodes:
                self.input_nodes.pop(node_to_remove)
            if node_to_remove in self.output_nodes:
                self.output_nodes.pop(node_to_remove)
                
        edges_new = []
        for edge in self.edges:
            if edge[0] in all_remove_nodes_unique or edge[1] in all_remove_nodes_unique:
                continue
            else:
                edges_new.append(edge)
        self.edges = edges_new
        
        disconnected_nodes = self._find_disconnected_nodes()
        for disconnected_node in disconnected_nodes:
            self.nodes.pop(disconnected_node)
            if disconnected_node in self.input_nodes:
                self.input_nodes.pop(disconnected_node)
            if disconnected_node in self.output_nodes:
                self.output_nodes.pop(disconnected_node)
            
        edges_new = []
        for edge in self.edges:
            if edge[0] in disconnected_nodes or edge[1] in disconnected_nodes:
                continue
            else:
                edges_new.append(edge)
        self.edges = edges_new
        
    def _find_remove_pattern(self, pattern):
        
        pattern_node_names = pattern.split("->")
        
        def _dfs_helper(node, node_names):
            
            remove_nodes = []
            if node is None or node.op_name is None:
                return None
            
            if node.op_name == node_names[0]:
                outgoing_nodes = self.outgoing(node)
                nodes_child = node_names[1:]

                if len(nodes_child) == 0:
                    return [[node.id]]

                if len(outgoing_nodes) > 0:
                    for child in outgoing_nodes:
                        marked_nodes = _dfs_helper(child,nodes_child)
                        if marked_nodes is not None:
                            for marked_node in marked_nodes:
                                remove_nodes.append([node.id] + marked_node)
                    return remove_nodes
                else:
                    return None
            else:
                return None

        nodes_path_to_remove = []
        for node in self.nodes.values():
            marked_nodes_path = _dfs_helper(node, pattern_node_names)
            if marked_nodes_path is not None:
                nodes_path_to_remove = nodes_path_to_remove + marked_nodes_path
                
        return nodes_path_to_remove
    
    def _find_disconnected_nodes(self):
        
        visited_connected = set()
        def _dfs_helper(node):

            outgoing_nodes = self.outgoing(node)
            if len(outgoing_nodes)==0: 
                return False
            if node.id in self.output_nodes:
                return True
            if node.id in visited_connected:
                return True
            
            connected = False
            for child in outgoing_nodes:
                connected = (connected or _dfs_helper(child))
                if connected:
                    visited_connected.add(node.id)
            return connected

        disconnected_nodes = []
        for node in self.nodes.values():
            connected = _dfs_helper(node)
            if not connected:
                disconnected_nodes.append(node.id)
                
        return disconnected_nodes
                      
    def replace_eligible_matmul_as_linear(self):
        # First pass get all eligible nodes
        matmul_nodes = list()
        for node in self.nodes.values():
            if node.op_name != 'matmul':
                continue
            do_convert = False
            transpose_weight_node = None
            add_bias_node = None
            for node_in in self.incoming(node):
                if node_in.op_name == 'transpose' and len(self.incoming(node_in)) == 0:
                    do_convert = True
                    transpose_weight_node = node_in
                    for node_out in self.outgoing(node):
                        if node_out.op_name == 'add' and len(self.incoming(node_out)) == 1:
                            add_bias_node = node_out
            if do_convert:
                matmul_nodes.append(
                    {
                        'matmul' : node,
                        'transpose_weight' : transpose_weight_node,
                        'add_bias' : add_bias_node
                    }
                )
        
        removed_add_bias_nodes = set()
        for node_dict in matmul_nodes:
            matmul_node = node_dict['matmul']
            transpose_weight_node = node_dict['transpose_weight']
            add_bias_node = node_dict['add_bias']
            
            # Reformulate matmul node as linear node 
            matmul_node.op_name = 'linear'
            matmul_node.op = transpose_weight_node.op
            matmul_node.param_names = transpose_weight_node.param_names
            self.remove(transpose_weight_node)
            for node_group in self.op_name_to_node_group_comp_op.values():
                if node_group.contain_node(transpose_weight_node):
                    node_group.remove_node(transpose_weight_node)
                    node_group.add_node(matmul_node)
            
            # Merge add bias node into linear node        
            if add_bias_node is not None:
                matmul_node.param_names.extend(add_bias_node.param_names)
                for node_out in self.outgoing(add_bias_node):
                    self.add_edge(matmul_node, node_out)
                self.remove(add_bias_node)
                removed_add_bias_nodes.add(add_bias_node)
                for node_group in self.op_name_to_node_group_comp_op.values():
                    if node_group.contain_node(add_bias_node):
                        node_group.remove_node(add_bias_node)
        
    def remove_isolated_nodes(self):
        """Remove nodes that does not have incoming nodes and no params"""
        def all_nodes_have_incoming(graph):
            result = True
            for node in graph.nodes.values():
                if node.id == 'dummy_input':
                    continue
                if len(graph.incoming(node)) == 0 and len(node.param_names) == 0:
                    result = False
            return result

        while not all_nodes_have_incoming(self):
            nodes_no_incoming = list()            
            for node in self.nodes.values():
                if node.id == 'dummy_input':
                    continue
                if len(self.incoming(node)) == 0 and len(node.param_names) == 0:
                    nodes_no_incoming.append(node)
            self.remove(nodes_no_incoming)
    
    def add_node(self, node):
        node_id = self.id(node)
        self.nodes[node_id] = node

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

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self.nodes.get(k) for k in key]
        else:
            return self.nodes.get(key)

    def add_edge(self, node1, node2, label=None):
        # If the edge is already present, don't add it again.
        # TODO: If an edge exists with a different label, still don't add it again.
        edge = (self.id(node1), self.id(node2), label)
        if edge not in self.edges:
            self.edges.append(edge)

    def remove(self, nodes):
        """Remove a node and its edges."""
        nodes = nodes if isinstance(nodes, list) else [nodes]
        for node in nodes:
            k = self.id(node)
            self.edges = list(filter(lambda e: e[0] != k and e[1] != k, self.edges))
            del self.nodes[k]
            
    def _get_trace_graph(self, model, dummy_input, optimized_onnx=False):
        # Run the Pytorch graph to get a trace and generate a graph from it
        trace_graph = None
        with torch.no_grad():
            trace_graph, _ = torch.jit._get_trace_graph(model, dummy_input)
            
        if not optimized_onnx:
            # Optimize trace graph based on _optimize_graph of https://github.com/pytorch/pytorch/blob/main/torch/onnx/utils.py
            # Inline everything
            _C._jit_pass_inline(trace_graph)

            # Remove fork/wait nodes
            _C._jit_pass_inline_fork_wait(trace_graph)
            _C._jit_pass_lint(trace_graph)

            _C._jit_pass_onnx_autograd_function_process(trace_graph)
            _C._jit_pass_lower_all_tuples(trace_graph)
        
            # run dce to eliminate dead parts of the graph that might have been
            # left behind by things like symbolic_override
            _C._jit_pass_dce(trace_graph)
            _C._jit_pass_lint(trace_graph)

            # CSE should improve perf when Autocast is used with disabled cache
            # Autocast is disabled due to a limitation on tracer as described at https://github.com/pytorch/pytorch/issues/84092
            # Must run before _C._jit_pass_erase_number_types to prevent type substitution
            if _C._jit_pass_cse(trace_graph):
                _C._jit_pass_onnx_lint(trace_graph)

            _C._jit_pass_canonicalize_graph_fuser_ops(trace_graph)
            _C._jit_pass_lint(trace_graph)
            _C._jit_pass_peephole(trace_graph, True)
            _C._jit_pass_fuse_addmm(trace_graph)
            _C._jit_pass_lint(trace_graph)

            _C._jit_pass_peephole(trace_graph, True)
            _C._jit_pass_lower_all_tuples(trace_graph)
            
            trace_graph = _C._jit_pass_canonicalize(trace_graph)
        else:
            if LooseVersion(torch.__version__) >= LooseVersion('1.9.0') and \
            LooseVersion(torch.__version__) <= LooseVersion('1.11.10'):
                trace_graph = torch.onnx._optimize_trace(trace_graph, torch.onnx.OperatorExportTypes.ONNX)
            elif LooseVersion(torch.__version__) >= LooseVersion('1.13.0'):
                trace_graph = torch.onnx._optimize_graph(trace_graph, torch.onnx.OperatorExportTypes.ONNX)
            else:
                raise "Torch {} is not supported because of some bug in _optimize_trace.".format(torch.__version__)
        return trace_graph

    def _get_module_type(self, module):
        return type(module).__name__

    def _parse_modules(self, model):
        model_param_names = set()
        for name, _ in model.named_parameters():
            model_param_names.add(name)
        
        # Find the root module
        for m in model.modules():
            module_param_names = set([name for name, _ in m.named_parameters()])
            if module_param_names == model_param_names:
                self.root_module = m
                break

        self.basic_ops = dict()
        self.composed_ops = dict()
             
        def find_compose_op_dfs_helper(module, module_name, composed_op):
            module_type = self._get_module_type(module)
            if module_type in COMPOSED_MODULES:
                composed_op = COMPOSED_MODULES[module_type](
                    id = module_name,
                    _type = module_type,
                    module = module)
                self.composed_ops[composed_op.id] = composed_op
                return 
            
            for name, module_child in module.named_children():
                find_compose_op_dfs_helper(module_child, module_name + '.' + name if module_name != '' else name, composed_op)

        find_compose_op_dfs_helper(self.root_module, "", None)

        def find_basic_op_dfs_helper(module, module_name, basic_op):
            module_type = self._get_module_type(module)
            if module_type in COMPOSED_MODULES:
                return
            if module_type in BASIC_MODULES:
                basic_op = BASIC_MODULES[module_type](
                    id = module_name,
                    _type = module_type,
                    module = module)
                self.basic_ops[basic_op.id] = basic_op
                return 
            
            for name, module_child in module.named_children():
                find_basic_op_dfs_helper(module_child, module_name + '.' + name if module_name != '' else name, basic_op)
            
        find_basic_op_dfs_helper(self.root_module, "", None)   
        
        self.param_name_to_operator = dict()
        self.op_name_to_node_group_comp_op = dict()
        for op_name in self.composed_ops:
            compose_op = self.composed_ops[op_name]
            self.op_name_to_node_group_comp_op[op_name] = NodeGroupComposedOp(op=compose_op)
            for p_name, _ in compose_op.named_parameters():
                self.param_name_to_operator[op_name + '.' + p_name] = compose_op

        for op_name in self.basic_ops:
            basic_op = self.basic_ops[op_name]
            for p_name, _ in basic_op.named_parameters():
                self.param_name_to_operator[op_name + '.' + p_name] = basic_op

    def id(self, node):
        """Returns a unique node identifier. If the node has an id
        attribute (preferred), it's used. Otherwise, the hash() is returned."""
        return node.id if hasattr(node, "id") else hash(node)

    def torch_node_id(self, node):
        """Returns a unique ID for a node."""
        return "node-" + "-".join(["{}".format(o.unique()) for o in node.outputs()])

    def _parse_tensors_info(self, state_dict, torch_graph_str):
        """Use hack to parse tensor info, should be better option for doing it"""
        prefix_str = "graph"
        assert torch_graph_str.startswith(prefix_str), "Invalid graph str to be parsed"

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

        tensors_str = get_str_inside_parenthesis(torch_graph_str, prefix_str = prefix_str)
        tensors_str_list = [s.strip() for s in tensors_str.split('%')][1:]

        self.param_id_to_name = dict()

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
                self.inputs['node-' + str(i)] = (i, tensor_id, tensor_str_split)
                cur_input += 1
            elif tensor_type == "params":
                self.param_id_to_name[int(tensor_id)] = self.param_names[cur_param]
                cur_param += 1        
    
    def build_dot(self, vertical=False, by_node_groups=True):
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
        if len(self.node_groups) == 0 or not by_node_groups:
            for node in self.nodes.values():
                if node.id == "dummy_input":
                    dot.attr("node", shape="ellipse", 
                            style="filled", margin="0,0",
                            fillcolor=self.theme["fill_color"],
                            color=self.theme["outline_color"],
                            fontsize=self.theme["font_size"],
                            fontname=self.theme["font_name"])
                    label = "<tr><td cellpadding='6'>{}</td></tr>".format(node.id)
                    label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"
                    dot.node(str(node.id), label)
                elif node.id == "dummy_output":
                    dot.attr("node", shape="doubleoctagon", 
                            style="filled", margin="0,0",
                            fillcolor=self.theme["fill_color"],
                            color=self.theme["outline_color"],
                            fontsize=self.theme["font_size"],
                            fontname=self.theme["font_name"])
                    label = "<tr><td cellpadding='6'>{}</td></tr>".format(node.id)
                    label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"
                    dot.node(str(node.id), label)
                else:
                    dot.attr("node", shape="box", 
                            style="filled", margin="0,0",
                            fillcolor=self.theme["fill_color"],
                            color=self.theme["outline_color"],
                            fontsize=self.theme["font_size"],
                            fontcolor=self.theme["font_color"],
                            fontname=self.theme["font_name"])
                    label = "<tr><td cellpadding='6'>{}</td></tr>".format(node.title)
                    if node.id:
                        label += "<tr><td>{}</td></tr>".format(node.id)
                    label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"
                    dot.node(str(node.id), label)
        else:
            node_colors = dict()
            for node in self.nodes.values():
                node_colors[node.id] = list()

            nodes_in_prunable_node_groups = set()
            nodes_in_auxiliary_node_groups = set()
            for node_group in self.node_groups.values():
                random_number = random.randint(0,16777215)
                hex_number = str(hex(random_number))
                color ='#'+ hex_number[2:]
                is_prunable = node_group.is_prunable
                is_auxiliary = node_group.is_auxiliary
                for node in node_group:
                    node_colors[node.id].append(color)
                    if is_prunable:
                        nodes_in_prunable_node_groups.add(node.id)
                    if is_auxiliary:
                        nodes_in_auxiliary_node_groups.add(node.id)
                        
            for node_id in node_colors:
                if len(node_colors[node_id]) == 0:
                    node_colors[node_id] = self.theme["fill_color"]

            for node in self.nodes.values():
                if node.id == "dummy_input":
                    dot.attr("node", shape="ellipse", 
                            style="filled", margin="0,0",
                            fillcolor=self.theme["fill_color"],
                            color=self.theme["outline_color"],
                            fontsize=self.theme["font_size"],
                            fontcolor=self.theme["font_color"],
                            fontname=self.theme["font_name"])
                    label = "<tr><td cellpadding='6'>{}</td></tr>".format(node.id)
                    label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"
                    dot.node(str(node.id), label)
                elif node.id == "dummy_output":
                    dot.attr("node", shape="doubleoctagon", 
                            style="filled", margin="0,0",
                            fillcolor=self.theme["fill_color"],
                            color=self.theme["outline_color"],
                            fontsize=self.theme["font_size"],
                            fontcolor=self.theme["font_color"],
                            fontname=self.theme["font_name"])
                    label = "<tr><td cellpadding='6'>{}</td></tr>".format(node.id)
                    label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"
                    dot.node(str(node.id), label)
                else:
                    color = ":".join(node_colors[node.id])
                    if len(node.param_names) == 0:
                        dot.attr("node", shape="box" if node.id not in nodes_in_auxiliary_node_groups else "ellipse",
                                style="filled" if node.id in nodes_in_prunable_node_groups else "dashed", 
                                margin="0,0",
                                fillcolor=color,
                                color=color if node.id not in nodes_in_prunable_node_groups else self.theme["outline_color"],
                                fontsize=self.theme["font_size"],
                                fontcolor=color if node.id not in nodes_in_prunable_node_groups else "#FFFFFF",
                                fontname=self.theme["font_name"])
                    elif len(node.param_names) > 0:
                        dot.attr("node", shape="box" if node.id not in nodes_in_auxiliary_node_groups else "ellipse", 
                                style="filled" if node.id in nodes_in_prunable_node_groups else "dashed", 
                                margin="0,0",
                                fillcolor=color,
                                color=color if node.id not in nodes_in_prunable_node_groups else self.theme["outline_color"],
                                fontsize=self.theme["font_size"],
                                fontcolor=self.theme["font_color"],
                                fontname=self.theme["font_name"])                        

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

    def visited_dict(self):
        visited = dict()
        for node in self.nodes.values():
            visited[node.id] = False
        return visited

    def random_set_zero_groups(self, target_group_sparsity=None, num_group_divisible=2):
        print("Random set zero groups")
        for node_group in self.node_groups.values():
            if not node_group.is_prunable and not node_group.is_auxiliary:
                continue
            
            num_groups = node_group.get_num_groups()
            param_groups = node_group.get_param_groups()
            
            assert target_group_sparsity is None or (target_group_sparsity >= 0 and target_group_sparsity < 1.0)
            curr_group_sparsity = np.random.random() if target_group_sparsity is None else target_group_sparsity
            num_zero_groups = max(min(int(curr_group_sparsity * num_groups) // num_group_divisible * num_group_divisible, num_groups - 1), 0)
            zero_group_idxes = np.random.choice(list(range(0, num_groups - 1)), num_zero_groups, replace=False)
            
            if len(param_groups['params']) == 0:
                continue   
            
            for (p_name, param, p_transform) in zip(param_groups['p_names'], param_groups['params'], param_groups['p_transform']):
                # Skip lora_A which is unprunable if any
                if 'lora_A' in p_name:
                    continue
                if p_transform == TensorTransform.TRANSPOSE and len(param.data.shape) > 1:
                    param.data[:, zero_group_idxes, ...] = 0.0
                elif p_transform == TensorTransform.MULTIHEAD:
                    multi_head_zero_group_idxes = zero_group_idxes.tolist()
                    for h in range(1, param_groups['num_heads']):
                        multi_head_zero_group_idxes.extend([i + param_groups['head_dim'] * h for i in zero_group_idxes.tolist()])
                    param.data[multi_head_zero_group_idxes] = 0.0
                else:
                    param.data[zero_group_idxes] = 0.0
                    
            for ng_id, offset in param_groups['auxiliary_ngs']:
                aux_pg = self.node_groups[ng_id].get_param_groups()
                for aux_p in aux_pg['params']:
                    if aux_p.grad is None:
                        continue
                    aux_p.data[offset+zero_group_idxes, ...] = 0.0

    def set_pruning_redundant_idxes(self):
        for node_group in self.node_groups.values():
            if node_group.is_prunable and not node_group.is_auxiliary:
                node_group.set_pruning_redundant_idxes()
        for node_group in self.node_groups.values():
            if node_group.is_auxiliary:
                node_group.set_pruning_redundant_idxes()
        
    def skip_operators(self, operators=list()):
        '''
        Make the node groups contains target operator unprunable
        '''
        for node_group in self.node_groups.values():
            if len(node_group.param_names) == 0 or not node_group.is_prunable:
                continue
            if type(node_group).__name__ == 'NodeGroupComposedOp':
                if node_group.op._type in operators:
                    node_group.is_prunable = False
            elif type(node_group).__name__ == 'NodeGroup':
                for node in node_group:
                    if len(node.param_names) == 0 or not node.op:
                        continue
                    if node.op._type in operators:
                        node_group.is_prunable = False
                        break
    
    def set_trainable(self):
        self.set_param_grad_no_grad(self._model)
        for node_group in self.node_groups.values():
            node_group.is_trainable = False
            if len(node_group.param_names) == 0:
                node_group.is_tranable = False
                node_group.is_prunable = False
                continue
            
            all_param_no_grad = True
            for param_name in node_group.param_names:
                if param_name in self.params_grad:
                    node_group.is_trainable = True
                    all_param_no_grad = False
                    break
            if all_param_no_grad:
                node_group.is_prunable = False
    
    def set_param_grad_no_grad(self, model):
        self.params_grad = dict()
        self.params_no_grad = dict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params_grad[name] = param
            else:
                self.params_no_grad[name] = param

        for name in model.state_dict():
            self.param_names.append(name)
            if name not in self.params_grad:
                self.params_no_grad[name] = model.state_dict()[name]
    
    def get_param_groups(self):
        param_groups = dict()
        for node_group in self.node_groups.values():
            if node_group.is_trainable:
                ng_param_group = node_group.get_param_groups()
                if len(ng_param_group['params']) > 0:
                    param_groups[node_group.id] = ng_param_group
        # Second pass for tackling auxliary node groups
        for node_group in self.node_groups.values():
            if node_group.is_auxiliary and node_group.is_trainable:
                offset = 0
                for depend_ng in node_group.dependent_node_groups:
                    depend_ng_pg = param_groups[depend_ng.id]
                    depend_ng_pg['auxiliary_ngs'].append((node_group.id, offset))
                    offset += depend_ng.num_groups
        
        untrainable_param_group_ids = set()
        for param_group in param_groups.values():
            if len(param_group['auxiliary_ngs']) > 0:
                continue
            all_params_no_req_grad = True
            for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
                if param.requires_grad:
                    all_params_no_req_grad = False
            if all_params_no_req_grad:
                untrainable_param_group_ids.add(param_group['id'])
        
        for remove_id in untrainable_param_group_ids:
            del param_groups[remove_id]

        param_groups = dict(sorted(param_groups.items(), key=lambda kv:(kv[0], kv[1])))
        return param_groups.values()
        
    def cluster_node_groups(self, num_clusters=1):
        if num_clusters == 1:
            self.node_group_clusters = dict()
            self.node_group_clusters[0] = list()
            for node_group in self.node_groups.values():
                if not node_group.is_prunable or not node_group.is_trainable:
                    continue
                self.node_group_clusters[0].append(node_group)
        else:
            from sklearn.cluster import KMeans
            
            node_group_ids = []
            node_group_sizes = []
            for node_group in self.node_groups.values():
                if not node_group.is_prunable or not node_group.is_trainable:
                    continue
                node_group_ids.append(node_group.id)
                node_group_sizes.append([node_group.get_num_groups(), 1.0])
            node_group_sizes = np.array(node_group_sizes)

            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(node_group_sizes)
            
            self.node_group_clusters = dict()
            for node_group_id, node_group_cluster_id in zip(node_group_ids, kmeans.labels_.tolist()):
                if node_group_cluster_id not in self.node_group_clusters:
                    self.node_group_clusters[node_group_cluster_id] = list()
                node_group = self.node_groups[node_group_id]
                self.node_group_clusters[node_group_cluster_id].append(node_group)