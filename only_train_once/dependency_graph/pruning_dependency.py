import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from graph.node_group import NodeGroup
from operation.operator import UNPRUNABLE_BASIC_OPERATORS, UNPRUNABLE_COMPOSED_OPERATORS
from transform import is_spread_transformation, TensorTransform, SPREAD_TRANSFORM_MAP

def get_non_stem_nodes(graph, skip_node_ids=set()):
    non_stem_nodes = list()
    for node in graph.nodes.values():
        if node.id in skip_node_ids:
            continue
        if not node.is_stem() and not node.is_concat(axis=1) and not node.is_dummy(): # operator concat is left for next version
            non_stem_nodes.append(node)
    return non_stem_nodes

def get_non_stem_node_groups(graph, nodes):
    node_groups = []
    visited = dict()
    for node in nodes:
        visited[node.id] = False

    def dfs_helper(graph, node, cc):
        visited[node.id] = True
        cc.add_node(node)
        for node_out in graph.outgoing(node):
            if node_out.id in visited:
                if not visited[node_out.id]:
                    dfs_helper(graph, node_out, cc)
        for node_in in graph.incoming(node):
            if node_in.id in visited:
                if not visited[node_in.id]:
                    dfs_helper(graph, node_in, cc)

    for node in nodes:
        if not visited[node.id]:
            connected_component = NodeGroup()
            dfs_helper(graph, node, connected_component)
            node_groups.append(connected_component)
    return node_groups

def grow_non_stem_node_groups(graph, node_groups, skip_node_ids=set()):
    for node_group in node_groups:
        grow_non_stem_node_group(graph, node_group, skip_node_ids)
    return node_groups

def grow_non_stem_node_group(graph, node_group, skip_node_ids=set()):
    visited = {}
    for node_id in graph.nodes:
        visited[node_id] = False if node_id not in skip_node_ids else True

    new_nodes = list()
    def dfs_helper(graph, node):
        if (node.is_stem() or node.is_concat(axis=1)) and not node.is_dummy():
            new_nodes.append(node)
            return 
        visited[node.id] = True
        for node_in in graph.incoming(node):
            if not visited[node_in.id]:
                dfs_helper(graph, node_in)

    for node in node_group:
        if not visited[node.id]:
            dfs_helper(graph, node)

    node_group.add_nodes(new_nodes)

def merge_node_groups(node_groups):
    # Merge node groups if they share intersected nodes
    pool = set(node_groups)
    merged_node_groups = []
    while pool:
        merged_node_groups.append(pool.pop())
        while True:
            for cc in pool:
                if merged_node_groups[-1].nodes.keys() & cc.nodes.keys():
                    merged_node_groups[-1].merge(cc)
                    pool.remove(cc)
                    break
            else:
                break
    return merged_node_groups

def get_remaining_nodes(node_groups, all_nodes):
    remaining_nodes = []
    included_nodes = []
    for node_group in node_groups:
        if type(node_group).__name__ == 'NodeGroupComposedOp':
            included_nodes.extend([node.id for node in node_group.nodes.values() \
                               if node.id not in node_group.output_nodes])
        else:
            included_nodes.extend([node.id for node in node_group.nodes.values()])            
    remaining_nodes = [all_nodes[node_id] for node_id in all_nodes if node_id not in included_nodes \
                       and (node_id != 'dummy_input' and node_id != 'dummy_output')]
    return remaining_nodes
    
def group_individual_nodes(individual_nodes):
    singleton_node_groups = list()
    for node in individual_nodes:
        node_group = NodeGroup()
        node_group.add_node(node)
        singleton_node_groups.append(node_group)
    return singleton_node_groups

def group_nodes_composed_operator(graph):
    adj_nodes = set()
    # Find all paths between two vertices is NP-hard
    def dfs_helper(graph, cur_node, dst_ids, visited, path, verbose=False):
        nonlocal adj_nodes
        visited[cur_node.id]= True
        path.append(cur_node)

        if cur_node.id in dst_ids:
            adj_nodes = adj_nodes.union(set(path))
            return True
        for node_in in graph.incoming(cur_node):
            if not visited[node_in.id]:
                dfs_helper(graph, node_in, dst_ids, visited, path, verbose)
        path.pop()
        return False if len(adj_nodes) == 0 else True
    
    node_groups = list()
    old_node_group_ids = list()
    for node_group_id in graph.node_groups:
        node_group = graph.node_groups[node_group_id]
        if node_group.num_nodes() > 1:
            adj_nodes = set()
            for node in node_group.nodes.values():
                dst_ids = set(node_group.nodes.keys()).difference(set([node.id]))
                dfs_helper(graph, node, dst_ids, graph.visited_dict(), list())
            node_group.add_nodes(adj_nodes)
            node_groups.append(node_group)
        old_node_group_ids.append(node_group_id)

    for old_node_group_id in old_node_group_ids:
        del graph.node_groups[old_node_group_id]
    
    for node_group in node_groups:
        node_group.set_node_equivalence()
        node_group.set_output_nodes(graph)
    return node_groups

def set_auxiliary_node_groups(graph):
    # tackle auxiliary node groups
    visited = dict()
    for node_group in graph.node_groups.values():
        if node_group.set_auxiliary():
            visited[node_group.id] = False
    
    def dfs_helper(graph, node_group, dependent_node_groups):
        if not node_group.is_auxiliary:
            if node_group.contain_stem_op():
                dependent_node_groups.append(node_group)
            return 
        elif visited[node_group.id]:
            if hasattr(node_group, 'dependent_node_groups'):
                dependent_node_groups.extend(node_group.dependent_node_groups)
            return 
        
        concat_nodes = node_group.get_concat_nodes()
        if len(concat_nodes) == 0:
            return

        concat_node = concat_nodes[0]
        for node_in in graph.incoming(concat_node):
            if node_in.id in ['dummy_input']:
                continue
            node_group_in = graph.node_groups[node_in.node_group_ids[0]]
            if node_group_in.id != node_group.id:
                dfs_helper(graph, node_group_in, dependent_node_groups)

    for node_group in graph.node_groups.values():
        if node_group.is_auxiliary:
            if visited[node_group.id]:
                continue
            node_group.dependent_node_groups = list()
            dfs_helper(graph, node_group, node_group.dependent_node_groups)

    # Tackle connection between stem node group and auxiliary node groups
    for node_group in graph.node_groups.values():
        if node_group.is_auxiliary:
            if len(node_group.dependent_node_groups) == 0:
                node_group.is_auxiliary = False
                continue
            if len(node_group.param_names) == 0:
                continue
            offset = 0
            for depend_node_group in node_group.dependent_node_groups:
                if not hasattr(depend_node_group, 'auxilary_node_groups'):
                    depend_node_group.auxilary_node_groups = list()
                depend_node_group.auxilary_node_groups.append((node_group, offset))
                offset += depend_node_group.num_groups

def merge_depth_conv_node_groups(graph):
    visited = dict()
    for node in graph.nodes.values():
        visited[node.id] = False

    def dfs_helper(node, groups, node_groups_to_merge):
        # print("line 203", node.id)
        if node.is_conv():
            if node.op.module.groups == 1 and node.op.module.out_channels == groups:
                node_groups_to_merge.append(graph.node_groups[node.node_group_ids[0]])
                return node_groups_to_merge
        for node_in in graph.incoming(node):
            if not visited[node_in.id]:
                visited[node_in.id] = True
                dfs_helper(node_in, groups, node_groups_to_merge)
        return node_groups_to_merge

    for node in graph.nodes.values():
        if node.is_conv():
            if hasattr(node.op.module, 'groups'):
                if node.op.module.groups == node.op.module.in_channels:
                    for node_id in graph.nodes:
                        visited[node_id] = False
                    node_groups_to_merge = dfs_helper(node, node.op.module.groups, [graph.node_groups[node.node_group_ids[0]]])
                    # If we have node groups to merge
                    if len(node_groups_to_merge) > 1:
                        dummy_node_group_id = node_groups_to_merge[0].id
                        node_groups_to_merge[0].merge(node_groups_to_merge[1])
                        del graph.node_groups[dummy_node_group_id]
                        del graph.node_groups[node_groups_to_merge[1].id]
                        graph.node_groups[node_groups_to_merge[0].id] = node_groups_to_merge[0]
                        for node in node_groups_to_merge[0]:
                            node.node_group_ids[0] = node_groups_to_merge[0].id

def merge_basic_composed_node_groups(graph):
    composed_node_groups = dict()
    for node_group in graph.node_groups.values():
        if type(node_group).__name__ == 'NodeGroupComposedOp':
            composed_node_groups[node_group.id] = node_group
    
    new_composed_node_groups = list()
    merged_node_group_ids = list()
    for node_group in graph.node_groups.values():
        if type(node_group).__name__ == 'NodeGroup':
            for composed_node_group_id in composed_node_groups:
                if composed_node_group_id in merged_node_group_ids:
                    continue
                composed_node_group = composed_node_groups[composed_node_group_id]
                if set(node_group.param_names) == set(composed_node_group.param_names):
                    merged_node_group_ids.append(node_group.id)
                    merged_node_group_ids.append(composed_node_group_id)
                    for node in node_group:
                        if node.id in composed_node_group.nodes:
                            continue
                        composed_node_group.add_node(node)
                    new_composed_node_groups.append(composed_node_group)
                        
    # Remove 
    for node_group_id in merged_node_group_ids:
        del graph.node_groups[node_group_id]
    
    # Add
    for node_group in new_composed_node_groups:
        graph.node_groups[node_group.id] = node_group
    
    for node in graph.nodes.values():
        node.node_group_ids = list()
    
    for node_group in graph.node_groups.values():
        for node in node_group:
            node.node_group_ids.append(node_group.id)
    
def build_pruning_dependency_graph(graph):    
    # Step 0: Construct connected components for composed operator.
    node_groups_composed_op = group_nodes_composed_operator(graph)
    skip_node_ids = set()
    for node_group in node_groups_composed_op:
        skip_node_ids |= node_group.get_node_ids(skip_output_node=True)

    # Step 1: Get non-stem nodes with shape dependent
    non_stem_nodes = get_non_stem_nodes(graph, skip_node_ids=skip_node_ids) 

    # Step 2: Find the connected components over non-stem nodes
    non_stem_node_groups = get_non_stem_node_groups(graph, non_stem_nodes)

    # Step 3: Grow the connected components till all incoming nodes are stem nodes 
    # and all outgoing nodes has non-stem nodes but has stem outgoing nodes.
    grown_node_groups = grow_non_stem_node_groups(graph, non_stem_node_groups, skip_node_ids)

    # Step 4: Merge node groups if any intersection
    merged_node_groups = merge_node_groups(grown_node_groups)

    # Step 5: Group the remaining parameters over individual nodes
    remaining_nodes = get_remaining_nodes(merged_node_groups + node_groups_composed_op, graph.nodes)
    singleton_node_groups = group_individual_nodes(remaining_nodes)

    # Step 6: Setup connected components fo graph
    for node_group in merged_node_groups + node_groups_composed_op + singleton_node_groups:
        graph.node_groups[node_group.id] = node_group
        for node in node_group:
            node.node_group_ids.append(node_group.id)

    # Step 7: TODO: Tackle group conv
    '''We conly consider a special case that groups=in_channel, =out_channel of incoming conv'''
    merge_depth_conv_node_groups(graph)
    
    # Step 8: Set auxilary node groups
    set_auxiliary_node_groups(graph)

    # Setp 9: Merge Basic Node Group into Composed Node Group if being a subset
    merge_basic_composed_node_groups(graph)
    
    # Step 10: Set prunable for node_group
    for node_group in graph.node_groups.values():
        # If there is no trainable variables in the node_groups, then un-prunable.
        if len(node_group.param_names) == 0:
            node_group.is_prunable = False
        # If is adjacent to the output, then un-prunable.
        if node_group.contain_some_nodes(graph.output_nodes.values()):
            node_group.is_prunable = False
        # If there is no stem op in the node_group, then un-prunable.
        if not node_group.contain_stem_op() and not node_group.is_auxiliary:
            node_group.is_prunable = False
    
    # Tackle a weight sharing case, if one weight belongs to a unprunable node group, yet also belongs to a prunable node group,
    # the prunable, node group needs to be set unprunable.
    unprunable_param_names = set()
    for node_group in graph.node_groups.values():
        if not node_group.is_prunable:
            unprunable_param_names = unprunable_param_names.union(set(node_group.param_names))
    for node_group in graph.node_groups.values():
        if node_group.is_prunable:
            if set(node_group.param_names) & unprunable_param_names:
                node_group.is_prunable = False
    
    for node_group in graph.node_groups.values():     
        if type(node_group).__name__ == 'NodeGroupComposedOp':
            if type(node_group.op).__name__ in UNPRUNABLE_COMPOSED_OPERATORS:
                node_group.is_prunable = False
        else:
            for node in node_group:
                if node.op_name in UNPRUNABLE_BASIC_OPERATORS:
                    node_group.is_prunable = False

    # If dummy input directly added or mul into a node group, mark it as unprunable. 
    dummy_input_node = graph.nodes['dummy_input']
    for node_out in graph.outgoing(dummy_input_node):
        if node_out.op_name == 'add' or node_out.op_name == 'mul':
            node_group_id = node_out.node_group_ids[0]
            graph.node_groups[node_group_id].is_prunable = False
    
    # Overwrite node number of groups and p_transform if includes spread_transform
    for node_group in graph.node_groups.values():
        overwrite_p_transforms = set()
        overwrite_num_groups = 0
        fixed_node_ids = set()
        for node in node_group:
            if len(node.param_names) == 0 or not node.op:
                continue
            node_param_groups = node.op.get_param_groups(param_names=node.param_names)
            for p_transform in node_param_groups['p_transform']:
                if is_spread_transformation(p_transform):
                    overwrite_p_transforms.add(p_transform)
                    overwrite_num_groups = node_param_groups['num_groups']
                    fixed_node_ids.add(node.id)

        if len(overwrite_p_transforms) == 1:
            overwrite_p_transform = next(iter(overwrite_p_transforms))
            for node in node_group:
                if len(node.param_names) == 0 or not node.op or node.id in fixed_node_ids:
                    continue
                node.op.num_groups = overwrite_num_groups
                node.op.p_transform = SPREAD_TRANSFORM_MAP[overwrite_p_transform]
            node_group.overwrite_p_transform = overwrite_p_transform
        elif len(overwrite_p_transforms) > 1:
            raise NotImplementedError('One node group has two distinct spread_p_transforms.')

    # If one node group is auxiliary, and has group norm with groups > 1
    # We currently mark its dependent node groups as unprunable 
    for node_group in graph.node_groups.values():
        if not node_group.is_auxiliary:
            continue
        fixed_node_ids = set()

        for node in node_group:
            if len(node.param_names) == 0 or not node.op:
                continue
            if type(node.op).__name__ == 'GroupNormOTO':
                if node.op.num_groups > 1:
                    for depend_node_group in node_group.dependent_node_groups:
                        depend_node_group.is_prunable = False