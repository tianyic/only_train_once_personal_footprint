import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from graph.connected_component import ConnectedComponent

def get_non_stem_nodes(graph):
    non_stem_nodes = list()
    for node in graph.nodes.values():
        if not node.is_stem() and not node.is_concat(axis=1): # operator concat is left for next version
            non_stem_nodes.append(node)
    return non_stem_nodes

def get_connected_components(graph, nodes):
    connected_components = []
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
            connected_component = ConnectedComponent()
            dfs_helper(graph, node, connected_component)
            connected_components.append(connected_component)
    return connected_components

def grow_connected_components(graph, connected_components):
    print("grow_non_stem_connected_components")
    for connected_component in connected_components:
        grow_connected_component(graph, connected_component)
    return connected_components

def grow_connected_component(graph, connected_component):
    visited = {}
    for node_id in graph.nodes:
        visited[node_id] = False

    new_nodes = list()
    def dfs_helper(graph, node):
        if node.is_stem() or node.is_concat(axis=1):
            new_nodes.append(node)
            return 
        visited[node.id] = True
        for node_in in graph.incoming(node):
            if not visited[node_in.id]:
                dfs_helper(graph, node_in)

    for node in connected_component.nodes.values():
        if not visited[node.id]:
            dfs_helper(graph, node)

    connected_component.add_nodes(new_nodes)

def merge_connected_components(connected_components):
    # Merge connected components if they share intersected nodes
    pool = set(connected_components)
    merged_ccs = []
    while pool:
        merged_ccs.append(pool.pop())
        while True:
            for cc in pool:
                if merged_ccs[-1].nodes.keys() & cc.nodes.keys():
                    merged_ccs[-1].merge(cc)
                    pool.remove(cc)
                    break
            else:
                break
    return merged_ccs

def get_remaining_nodes(connected_components, all_nodes):
    remaining_nodes = []
    included_nodes = []
    for cc in connected_components:
        included_nodes.extend([node.id for node in cc.nodes.values()])
    remaining_nodes = [all_nodes[node_id] for node_id in all_nodes if node_id not in included_nodes]
    return remaining_nodes
    
def group_individual_nodes(individual_nodes):
    print("group_individual_nodes")
    individual_connected_components = list()
    for node in individual_nodes:
        connected_component = ConnectedComponent()
        connected_component.add_node(node)
        individual_connected_components.append(connected_component)
    return individual_connected_components

def automated_partition_zigs(graph, opt=None):    
    # Step 1: Get non-stem nodes with shape dependent
    non_stem_nodes = get_non_stem_nodes(graph)    

    # Step 2: Find the connected components over non-stem nodes
    non_stem_connected_components = get_connected_components(graph, non_stem_nodes)
    
    # Step 3: Grow the connected components till all incoming nodes are stem nodes 
    # and all outgoing nodes has non-stem nodes but has stem outgoing nodes.
    # TODO: if no stem component anscenter is found, then the cc needs to be dropped off, because it is directly connected to input.
    grown_connected_components = grow_connected_components(graph, non_stem_connected_components)

    # Step 4: Merge connected components if any intersection
    merged_connected_components = merge_connected_components(grown_connected_components)

    # Step 5: Group the remaining parameters over individual nodes
    remaining_nodes = get_remaining_nodes(merged_connected_components, graph.nodes)
    individual_connected_components = group_individual_nodes(remaining_nodes)


    # Step 6: Setup connected components fo graph
    graph.set_connected_components(merged_connected_components + individual_connected_components)

    # Step 7: Set ZIGs type
    graph.set_zigs(opt)

    return graph
