import imp
import re
import copy

from . import ge
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# from graph.node import Node

class Rename():
    def __init__(self, op=None, name=None, to=None):
        assert op or name, "Either op or name must be provided"
        assert not(op and name), "Either op or name should be provided, but not both"
        assert bool(to), "The to parameter is required" 
        self.to = to
        self.op = re.compile(op) if op else None
        self.name = re.compile(name) if name else None
    
    def apply(self, graph):
        for i, node in enumerate(graph.nodes.values()):
            if self.op:
                node.op.name = self.op.sub(self.to, node.op.name)
            if self.name is None:
                node.name = str(node.op.name)
            else:
                node.name = self.name.sub(self.to, node.name)


# class Fold():
#     def __init__(self, pattern, to, name=None):
#         # TODO: validate that op and name are valid
#         self.pattern = ge.GEParser(pattern).parse()
#         self.to = to
#         self.name = name

#     def apply(self, graph):     
#         while True:
#             matches, _ = graph.search(self.pattern)
#             if not matches:
#                 break

#             # Replace pattern with new node
#             if self.to == "__first__":
#                 combo = matches[0]
#             elif self.to == "__last__":
#                 combo = matches[-1]
#             else:
#                 # find the most bottom child
#                 outputs = set()
#                 match_ids = [node.id for node in matches]
#                 for match_node in matches:
#                     for outgoing_node in graph.outgoing(match_node):
#                         if outgoing_node.id not in match_ids:
#                             outputs.add(outgoing_node)
#                 # combine operators
#                 combo_op = matches[0].op
#                 for i in range(1, len(matches)):
#                     combo_op += matches[i].op
#                 combo_op.name = self.to or self.pattern
#                 combo = Node(id=graph.sequence_id(),
#                              op=combo_op,
#                              output_shape=matches[-1].output_shape,
#                              outputs = list(outputs)) # TODO, check bugs
#                 combo._caption = "/".join(filter(None, [l.caption for l in matches]))
#             graph.replace(matches, combo)


class ConvBNFuse():
    def __init__(self, pattern, to, name=None):
        self.pattern = ge.GEParser(pattern).parse()
        self.to = to
        self.name = name

    def apply(self, graph):     
        graph.fused_conv_bns = list()
        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break
            for match_node in matches:
                match_node._skip_pattern_search = True
            graph.fused_conv_bns.append(matches)