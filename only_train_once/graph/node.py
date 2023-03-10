import numpy as np
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from operation.operator import Operator

class Node():
    """Represents a framework-agnostic neural network layer in a directed graph."""

    def __init__(self, id, op, op_params=None, params=[], inputs=[], outputs=[], output_shape=[]):
        """
        id: unique ID for the layer that doesn't repeat in the computation graph.
        name: Name to display
        op: Framework-agnostic operation .
        """
        self.id = id
        self.connected_components = dict() # the connected component id that it belongs to.
        if isinstance(op, Operator):
            self.op = op
        else:
            self.op = Operator(name=op, params=op_params)
        
        self.inputs = ['out-' + str(i) for i in inputs]
        self.outputs = ['out-' + str(o) for o in outputs]
        self.params = params    
        self.output_shape = output_shape
        self.input_shape = []
        self.op_params = op_params if op_params else {}
        self._skip_pattern_search = False
        
    @property
    def cc_id(self):
        return list(self.connected_components.keys())[0]

    @property
    def title(self):
        # Default
        title = self.op.name
        if "kernel_shape" in self.op_params:
            # Kernel
            kernel = self.op_params["kernel_shape"]
            title += "x".join(map(str, kernel))
        if "stride" in self.op_params:
            stride = self.op_params["stride"]
            if np.unique(stride).size == 1:
                stride = stride[0]
            if stride != 1:
                title += "/s{}".format(str(stride))
        return title

    def __repr__(self):
        args = (self.op, self.id, self.title, self.inputs, self.outputs)
        f = "<Node: op: {}, id: {}, title: {}, inputs: {}, outputs: {}"
        if self.params:
            args += (str(self.params),)
            f += ", params: {:}"
        if self.op_params:
            args += (str(self.op_params),)
            f += ", op_params: {:}"
        if self.output_shape:
            args += (str(self.output_shape),)
            f += ", output_shape: {:}"
        if self.input_shape:
            args += (str(self.input_shape),)
            f += ", input_shape: {:}"
        f += ">"
        return f.format(*args)

    def is_concat(self, axis=None):
        # Check if concat at first
        _is_concat = self.op.name == "Concat" or self.op.name == 'concat'
        if axis == None:
            return _is_concat
        # Check if axis match
        if _is_concat:
            if 'axis' in self.op_params:
                if self.op_params['axis'] == axis:
                    return True
                else:
                    return False
            else:
                return False
        return _is_concat

    def is_conv(self):
        return self.op.name == "Conv" or self.op.name == 'conv'

    def is_linear(self):
        return self.op.name == "Linear" or self.op.name == 'linear' \
            or self.op.name == "Gemm" or self.op.name == "gemm"

    def is_stem(self):
        return self.op.type == "Stem" or self.op.type == "stem"

    def is_zero_invariant(self):
        return self.op.zero_invariant
    
