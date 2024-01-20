import numpy as np
    
class Node:
    def __init__(self, id=None, op_name="", op=None, inputs=[], outputs=[], param_names=[], output_shape=[]):
        super().__init__()
        self.id = id    
        self.op = op
        self.op_name = op_name
        self.inputs = ['node-' + str(i) for i in inputs]
        self.outputs = ['node-' + str(o) for o in outputs]
        self.param_names = param_names
        self.node_group_ids = list()
        self.pruned_status = {
            "out_dim": False,
            "in_dim": False
        }
        self.output_shape = output_shape
        self.input_shape = []

    def __repr__(self) -> str:
        return f"Node id: {self.id}, op_name: {self.op_name}, param_names: {self.param_names}"
    
    @property
    def title(self):
        if not self.op:
            return self.op_name
        # Default
        title = (self.op_name + '-' + self.op._type) if self.op_name != self.op._type else self.op._type
        if "kernel_shape" in self.op.cfg_params:
            # Kernel
            kernel = self.op.cfg_params["kernel_shape"]
            title += "x".join(map(str, kernel))
        if "stride" in self.op.cfg_params:
            stride = self.op.cfg_params["stride"]
            if np.unique(stride).size == 1:
                stride = stride[0]
            if stride != 1:
                title += "/s{}".format(str(stride))
        return title
    
    def is_stem(self):
        if self.op is not None:
            if self.op.is_basic:
                return self.op.is_stem
            else:
                return self.is_conv() or self.is_convtranspose() or self.is_linear()
        else:
            return False

    def is_conv(self):
        return self.op_name == "Conv" or self.op_name == 'conv'

    def is_convtranspose(self):
        return self.op_name == "ConvTranspose" or self.op_name == 'convtranspose'
    
    def is_linear(self):
        return self.op_name == "Linear" or self.op_name == 'linear' \
            or self.op_name == "Gemm" or self.op_name == "gemm"
    
    def is_concat(self, axis=None):
        # Check if concat at first
        _is_concat = self.op_name == "Concat" or self.op_name == 'concat'
        if axis == None:
            return _is_concat
        # Check if axis match
        if _is_concat and hasattr(self.op, 'cfg_params'):
            if 'axis' in self.op.cfg_params:
                return True if self.op.cfg_params['axis'] == axis else False
            else:
                return False
        return _is_concat

    def is_dummy(self):
        return True if self.id == 'dummy_input' or self.id == 'dummy_output' else False

    
