import imp
from .graph import Graph
from .zig.zig import automated_partition_zigs
from .optimizer import DHSPG
from .compression.compression import automated_compression
import os
from .flops.flops import compute_flops

class OTO:
    def __init__(self, model=None, dummy_input=None):
        self._graph = None
        self._model = model
        self._dummy_input = dummy_input

        if self._model is not None and self._dummy_input is not None:
            self.initialize(model=self._model, dummy_input=self._dummy_input)
            self.partition_zigs()
        self.compressed_model_path = None
        self.full_group_sparse_model_path = None

    def initialize(self, model=None, dummy_input=None):
        model = model.eval()
        self._model = model
        self._dummy_input = dummy_input
        self._graph = Graph(model, dummy_input)

    def partition_zigs(self):
        self._graph = automated_partition_zigs(self._graph)
    
    def visualize_zigs(self, out_dir=None, view=False, vertical=True):
        self._graph.build_dot(verbose=True, vertical=vertical).render(\
            os.path.join(out_dir if out_dir is not None else './', \
                self._model.name if hasattr(self._model, 'name') else type(self._model).__name__ + '_zig.gv'), \
                view=view)

    def dhspg(self, lr=0.1, lmbda=1e-3, lmbda_amplify=1.1, hat_lmbda_coeff=10, epsilon=0.0, weight_decay=0.0, first_momentum=0.0, second_momentum=0.0, \
               variant='sgd', target_group_sparsity=0.5, tolerance_group_sparsity=0.05, partition_step=None, start_pruning_steps=0, half_space_project_steps=None,\
               warm_up_steps=0, dampening=0.0, group_divisible=1, fixed_zero_groups=True):
        self._optimizer = DHSPG(
            params=self._graph.params_groups(epsilon=epsilon),
            lr=lr,
            lmbda=lmbda,
            lmbda_amplify=lmbda_amplify,
            hat_lmbda_coeff=hat_lmbda_coeff,
            epsilon=epsilon,
            weight_decay=weight_decay,
            first_momentum=first_momentum,
            second_momentum=second_momentum,
            dampening=dampening,
            variant=variant,
            target_group_sparsity=target_group_sparsity, 
            tolerance_group_sparsity=tolerance_group_sparsity,
            partition_step=partition_step,
            warm_up_steps=warm_up_steps,
            start_pruning_steps=start_pruning_steps,
            half_space_project_steps=half_space_project_steps,
            group_divisible=group_divisible,
            fixed_zero_groups=fixed_zero_groups)
        return self._optimizer

    def compress(self, compressed_model_path=None, dummy_input=None, dynamic_axes=[False, dict()]):
        _, self.compressed_model_path, self.full_model_path = automated_compression(
            oto_graph=self._graph,
            model=self._model,
            dummy_input=self._dummy_input if dummy_input is None else dummy_input,
            compressed_model_path=compressed_model_path,
            dynamic_axes=dynamic_axes)
    
    def random_set_zero_groups(self):
        self._graph.random_set_zero_groups()
    
    def compute_flops(self, compressed=False, verbose=False):
        flops_info = compute_flops(self._graph, compressed=compressed)
        return flops_info['total'] if not verbose else flops_info
        
    def compute_num_params(self, compressed=False):
        import onnx
        import numpy as np
        if compressed:
            if self.compressed_model_path is None:
                raise "Compressed model does not exist, please compress at first."            
            onnx_model = onnx.load(self.compressed_model_path)
            onnx_graph = onnx_model.graph
            num_params = 0
            for tensor in onnx_graph.initializer:
                num_params += np.prod(tensor.dims)
            return num_params
        else:
            return sum([w.numel() for _, w in self._model.named_parameters()])