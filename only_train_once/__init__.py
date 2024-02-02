from .graph import Graph
from .dependency_graph import build_pruning_dependency_graph
from .subnet_construction import automated_pruning_compression
import os

class OTO:
    def __init__(self, model=None, dummy_input=None, compress_mode='prune', skip_patterns=None, strict_out_nodes=False):
        self._graph = None
        self._model = model
        self._dummy_input = dummy_input
        self._skip_patterns = skip_patterns
        self._strict_out_nodes = strict_out_nodes
        self._mode = compress_mode

        if self._model is not None and self._dummy_input is not None:
            self.initialize(model=self._model, dummy_input=self._dummy_input, skip_patterns=self._skip_patterns, strict_out_nodes=self._strict_out_nodes)
            if self._mode == 'prune':    
                self.partition_pzigs()
                self.set_trainable()
                self._graph.cluster_node_groups()
            elif self._mode == 'erase':
                # Will be released
                raise NotImplementedError

        self.compressed_model_path = None
        self.full_group_sparse_model_path = None
                    
    def cluster_node_groups(self, num_clusters=1):
        self._graph.cluster_node_groups(num_clusters=num_clusters)
        
    def initialize(self, model=None, dummy_input=None, skip_patterns=None, strict_out_nodes=False):
        model = model.eval()
        self._model = model
        self._dummy_input = dummy_input
        self._graph = Graph(model, dummy_input, skip_patterns=skip_patterns, strict_out_nodes=strict_out_nodes)

    def partition_pzigs(self):
        build_pruning_dependency_graph(self._graph)

    def visualize(self, out_dir=None, view=False, vertical=True, by_node_groups=True, display_params=False):
        self._graph.build_dot(vertical=vertical, by_node_groups=by_node_groups, display_params=display_params).render(\
            os.path.join(out_dir if out_dir is not None else './', \
                self._model.name if hasattr(self._model, 'name') else type(self._model).__name__ + '_pruning_dependency'), \
                view=view)

    def hesso(self, lr=0.1, weight_decay=None, first_momentum=None, second_momentum=None, \
               variant='sgd', target_group_sparsity=0.5, start_pruning_step=0, \
               pruning_steps=1, pruning_periods=1, device='cuda',\
               dampening=None, group_divisible=1, fixed_zero_groups=True, importance_score_criteria='default'):
        from .optimizer import HESSO
        self._optimizer = HESSO(
            params=self._graph.get_param_groups(),
            lr=lr,
            weight_decay=weight_decay,
            first_momentum=first_momentum,
            second_momentum=second_momentum,
            dampening=dampening,
            variant=variant,
            target_group_sparsity=target_group_sparsity, 
            start_pruning_step=start_pruning_step,
            pruning_periods=pruning_periods,
            pruning_steps=pruning_steps,
            group_divisible=group_divisible,
            importance_score_criteria=importance_score_criteria, 
            device=device
        )
        return self._optimizer

    def dhspg(self, lr=0.1, weight_decay=None, first_momentum=None, second_momentum=None, \
               variant='sgd', target_group_sparsity=0.5, tolerance_group_sparsity=0.01, start_pruning_step=0, \
               pruning_steps=1, pruning_periods=1, device='cuda', \
               dampening=None, group_divisible=1, fixed_zero_groups=True, importance_score_criteria='default'):
        from .optimizer import DHSPG
        self._optimizer = DHSPG(
            params=self._graph.get_param_groups(),
            lr=lr,
            weight_decay=weight_decay,
            first_momentum=first_momentum,
            second_momentum=second_momentum,
            dampening=dampening,
            variant=variant,
            target_group_sparsity=target_group_sparsity, 
            tolerance_group_sparsity=tolerance_group_sparsity,
            start_pruning_step=start_pruning_step,
            pruning_periods=pruning_periods,
            pruning_steps=pruning_steps,
            group_divisible=group_divisible,
            fixed_zero_groups=fixed_zero_groups,
            importance_score_criteria=importance_score_criteria, 
            device=device
        )
        return self._optimizer

    def lhspg(self, lr=0.1, epsilon=0.0, weight_decay=None, first_momentum=None, second_momentum=None, \
               variant='sgd', target_group_sparsity=0.5, tolerance_group_sparsity=0.01, start_pruning_step=0, \
               pruning_steps=1, pruning_periods=1, device='cuda', \
               dampening=None, group_divisible=1, fixed_zero_groups=True, lora_update_freq=4, importance_score_criteria=None):
        from .optimizer import LHSPG
        self._optimizer = LHSPG(
            params=self._graph.get_param_groups(),
            lr=lr,
            weight_decay=weight_decay,
            first_momentum=first_momentum,
            second_momentum=second_momentum,
            dampening=dampening,
            variant=variant,
            target_group_sparsity=target_group_sparsity, 
            tolerance_group_sparsity=tolerance_group_sparsity,
            start_pruning_step=start_pruning_step,
            pruning_periods=pruning_periods,
            pruning_steps=pruning_steps,
            group_divisible=group_divisible,
            fixed_zero_groups=fixed_zero_groups,
            importance_score_criteria=importance_score_criteria, 
            device=device,
            lora_update_freq=lora_update_freq
        )
        return self._optimizer
    
    def h2spg(self, **kwargs):
        # Will be released
        raise NotImplementedError

    def skip_operators(self, operator_list=list()):
        self._graph.skip_operators(operator_list)
    
    def set_trainable(self):
        self._graph.set_trainable()

    def construct_subnet(self, merge_lora_to_base=False, unmerge_lora_to_base=False, export_huggingface_format=False, export_float16=False, out_dir='./', \
                 full_group_sparse_model_dir=None, compressed_model_dir=None, save_full_group_sparse_model=True, ckpt_format='torch'):
        full_group_sparse_model_dir = out_dir if full_group_sparse_model_dir is None else full_group_sparse_model_dir
        compressed_model_dir = out_dir if compressed_model_dir is None else compressed_model_dir    
        if self._mode == 'prune':
            self.compressed_model_path, self.full_group_sparse_model_path = automated_pruning_compression(
                oto_graph=self._graph,
                model=self._model,
                merge_lora_to_base=merge_lora_to_base,
                unmerge_lora_to_base=unmerge_lora_to_base,
                export_huggingface_format=export_huggingface_format,
                export_float16=export_float16,
                full_group_sparse_model_dir=full_group_sparse_model_dir,
                compressed_model_dir=compressed_model_dir,
                save_full_group_sparse_model=save_full_group_sparse_model,
                ckpt_format=ckpt_format)    
        elif self._mode == 'erase':
            # Will be released
            raise NotImplementedError
        
    def random_set_zero_groups(self, target_group_sparsity=None):
        self._graph.random_set_zero_groups(target_group_sparsity=target_group_sparsity)
    
    def mark_unprunable_by_node_ids(self, node_ids=list()):
        for node_group in self._graph.node_groups.values():
            for node_id in node_ids:
                if node_id in node_group.nodes:
                    node_group.is_prunable = False

    def mark_unprunable_by_param_names(self, param_names=list()):
        param_names_set = set(param_names)
        for node_group in self._graph.node_groups.values():
            if set(node_group.param_names) & param_names_set:
                node_group.is_prunable = False

    def compute_flops(self, in_million=True, in_billion=False):
        return self._graph.compute_flops(in_million=in_million, in_billion=in_billion)
    
    def compute_num_params(self, in_million=True, in_billion=False):
        return self._graph.compute_num_params(in_million=in_million, in_billion=in_billion)