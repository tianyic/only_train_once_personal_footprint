import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F

from .hyperparameter import DEFAULT_OPT_PARAMS, SUPPORT_GRADIENT_ESTIMATES
from .importance_score import calculate_importance_score_dhspg
from only_train_once.transform import tensor_transformation, TensorTransform

class HESSO(Optimizer):
    '''
    HESSO: Hybrid Efficient Structured Sparse Optimizer
    '''
    def __init__(self, params, variant='sgd', lr=required, first_momentum=None, second_momentum=None, \
                 dampening=None, weight_decay=None, target_group_sparsity=0.5, \
                 tolerance_group_sparsity=0.05, start_pruning_step=0, pruning_steps=None, pruning_periods=1, \
                 group_divisible=1, fixed_zero_groups=True, importance_score_criteria='default'):

        print("Setup HESSO")
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if variant not in SUPPORT_GRADIENT_ESTIMATES:
            raise ValueError("Need to select a gradient estimation from {}".format(SUPPORT_GRADIENT_ESTIMATES))
        
        self.num_steps = 0
        self.start_pruning_step = start_pruning_step
        self.pruning_periods = int(max(1, pruning_periods)) # How many periods that the pruning last for.
        self.pruning_steps = pruning_steps
        self.pruning_period_duration = self.pruning_steps // self.pruning_periods # How many pruning steps for each period
        self.curr_pruning_period = 0 # Track pruning period

        # Set up hyper-parameters related to baseline optimizer
        first_momentum = first_momentum if first_momentum is not None else DEFAULT_OPT_PARAMS[variant]['first_momentum']
        second_momentum = second_momentum if second_momentum is not None else DEFAULT_OPT_PARAMS[variant]['second_momentum']
        dampening = dampening if dampening is not None else DEFAULT_OPT_PARAMS[variant]['dampening']
        weight_decay = weight_decay if weight_decay is not None else DEFAULT_OPT_PARAMS[variant]['weight_decay']
        
        self.fixed_zero_groups = fixed_zero_groups
        
        self.safe_guard = 1e-8
        self.target_group_sparsity = target_group_sparsity
        self.tolerance_group_sparsity = tolerance_group_sparsity
        
        self.norm_important_groups = 0.0 # norm for important groups
        self.norm_redundant_groups = 0.0 # norm for redundant groups
        self.num_important_groups = 0 # number of important groups
        self.num_redundant_groups = 0 # number of redundant groups

        self.pruned_group_idxes = list()
        if importance_score_criteria == 'default':
            self.importance_score_criteria = {'magnitude': 0.2, 'avg_magnitude': 0.2,\
                                              'cosine_similarity': 0.2, \
                                              'taylor_first_order': 0.2, 'taylor_second_order': 0.2}
        else:
            self.importance_score_criteria = importance_score_criteria

        defaults = dict(variant=variant, lr=lr, weight_decay=weight_decay, first_momentum=first_momentum, second_momentum=second_momentum, \
                        dampening=dampening, grad_variant=dict(), \
                        global_start_idx=0, global_idx=0)
        
        super(HESSO, self).__init__(params, defaults)

        self.group_divisible = group_divisible
        self.first_moment_grads = dict()
        self.second_moment_grads = dict()

        # Set up total number of prunable groups
        self.total_num_groups = 0
        
        for param_group in params:
            if param_group['is_prunable'] and not param_group['is_auxiliary']:
                self.total_num_groups += param_group['num_groups']

        self.target_num_redundant_groups = int(self.total_num_groups * min(self.target_group_sparsity, 0.999))

        # print(self.target_num_redundant_groups, self.total_num_groups)
        self.active_num_redundant_groups = list()
        # Set up active number redundant groups for each pruning period
        groups_sum = 0
        for p in range(self.pruning_periods):
            if p == self.pruning_periods - 1:
                self.active_num_redundant_groups.append(
                    self.target_num_redundant_groups - groups_sum
                )
            else:
                self.active_num_redundant_groups.append(
                    self.target_num_redundant_groups // self.pruning_periods
                )
                groups_sum += self.active_num_redundant_groups[p]
        print("Target redundant groups per period: ", self.active_num_redundant_groups)
        
        self.important_idxes = dict()
        self.pruned_idxes = dict()
        self.active_redundant_idxes = dict()
        
        for param_group in params:
            self.important_idxes[param_group['id']] = [i for i in range(param_group['num_groups'])]
            self.pruned_idxes[param_group['id']] = list()
            self.active_redundant_idxes[param_group['id']] = list()
        
        self.auxiliary_param_groups = dict()
        for group in self.param_groups:
            if group['is_auxiliary']:
                self.auxiliary_param_groups[group['id']] = group
        
        self.curr_group_sparsity, _, self.curr_num_zero_groups = self.compute_group_sparsity_param_norm()

    def __setstate__(self, state):
        super(HESSO, self).__setstate__(state)

    def get_first_momentum_grad(self, name, first_moment, dampening, grad):
        if first_moment > 0:
            if name not in self.first_moment_grads:
                buf = self.first_moment_grads[name] = grad
            else:
                buf = self.first_moment_grads[name]
                buf.mul_(first_moment).add_(grad, alpha=(1.0-dampening))
            return buf
        else:
            return grad

    def get_second_momentum_grad_square(self, name, second_moment, dampening, grad):
        if second_moment > 0:
            if name not in self.second_moment_grads:
                buf = self.second_moment_grads[name] = grad * grad
            else:
                buf = self.second_moment_grads[name]
                buf.mul_(second_moment).add_(grad * grad, alpha=(1.0-dampening))
            return buf
        else:
            return grad * grad


    def compute_grad_variant(self):
        for i, group in enumerate(self.param_groups):
            is_adam = group['variant'] == 'adam' or group['variant'] == 'adamw'
            first_bias_correction = 1.0 - group['first_momentum'] ** self.num_steps if is_adam else None
            second_bias_correction = 1.0 - group['second_momentum'] ** self.num_steps if is_adam else None
            group['grad_variant'] = list()
            for j, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                refined_grad_f = torch.clone(p.grad.data).detach()
                if group['weight_decay'] is not None and group['variant'] != 'adamw':
                    refined_grad_f += group['weight_decay'] * p.data
                if not is_adam:
                    if group['first_momentum'] > 0.0 or group['dampening'] > 0.0:
                        refined_grad_f = self.get_first_momentum_grad(f"grad_first_moment_buffer_group_{i}_param_{j}", 
                            group['first_momentum'], group['dampening'], refined_grad_f)
                    group['grad_variant'].append(refined_grad_f)
                else:
                    first_moment_grad = self.get_first_momentum_grad(f"grad_first_moment_buffer_group_{i}_param_{j}", 
                        group['first_momentum'], group['first_momentum'], refined_grad_f)                    
                    second_moment_grad_sq = self.get_second_momentum_grad_square(f"grad_second_moment_buffer_group_{i}_param_{j}", 
                        group['second_momentum'], group['second_momentum'], refined_grad_f)        

                    exp_avg_first_moment_grad = first_moment_grad / first_bias_correction
                    exp_avg_second_moment_grad_sq = second_moment_grad_sq / second_bias_correction
                    denom = exp_avg_second_moment_grad_sq.sqrt().add_(self.safe_guard)
                    group['grad_variant'].append(exp_avg_first_moment_grad / denom)
        
    def reach_target_group_sparsity(self):
        if self.curr_num_zero_groups < self.target_num_redundant_groups:
            return False
        else:
            return True
        
    def compute_importance_scores(self):
        global_start_idx = 0
        self.global_scores = list() # Accumulate global scores
        # Calculate raw importance scores by varying criteria
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                calculate_importance_score_dhspg(self.importance_score_criteria, group)

        # Normalize importance_score
        # Calculate normalization_denoms
        normalization_denoms = dict.fromkeys(self.importance_score_criteria.keys(), self.safe_guard)
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                for proxy_name in self.importance_score_criteria:
                    normalization_denoms[proxy_name] += torch.sum(group['importance_scores'][proxy_name] ** 2, dim=0).item()
        for proxy_name in normalization_denoms:
            normalization_denoms[proxy_name] = np.sqrt(normalization_denoms[proxy_name]) + self.safe_guard

        global_start_idx = 0
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                group['importance_scores']['overall'] = None
                for proxy_name in self.importance_score_criteria:
                    if not proxy_name in group['importance_scores']:
                        continue
                    group['importance_scores'][proxy_name].mul_(self.importance_score_criteria[proxy_name] / normalization_denoms[proxy_name])
                    if group['importance_scores']['overall'] is None:
                        group['importance_scores']['overall'] = group['importance_scores'][proxy_name].clone()
                    else:
                        group['importance_scores']['overall'] += group['importance_scores'][proxy_name]
                group['global_start_idx'] = global_start_idx
                group['global_idxes'] = np.arange(global_start_idx, global_start_idx+group['num_groups'])
                global_start_idx += group['num_groups']
                self.global_scores.append(group['importance_scores']['overall'])
        
    def identify_redundant_groups(self):
        global_scores = torch.cat(self.global_scores, dim=0)
        curr_active_num_redundant_groups = self.active_num_redundant_groups[self.curr_pruning_period]
        curr_K = len(self.pruned_group_idxes) + curr_active_num_redundant_groups
        _, top_indices = torch.topk(-global_scores, curr_K)
        top_indices = top_indices.cpu().numpy()
        top_indices = np.setdiff1d(top_indices, self.pruned_group_idxes)[:curr_active_num_redundant_groups].tolist()
        self.pruned_group_idxes.extend(top_indices)

        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                global_active_redundant_idx = np.intersect1d(top_indices, group['global_idxes'])
                self.active_redundant_idxes[group['id']] = (global_active_redundant_idx - group['global_start_idx']).tolist()
                # Refine important_idx by group_divisible
                if group['num_groups'] < self.group_divisible:
                    self.active_redundant_idxes[group['id']] = list()
                    self.pruned_idxes[group['id']] = list()
                else:
                    curr_num_important_groups = len(self.important_idxes[group['id']])
                    trial_num_important_groups = curr_num_important_groups - len(self.active_redundant_idxes[group['id']])
                    if trial_num_important_groups % self.group_divisible != 0 or trial_num_important_groups <= 0:
                        ratio = trial_num_important_groups // self.group_divisible + 1 # Add one will preserve more groups, otherwise will slim more.
                        refined_num_important_groups = None
                        if ratio <= 1 or trial_num_important_groups == 0:
                            refined_num_important_groups = max(int(self.group_divisible), 1)
                        else:
                            refined_num_important_groups = max(int(ratio * self.group_divisible), int(self.group_divisible))
                        refined_num_important_groups = min(group['num_groups'], refined_num_important_groups)
                        refined_num_active_redundant_groups = group['num_groups'] - len(self.pruned_idxes[group['id']]) - refined_num_important_groups
                        self.target_num_redundant_groups += (refined_num_active_redundant_groups - len(self.active_redundant_idxes[group['id']]))
                        self.active_redundant_idxes[group['id']] = self.active_redundant_idxes[group['id']][:refined_num_active_redundant_groups]
                self.important_idxes[group['id']] = [i for i in self.important_idxes[group['id']] if (i not in self.active_redundant_idxes[group['id']] and i not in self.pruned_idxes[group['id']])]
                group['active_redundant_bool'] = torch.zeros(group['num_groups'], dtype=torch.bool).cuda()
                group['active_redundant_bool'][self.active_redundant_idxes[group['id']]] = True
        return

    def commit_redundant_idxes(self):
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                self.pruned_idxes[group['id']].extend(self.active_redundant_idxes[group['id']])
                self.active_redundant_idxes[group['id']] = list()
                self.important_idxes[group['id']] = [i for i in range(group['num_groups']) if i not in self.pruned_idxes[group['id']]]
                group['importance_scores'] = dict()
                
    def step(self):   
        self.num_steps += 1   
          
        # First pass to compute gradient variant via different criteria
        self.compute_grad_variant()

        # Partition groups into important and redundant groups  
        if self.num_steps >= self.start_pruning_step and not self.reach_target_group_sparsity() and \
            self.curr_pruning_period < self.pruning_periods:
            if (self.num_steps - self.start_pruning_step - 1) % self.pruning_period_duration == 0:
                self.commit_redundant_idxes()
                self.compute_importance_scores()
                self.identify_redundant_groups()
                self.curr_pruning_period += 1
                
        # Second pass to update variables        
        t = (self.num_steps - self.start_pruning_step) % self.pruning_period_duration
        for i, group in enumerate(self.param_groups):
            # print(self.num_steps, group['id'], group['is_prunable'], group['p_names'], group['num_groups'], group['p_transform'], group['auxiliary_ngs'])
            if not group['is_prunable'] or len(self.active_redundant_idxes[group['id']]) == 0:
                for j, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    if group['weight_decay'] is not None and group['variant'] == 'adamw':
                        p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                    p.data.add_(group['grad_variant'][j], alpha=-group['lr'])
            elif group['is_prunable'] and len(self.active_redundant_idxes[group['id']]) > 0:
                for j, (p, p_transform) in enumerate(zip(group['params'], group['p_transform'])):
                    if p.grad is None:
                        continue
                    if group['weight_decay'] is not None and group['variant'] == 'adamw':
                        p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                    p.data.add_(group['grad_variant'][j], alpha=-group['lr'])
                    if p_transform == TensorTransform.TRANSPOSE and len(p.data.shape) > 1:
                        p.data[:, group['active_redundant_bool'], ...] *= (self.pruning_period_duration - t - 1.0) / (self.pruning_period_duration - t)
                    else:
                        p.data[group['active_redundant_bool']] *= (self.pruning_period_duration - t - 1.0) / (self.pruning_period_duration - t)
                    
                    # Tackle auxiliary params
                    for ng_id, offset in group['auxiliary_ngs']:
                        aux_pg = self.auxiliary_param_groups[ng_id]
                        for aux_p in aux_pg['params']:
                            if aux_p.grad is None:
                                continue
                            # TODO: this update seems wrong.
                            aux_p.data[offset:offset+group['num_groups'], ...] *= (self.pruning_period_duration - t - 1.0) / (self.pruning_period_duration - t)

            if len(self.pruned_idxes[group['id']]) > 0:
                for j, (p, p_transform) in enumerate(zip(group['params'], group['p_transform'])):
                    if p_transform == TensorTransform.TRANSPOSE and len(p.data.shape) > 1:
                        p.data[:, self.pruned_idxes[group['id']]] = 0.0        
                    else:
                        p.data[self.pruned_idxes[group['id']]] = 0.0
            
        if self.num_steps >= self.start_pruning_step and t == self.pruning_period_duration - 1:
            self.commit_redundant_idxes()
            
        self.curr_group_sparsity, _, self.curr_num_zero_groups = self.compute_group_sparsity_param_norm()                
        return 
    
    def compute_group_sparsity_param_norm(self):
        total_num_zero_groups = 0
        norm_x = 0.0

        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                norm_group = None
                for param, p_transform in zip(group['params'], group['p_transform']):
                    param_transform = tensor_transformation(param, p_transform, group['num_groups'])
                    if norm_group == None:
                        norm_group = torch.norm(param_transform, dim=1) ** 2
                    else:
                        norm_group += torch.norm(param_transform, dim=1) ** 2
                norm_group = torch.sqrt(norm_group)
                
                num_zero_groups = torch.sum(norm_group == 0).item()
                total_num_zero_groups += num_zero_groups
                norm_x += torch.sum(norm_group).item()
                        
        group_sparsity = total_num_zero_groups / float(self.total_num_groups + self.safe_guard)
        return group_sparsity, norm_x, total_num_zero_groups

    def compute_norm_groups(self):
        self.norm_important_groups = 0.0
        self.norm_redundant_groups = 0.0
        self.num_important_groups = 0
        self.num_redundant_groups = 0
        
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                id = group['id']
                import_idxes = self.important_idxes[id]
                redund_idxes = self.pruned_idxes[id] + self.active_redundant_idxes[id]
                norm_group = None
                for param, p_transform in zip(group['params'], group['p_transform']):
                    param_transform = tensor_transformation(param, p_transform, group['num_groups'])
                    if norm_group == None:
                        norm_group = torch.norm(param_transform, dim=1) ** 2
                    else:
                        norm_group += torch.norm(param_transform, dim=1) ** 2
                norm_group = torch.sqrt(norm_group)
                self.norm_important_groups += torch.sum(norm_group[import_idxes]).item()
                self.norm_redundant_groups += torch.sum(norm_group[redund_idxes]).item()
                self.num_important_groups += len(import_idxes)
                self.num_redundant_groups += len(redund_idxes)
                
        return self.norm_important_groups, self.norm_redundant_groups, self.num_important_groups, self.num_redundant_groups

    def set_learning_rate(self, lr):
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def get_learning_rate(self):
        for param_group in self.param_groups:
            lr = param_group['lr']
        return lr