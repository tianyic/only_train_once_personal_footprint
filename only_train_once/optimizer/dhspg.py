import os
import sys
import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F
import time

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from assets.group_types import GROUP_TYPE
from .hyperparameter import DEFAULT_OPT_PARAMS

class DHSPG(Optimizer):

    def __init__(self, params, variant='sgd', lr=required, lmbda=None, lmbda_amplify=None, hat_lmbda_coeff=None, epsilon=0.0, first_momentum=None, second_momentum=None, dampening=None, weight_decay=None, 
                 target_group_sparsity=0.5, tolerance_group_sparsity=0.05, start_pruning_steps=0, partition_step=None, half_space_project_steps=None, warm_up_steps=0, group_divisible=1, fixed_zero_groups=True):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if partition_step is None or half_space_project_steps is None:
            assert start_pruning_steps >= 0 
            self.partition_step = start_pruning_steps
            self.half_space_project_steps = start_pruning_steps
        else:
            self.partition_step = partition_step
            self.half_space_project_steps = half_space_project_steps
        
        # Set up hyper-parameters related to group sparsity exploration
        lmbda = lmbda if lmbda is not None else DEFAULT_OPT_PARAMS[variant]['lmbda']
        lmbda_amplify = lmbda_amplify if lmbda_amplify is not None else DEFAULT_OPT_PARAMS[variant]['lmbda_amplify']
        hat_lmbda_coeff = hat_lmbda_coeff if hat_lmbda_coeff is not None else DEFAULT_OPT_PARAMS[variant]['hat_lmbda_coeff']
        # Set up hyper-parameters related to baseline optimizer
        first_momentum = first_momentum if first_momentum is not None else DEFAULT_OPT_PARAMS[variant]['first_momentum']
        second_momentum = second_momentum if second_momentum is not None else DEFAULT_OPT_PARAMS[variant]['second_momentum']
        dampening = dampening if dampening is not None else DEFAULT_OPT_PARAMS[variant]['dampening']
        weight_decay = weight_decay if weight_decay is not None else DEFAULT_OPT_PARAMS[variant]['weight_decay']
        
        self.partitoned = False
        self.warm_up_steps = warm_up_steps
        
        self.fixed_zero_groups = fixed_zero_groups
        
        self.safe_guard = 1e-8
        self.target_group_sparsity = target_group_sparsity
        self.tolerance_group_sparsity = tolerance_group_sparsity
        self.x_norm_g_p_all = 0.0
        self.x_norm_g_np_all = 0.0
        self.num_groups_g_p_all = 0
        self.num_groups_g_np_all = 0

        defaults = dict(lr=lr, lmbda=lmbda, lmbda_amplify=lmbda_amplify, hat_lmbda_coeff=hat_lmbda_coeff, epsilon=epsilon,\
                        weight_decay=weight_decay, first_momentum=first_momentum, second_momentum=second_momentum, \
                        dampening=dampening, variant=variant, num_groups_g_p=0, num_groups_g_np=0, \
                        num_steps=0, partitoned=False, x_norm_g_p=0.0, x_norm_g_np=0.0, grad_variant=dict(),
                        global_start_idx=0, global_idx=0, group_divisible=group_divisible)
        super(DHSPG, self).__init__(params, defaults)

        self.auxilary_params = dict()
        for group in self.param_groups:
            if group['group_type'] == GROUP_TYPE['auxilary']:
                self.auxilary_params[group['cc_id']] = group

        self.curr_group_sparsity, _ = self.compute_group_sparsity_omega()

    def __setstate__(self, state):
        super(DHSPG, self).__setstate__(state)
   
    def update_xs(self, group, xs, new_flatten_xs):
        track_id = 0
        left_pointer = 0
        right_pointer = 0

        # tackle params belonging to the group itself
        for param in group['params']:
            if param.grad is None:
                continue
            right_pointer += xs[track_id].shape[1]      
            param.data.copy_(new_flatten_xs[:, left_pointer:right_pointer].contiguous().view(param.shape))
            track_id += 1
            left_pointer = right_pointer

        # tackle params belonging to auxiliary groups
        for cc_id, offset in group['auxiliary_ccs']:
            auxiliary_cc_group = self.auxilary_params[cc_id]
            for param in auxiliary_cc_group['params']:
                if param.grad is None:
                    continue
                right_pointer += xs[track_id].shape[1]
                param.data[offset:offset+group['num_groups'], ...] = new_flatten_xs[:, left_pointer:right_pointer].contiguous()\
                     .view(param[offset:offset+group['num_groups'], ...].shape)
                track_id += 1
                left_pointer = right_pointer

    def get_first_momentum_grad(self, param_state, key, momentum, dampening, grad):
        if momentum > 0:
            if key not in param_state:
                buf = param_state[key] = grad
            else:
                buf = param_state[key]
                buf.mul_(momentum).add_(grad, alpha=(1.0-dampening))
            return buf
        else:
            return grad 

    def get_second_momentum_grad_square(self, param_state, key, momentum, dampening, grad):
        if momentum > 0:
            if key not in param_state:
                buf = param_state[key] = grad * grad
            else:
                buf = param_state[key]
                buf.mul_(momentum).add_(grad * grad, alpha=(1.0-dampening))
            return buf
        else:
            return grad * grad

    def get_xs_grads(self, group, require_grad=True):
        xs = []
        grads = []
        for i, param in enumerate(group['params']):
            if param.grad is None and require_grad:
                continue
            if len(param.data.shape) == 1:
                xs.append(param.data.view(group['num_groups'], -1))
                if require_grad:
                    grads.append(group['grad_variant'][i].view(group['num_groups'], -1))
            elif len(param.data.shape) == 4: # conv layer
                xs.append(param.data.view(group['num_groups'], -1))
                if require_grad:
                    grads.append(group['grad_variant'][i].view(group['num_groups'], -1))
            else:
                xs.append(param.data.view(group['num_groups'], -1))
                if require_grad:
                    grads.append(group['grad_variant'][i].view(group['num_groups'], -1))

        for i, (cc_id, offset) in enumerate(group['auxiliary_ccs']):
            auxiliary_cc_group = self.auxilary_params[cc_id]
            for j, param in enumerate(auxiliary_cc_group['params']):
                if param.grad is None and require_grad:
                    continue
                auxilary_params = param.data[offset:offset + group['num_groups'], ...]
                if require_grad:
                    auxilary_grads = auxiliary_cc_group['grad_variant'][j][offset:offset + group['num_groups'], ...]
                if len(param.data.shape) == 1:
                    xs.append(auxilary_params.view(group['num_groups'], -1))
                    if require_grad:
                        grads.append(auxilary_grads.view(group['num_groups'], -1))
                elif len(param.data.shape) == 4: # conv layer
                    xs.append(auxilary_params.view(group['num_groups'], -1))
                    if require_grad:
                        grads.append(auxilary_grads.view(group['num_groups'], -1))
                else:
                    xs.append(auxilary_params.view(group['num_groups'], -1))
                    if require_grad:
                        grads.append(auxilary_grads.view(group['num_groups'], -1))
        return xs, grads

    def partition_groups(self):
        print("partition_groups")
        self.x_norm_g_p_all = 0.0
        self.x_norm_g_np_all = 0.0
        total_num_groups = 0
        global_cos_sims = []
        global_norm_groups_avg = []
        global_start_idx = 0
        for i, group in enumerate(self.param_groups):
            if group['group_type'] >= GROUP_TYPE['multi-head-linear']:
                xs, grads = self.get_xs_grads(group)
                flatten_x = torch.cat(xs, dim = 1)
                flatten_grad = torch.cat(grads, dim = 1)
                cos_sim = F.cosine_similarity(-flatten_x, -flatten_grad)
                group_size = flatten_x.shape[1] if len(flatten_x.shape) > 1 else 1
                norm_avg_groups = torch.norm(flatten_x, dim=1) / group_size
                global_cos_sims.append(cos_sim)
                global_norm_groups_avg.append(norm_avg_groups)
                total_num_groups += group['num_groups']
                group['global_start_idx'] = global_start_idx
                group['global_idx'] = np.arange(global_start_idx, global_start_idx+group['num_groups'])
                global_start_idx += group['num_groups']

        self.K = int(total_num_groups * min(self.target_group_sparsity + self.tolerance_group_sparsity, 0.999))
        global_cos_sims = torch.cat(global_cos_sims, dim=0)
        global_norm_groups_avg = torch.cat(global_norm_groups_avg, dim=0)

        cos_sims_scores = torch.nn.functional.normalize(global_cos_sims, dim=0)
        norms_scores = torch.nn.functional.normalize(1.0 / (global_norm_groups_avg + 1e-6), dim=0)
        scores = 0.5 * cos_sims_scores + 0.5 * norms_scores
        topk_scores, topk_indices = torch.topk(scores, self.K)

        topk_indices = topk_indices.cpu().numpy()
        
        self.num_groups_g_p_all = 0
        self.num_groups_g_np_all = 0
        
        for i, group in enumerate(self.param_groups):
            group['partitoned'] = True
            if group['group_type'] >= GROUP_TYPE['multi-head-linear']:
                xs, grads = self.get_xs_grads(group)
                flatten_x = torch.cat(xs, dim = 1)
                flatten_grad = torch.cat(grads, dim = 1)
                
                group['magnitude_penalize'] = np.intersect1d(topk_indices, group['global_idx'])
                group['not_magnitude_penalize'] = np.setdiff1d(group['global_idx'], group['magnitude_penalize'])

                group['magnitude_penalize'] -= group['global_start_idx']
                group['not_magnitude_penalize'] -= group['global_start_idx']

                if group['num_groups'] <= group['group_divisible']:
                    group['magnitude_penalize'] = []
                    group['not_magnitude_penalize'] = np.arange(0, group['num_groups'])
                else:
                    curr_num_np_groups = len(group['not_magnitude_penalize'])
                    ratio = float(curr_num_np_groups) / float(group['group_divisible'])
                    refined_num_np_groups = int(max(round(ratio), 1) * group['group_divisible'])
                    groups_scores = scores[group['global_start_idx']:group['global_start_idx']+group['num_groups']]
                    _, refined_np_groups_indices = torch.topk(-groups_scores, refined_num_np_groups)
                    refined_np_groups_indices = refined_np_groups_indices.cpu().numpy()
                    group['not_magnitude_penalize'] = refined_np_groups_indices
                    group['magnitude_penalize'] = np.setdiff1d(np.arange(0, group['num_groups']), group['not_magnitude_penalize'])            

                group['magnitude_penalize_bool'] = torch.zeros(group['num_groups'], dtype=torch.bool).cuda()
                group['magnitude_penalize_bool'][group['magnitude_penalize']] = True

                group['num_groups_g_p'] = len(group['magnitude_penalize'])
                group['num_groups_g_np'] = len(group['not_magnitude_penalize'])

                group['x_norm_g_p'] = torch.sum(torch.norm(flatten_x[group['magnitude_penalize']], p=2, dim=1))
                group['x_norm_g_np'] = torch.sum(torch.norm(flatten_x[group['not_magnitude_penalize']], p=2, dim=1))

                assert group['num_groups_g_p'] + group['num_groups_g_np'] == group['num_groups']

                self.x_norm_g_p_all += group['x_norm_g_p']
                self.x_norm_g_np_all += group['x_norm_g_np']
                self.num_groups_g_p_all += group['num_groups_g_p']
                self.num_groups_g_np_all += group['num_groups_g_np']
        
    def compute_grad_variant(self):
        for i, group in enumerate(self.param_groups):
            is_adam = group['variant'] == 'adam' or group['variant'] == 'adamw'
            first_bias_correction = 1.0 - group['first_momentum'] ** group['num_steps'] if is_adam else None
            second_bias_correction = 1.0 - group['second_momentum'] ** group['num_steps'] if is_adam else None
            
            group['grad_variant'] = dict()
            for j, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                refined_grad_f = torch.clone(p.grad.data).detach()
                if group['weight_decay'] is not None and group['variant'] != 'adamw':
                    refined_grad_f += group['weight_decay'] * p.data
                if not is_adam:
                    if group['first_momentum'] > 0.0 or group['dampening'] > 0.0:
                        refined_grad_f = self.get_first_momentum_grad(group, f"grad_first_moment_buffer_group_{i}_param_{j}", 
                            group['first_momentum'], group['dampening'], refined_grad_f)
                    group['grad_variant'][j] = refined_grad_f
                else:
                    first_moment_grad = self.get_first_momentum_grad(group, f"grad_first_moment_buffer_group_{i}_param_{j}", 
                        group['first_momentum'], group['first_momentum'], refined_grad_f)                    
                    second_moment_grad_sq = self.get_second_momentum_grad_square(group, f"grad_second_moment_buffer_group_{i}_param_{j}", 
                        group['second_momentum'], group['second_momentum'], refined_grad_f)        

                    exp_avg_first_moment_grad = first_moment_grad / first_bias_correction
                    exp_avg_second_moment_grad_sq = second_moment_grad_sq / second_bias_correction
                    denom = exp_avg_second_moment_grad_sq.sqrt().add_(self.safe_guard)
                    group['grad_variant'][j] = exp_avg_first_moment_grad / denom

    def reach_target_group_sparsity(self):
        if self.curr_group_sparsity < self.target_group_sparsity:
            return False
        else:
            return True

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            group['num_steps'] += 1

        # First pass to compute gradient variant via different criteria
        self.compute_grad_variant()

        # Partition groups into G_p and G_np
        if not self.param_groups[0]['partitoned'] and self.param_groups[0]['num_steps'] - 1 == self.partition_step:
            print(self.param_groups[0]['num_steps'] - 1, self.partition_step)
            self.partition_groups()

        # Second pass to update variables
        for i, group in enumerate(self.param_groups):
            if group['group_type'] == GROUP_TYPE['no-update']:
                continue
            elif group['group_type'] == GROUP_TYPE['auxilary']:
                # The variables of auxiliary CC are updated within their stem CCs.
                continue
            elif group['group_type'] == GROUP_TYPE['default']:
                for j, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    if group['weight_decay'] is not None and group['variant'] == 'adamw':
                        p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                    p.data.add_(group['grad_variant'][j], alpha=-group['lr'])
            elif group['group_type'] == GROUP_TYPE['multi-head-linear']:
                # Since the transformer is facing some difficulty to convert onnx, 
                # we postpone the support of multi-head-attention temporally. 
                raise NotImplementedError
            elif group['group_type'] == GROUP_TYPE['linear'] or \
                 group['group_type'] == GROUP_TYPE['conv']:
                xs, grads = self.get_xs_grads(group)
                flatten_x = torch.cat(xs, dim = 1)
                flatten_grad = torch.cat(grads, dim = 1)
                flatten_hat_x = flatten_x - group['lr'] * flatten_grad

                if group['weight_decay'] is not None and group['variant'] == 'adamw':
                    flatten_hat_x = flatten_hat_x - group['lr'] * group['weight_decay'] * flatten_x
                
                flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
                # If some groups needs to penalize magnitude    
                if group['num_groups_g_p'] > 0 and not self.reach_target_group_sparsity():
                    # Use adjusted lambda to aggressively penalize magnitude 
                    if group['num_steps'] >= self.warm_up_steps:
                        flatten_grad_norm = torch.norm(flatten_grad, p=2, dim=1)            
                        flatten_x_grad_inner_prod = torch.sum(flatten_x * flatten_grad, dim=1)

                        lambdas = torch.ones_like(flatten_x_norm) * group['lmbda']
                        # If not reach target group sparsity, adjust lambda if needed
                        if not self.reach_target_group_sparsity():
                            # Groups need to adjust lambda 
                            groups_adjust_lambda = group['magnitude_penalize_bool'].cuda() & (flatten_x_grad_inner_prod < 0)
                            lambdas_lower_bound = -flatten_x_grad_inner_prod[groups_adjust_lambda] / flatten_x_norm[groups_adjust_lambda]
                            lambdas_upper_bound = -(flatten_grad_norm[groups_adjust_lambda] * flatten_grad_norm[groups_adjust_lambda] *\
                                                flatten_x_norm[groups_adjust_lambda] / flatten_x_grad_inner_prod[groups_adjust_lambda])
                            lambdas_adjust = torch.clip((group['lmbda_amplify'] * lambdas_lower_bound), \
                                                        min=group['lmbda'], \
                                                        max=group['lmbda'] * group['hat_lmbda_coeff']) 
                            exceeding_upper_bound = lambdas_adjust >= lambdas_upper_bound
                            lambdas_adjust[exceeding_upper_bound] = (lambdas_upper_bound[exceeding_upper_bound] + lambdas_lower_bound[exceeding_upper_bound]) / 2.0
                            lambdas[groups_adjust_lambda] = lambdas_adjust

                        grad_mixed_l1_l2 = torch.zeros_like(flatten_grad)
                        non_zero_idxes = flatten_x_norm > 0.0
                        grad_mixed_l1_l2[non_zero_idxes] = flatten_x[non_zero_idxes] / (flatten_x_norm[non_zero_idxes] + self.safe_guard).unsqueeze(1)

                        flatten_hat_x[group['magnitude_penalize_bool']] -= group['lr'] * lambdas[group['magnitude_penalize_bool']].unsqueeze(1) * grad_mixed_l1_l2[group['magnitude_penalize_bool']]

                        if group['num_steps'] >= self.half_space_project_steps:
                            flatten_hat_x[group['magnitude_penalize_bool']] = self.half_space_project(flatten_hat_x[group['magnitude_penalize_bool']], flatten_x[group['magnitude_penalize_bool']], group['epsilon'], group['upper_group_sparsity'])
                    # Use default lambda to moderately penalize magnitude  
                    else:
                        lambdas = torch.ones_like(flatten_x_norm) * group['lmbda']
                        grad_mixed_l1_l2 = torch.zeros_like(flatten_grad)
                        non_zero_idxes = flatten_x_norm > 0.0
                        grad_mixed_l1_l2[non_zero_idxes] = flatten_x[non_zero_idxes] / (flatten_x_norm[non_zero_idxes] + self.safe_guard).unsqueeze(1)
                        flatten_hat_x[group['magnitude_penalize_bool']] -= group['lr'] * lambdas[group['magnitude_penalize_bool']].unsqueeze(1) * grad_mixed_l1_l2[group['magnitude_penalize_bool']]                
                # Fixed non_free variables
                if group['num_steps'] > self.half_space_project_steps or self.fixed_zero_groups:                   
                    zero_group_idxes = flatten_x_norm == 0.0
                    flatten_hat_x[zero_group_idxes, ...] = 0.0
                self.update_xs(group, xs, flatten_hat_x)
            else:
                raise("some parameters are not in any group type, please check")
        
        self.curr_group_sparsity, _ = self.compute_group_sparsity_omega()
        return 

    def half_space_project(self, hat_x, x, epsilon, upper_group_sparsity):
        num_groups = x.shape[0]
        x_norm = torch.norm(x, p=2, dim=1)
        before_group_sparsity = torch.sum(x_norm == 0) / float(num_groups)
        if before_group_sparsity < upper_group_sparsity:
            proj_idx = (torch.bmm(hat_x.view(hat_x.shape[0], 1, -1), x.view(x.shape[0], -1, 1)).squeeze() \
                < epsilon * x_norm ** 2)    
            
            trial_group_sparsity = torch.sum(torch.logical_or(proj_idx, x_norm == 0)) / float(num_groups)
            # if trial group sparsity larger than upper group sparsity, then control the size of half-space projection
            if trial_group_sparsity > upper_group_sparsity:
                max_num_proj_groups = int(num_groups * (trial_group_sparsity - upper_group_sparsity))
                max_num_proj_groups = min(max(0, max_num_proj_groups), num_groups - 1)
                proj_group_idxes = torch.arange(num_groups)[proj_idx == True]
                refined_proj_idxes = torch.randperm(torch.sum(proj_idx))[:max_num_proj_groups].sort()[0]
                hat_x[proj_group_idxes[refined_proj_idxes], ...] = 0.0
            else:
                hat_x[proj_idx, ...] = 0.0
        return hat_x

    def compute_group_sparsity_omega(self):
        total_num_groups = torch.zeros(len(GROUP_TYPE)) + 1e-6
        total_num_zero_groups = torch.zeros(len(GROUP_TYPE))

        omega = 0.0
        for group in self.param_groups:
            if group['group_type'] == GROUP_TYPE['no-update']:
                """Params that do not need update"""
                total_num_zero_groups[group['group_type']] = 0
            elif group['group_type'] == GROUP_TYPE['default']:
                """Params that are not included in the regularization"""
                total_num_zero_groups[group['group_type']] = 0
            elif group['group_type'] == GROUP_TYPE['auxilary']:
                pass
            elif group['group_type'] == GROUP_TYPE['multi-head-linear']:
                xs, _ = self.get_xs_grads(group, require_grad=False)
                flatten_x = torch.cat(xs, dim = 1)

                num_groups = flatten_x.shape[0] // group['num_heads']
                flatten_x = self.reorg_multi_head(flatten_x, num_groups, group['num_heads'])

                flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
                zero_group_idxes = flatten_x_norm == 0.0
                total_num_groups[group['group_type']] += num_groups
                total_num_zero_groups[group['group_type']] += torch.sum(zero_group_idxes).cpu()
                omega += torch.sum(flatten_x_norm)
            elif group['group_type'] == GROUP_TYPE['linear']:
                """Group for standard linear layer"""
                xs, _ = self.get_xs_grads(group, require_grad=False)
                flatten_x = torch.cat(xs, dim = 1)                       

                flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
                zero_group_idxes = flatten_x_norm == 0.0
                total_num_groups[group['group_type']] += flatten_x.shape[0]
                total_num_zero_groups[group['group_type']] += torch.sum(zero_group_idxes).cpu()
                omega += torch.sum(flatten_x_norm)
            elif group['group_type'] == GROUP_TYPE['conv']:
                xs, _ = self.get_xs_grads(group, require_grad=False)
                flatten_x = torch.cat(xs, dim = 1)                       

                flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
                zero_group_idxes = flatten_x_norm == 0.0
                total_num_groups[group['group_type']] += flatten_x.shape[0]
                total_num_zero_groups[group['group_type']] += torch.sum(zero_group_idxes).cpu()
                omega += torch.sum(flatten_x_norm)
        
        overall_group_sparsity = torch.sum(total_num_zero_groups) / torch.sum(total_num_groups)
        return overall_group_sparsity.cpu().numpy(), omega.cpu().numpy()

    def compute_norm_group_partitions(self):
        self.x_norm_g_p_all = 0.0
        self.x_norm_g_np_all = 0.0
        self.num_groups_g_p_all = 0
        self.num_groups_g_np_all = 0
        
        for i, group in enumerate(self.param_groups):
            if group['group_type'] >= GROUP_TYPE['multi-head-linear']:
                xs, _ = self.get_xs_grads(group, require_grad=False)
                flatten_x = torch.cat(xs, dim = 1)
                if 'magnitude_penalize' not in group or 'not_magnitude_penalize' not in group:
                    return 0.0, 0.0, 0, 0
                group['x_norm_g_p'] = torch.sum(torch.norm(flatten_x[group['magnitude_penalize']], p=2, dim=1))
                group['x_norm_g_np'] = torch.sum(torch.norm(flatten_x[group['not_magnitude_penalize']], p=2, dim=1))
                self.x_norm_g_p_all += group['x_norm_g_p']
                self.x_norm_g_np_all += group['x_norm_g_np']
                self.num_groups_g_p_all += group['num_groups_g_p']
                self.num_groups_g_np_all += group['num_groups_g_np']
        return self.x_norm_g_p_all.cpu().item(), self.x_norm_g_np_all.cpu().item(), self.num_groups_g_p_all, self.num_groups_g_np_all

    def set_learning_rate(self, lr):
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def get_learning_rate(self):
        for param_group in self.param_groups:
            lr = param_group['lr']
        return lr

    def set_lmbda(self, lmbda):
        for param_group in self.param_groups:
            param_group['lmbda'] = lmbda

    def get_lmbda(self):
        lmbda = 0.0
        for param_group in self.param_groups:
            lmbda = param_group['lmbda']
        return lmbda

    def set_hat_lmbda_coeff(self, hat_lmbda_coeff):
        for param_group in self.param_groups:
            param_group['hat_lmbda_coeff'] = hat_lmbda_coeff

    def get_hat_lmbda_coeff(self):
        hat_lmbda_coeff = 0.0
        for param_group in self.param_groups:
            hat_lmbda_coeff = param_group['hat_lmbda_coeff']
        return hat_lmbda_coeff

    def set_lmbda_amplify(self, lmbda_amplify):
        for param_group in self.param_groups:
            param_group['lmbda_amplify'] = lmbda_amplify

    def get_lmbda_amplify(self):
        lmbda_amplify = 0.0
        for param_group in self.param_groups:
            lmbda_amplify = param_group['lmbda_amplify']
        return lmbda_amplify

    def set_eps(self, eps):
        for param_group in self.param_groups:
            param_group['epsilon'] = eps
    
    def get_eps(self):
        eps = -1
        for param_group in self.param_groups:
            eps = max(eps, param_group['epsilon'])
        return eps
