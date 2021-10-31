import time
import os
import csv
import torch
import numpy as np
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
import argparse
import time

class HSPG(Optimizer):

    def __init__(self, params, lr=required, lmbda=required, momentum=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if lmbda is not required and lmbda < 0.0:
            raise ValueError("Invalid lambda: {}".format(lmbda))

        if momentum is not required and momentum < 0.0:
            raise ValueError("Invalid momentum: {}".format(momentum))

        defaults = dict(lr=lr, lmbda=lmbda, momentum=momentum)
        super(HSPG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(HSPG, self).__setstate__(state)
    
    def get_xs_grad_fs(self, params):
        xs = []
        grad_fs = []
        for p in params:
            if p.grad is None:
                continue
            if len(p.data.shape) == 1:
                xs.append(p.data.unsqueeze(1))
                grad_fs.append(p.grad.data.unsqueeze(1))
            elif len(p.data.shape) == 4: # conv layer
                xs.append(p.data.view(p.data.shape[0], -1))
                grad_fs.append(p.grad.data.view(p.grad.data.shape[0], -1))

            else:
                xs.append(p.data)
                grad_fs.append(p.grad.data)    
        return xs, grad_fs    

    def get_xs(self, params):
        xs = []
        for p in params:
            if len(p.data.shape) == 1:
                xs.append(p.data.unsqueeze(1))
            elif len(p.data.shape) == 4: # conv layer
                xs.append(p.data.view(p.data.shape[0], -1))
            else:
                xs.append(p.data)
        return xs    

    def update_xs_given_new_xs(self, params, xs, new_flatten_xs, shapes=None):
        if shapes is not None: # conv layer
            left_pointer = 0
            right_pointer = 0
            for i, x in enumerate(xs):
                right_pointer += shapes[i][1:].numel()
                params[i].data.copy_(new_flatten_xs[:, left_pointer:right_pointer].view(shapes[i]))
                left_pointer = right_pointer
        else:
            left_pointer = 0
            right_pointer = 0
            for i, x in enumerate(xs):
                right_pointer += x.shape[1]
                if right_pointer - left_pointer == 1:
                    params[i].data.copy_(new_flatten_xs[:, left_pointer:right_pointer].squeeze(1))
                else:
                    params[i].data.copy_(new_flatten_xs[:, left_pointer:right_pointer])
                left_pointer = right_pointer      

    def mixed_l1_l2_subgrad_psi(self, flatten_x, flatten_grad_f, lmbda):
        flatten_subgrad_reg = torch.zeros_like(flatten_grad_f)
        norm = torch.norm(flatten_x, p=2, dim=1)
        non_zero_mask = norm != 0
        flatten_subgrad_reg[non_zero_mask] = flatten_x[non_zero_mask] / (norm[non_zero_mask] + 1e-6).unsqueeze(1)
        flatten_subgrad_psi = flatten_grad_f + lmbda * flatten_subgrad_reg
        return flatten_subgrad_psi

    def grad_descent_update(self, x, lr, grad):
        return x - lr * grad

    def reorg_multi_head(self, x, num_groups, num_heads):
        return x.view(num_heads, num_groups, -1).permute(1, 0, 2).contiguous().view(num_groups, -1)

    def reverse_reorg_multi_head(self, x, num_groups, num_heads):
        return x.view(num_groups, num_heads, -1).permute(1, 0, 2).contiguous().view(num_heads * num_groups, -1)

    def sgd_step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            if group['group_type'] == 0:
                """Params that do not need update"""
                continue
            elif group['group_type'] == 1:
                """Params that are not included in the regularization"""
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad_f = p.grad.data
                    p.data.add_(-group['lr'], grad_f)
            elif group['group_type'] == 2:
                """Group for multi-head linear layer"""
                xs, grad_fs = self.get_xs_grad_fs(group['params'])
                flatten_x = torch.cat(xs, dim = 1)
                flatten_grad_f = torch.cat(grad_fs, dim = 1)

                num_groups = flatten_x.shape[0] // group['num_heads']
                flatten_x = self.reorg_multi_head(flatten_x, num_groups, group['num_heads'])
                flatten_grad_f = self.reorg_multi_head(flatten_grad_f, num_groups, group['num_heads'])

                flatten_subgrad_psi = self.mixed_l1_l2_subgrad_psi(flatten_x, flatten_grad_f, group['lmbda'])
                flatten_subgrad_psi = self.get_momentum_grad(group, 'momentum_buffer_sgd', group['momentum'], flatten_subgrad_psi)

                # recover shape
                flatten_next_x = self.grad_descent_update(flatten_x, group['lr'], flatten_subgrad_psi)
                flatten_next_x = self.reverse_reorg_multi_head(flatten_next_x, num_groups, group['num_heads'])

                self.update_xs_given_new_xs(group['params'], xs, flatten_next_x)

            elif group['group_type'] == 3:
                """Group for standard linear layer"""
                xs, grad_fs = self.get_xs_grad_fs(group['params'])
                flatten_x = torch.cat(xs, dim = 1)
                flatten_grad_f = torch.cat(grad_fs, dim = 1)
                flatten_subgrad_psi = self.mixed_l1_l2_subgrad_psi(flatten_x, flatten_grad_f, group['lmbda'])
                flatten_subgrad_psi = self.get_momentum_grad(group, 'momentum_buffer_sgd', group['momentum'], flatten_subgrad_psi)

                flatten_next_x = self.grad_descent_update(flatten_x, group['lr'], flatten_subgrad_psi)
                self.update_xs_given_new_xs(group['params'], xs, flatten_next_x)

            elif group['group_type'] == 4:
                """Group for standard Conv layer"""
                xs, grad_fs = self.get_xs_grad_fs(group['params'])
                flatten_x = torch.cat(xs, dim = 1)
                flatten_grad_f = torch.cat(grad_fs, dim = 1)

                flatten_subgrad_psi = self.mixed_l1_l2_subgrad_psi(flatten_x, flatten_grad_f, group['lmbda'])
                flatten_subgrad_psi = self.get_momentum_grad(group, 'momentum_buffer_sgd', group['momentum'], flatten_subgrad_psi)

                flatten_next_x = self.grad_descent_update(flatten_x, group['lr'], flatten_subgrad_psi)
                self.update_xs_given_new_xs(group['params'], xs, flatten_next_x, shapes=group['shapes'])

            else:
                raise("some parameters are not in any group type, please check")
        return loss

    def get_momentum_grad(self, param_state, key, momentum, grad):
        if momentum > 0:
            if key not in param_state:
                buf = param_state[key] = grad
            else:
                buf = param_state[key]
                buf.mul_(momentum).add_(grad)
            return buf
        else:
            return grad 
    
    def half_space_project(self, hat_x, x, epsilon, upper_group_sparsity):
        num_groups = x.shape[0]
        x_norm = torch.norm(x, p=2, dim=1)
        before_group_sparsity = torch.sum(x_norm == 0) / float(num_groups)
        if before_group_sparsity < upper_group_sparsity:
            proj_idx = (torch.bmm(hat_x.view(hat_x.shape[0], 1, -1), x.view(x.shape[0], -1, 1)).squeeze() \
                < epsilon * x_norm ** 2)    
            
            trial_group_sparsity = torch.sum(torch.logical_or(proj_idx, x_norm == 0))\
                 / float(num_groups)
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

    def half_space_step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if group['group_type'] == 0:
                """Params that do not need update"""
                continue
            elif group['group_type'] == 1:
                """Params that are not included in the regularization"""
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad_f = p.grad.data
                    grad_f = self.get_momentum_grad(self.state[p], 'momentum_buffer_half_space', group['momentum'], grad_f)
                    p.data.add_(-group['lr'], grad_f)        
                    continue
        
            elif group['group_type'] == 2:
                """Group for multi-head linear layer"""
                xs, grad_fs = self.get_xs_grad_fs(group['params'])
                flatten_x = torch.cat(xs, dim = 1)
                flatten_grad_f = torch.cat(grad_fs, dim = 1)

                num_groups = flatten_x.shape[0] // group['num_heads']
                flatten_x = self.reorg_multi_head(flatten_x, num_groups, group['num_heads'])
                flatten_grad_f = self.reorg_multi_head(flatten_grad_f, num_groups, group['num_heads'])
                
                flatten_subgrad_psi = self.mixed_l1_l2_subgrad_psi(flatten_x, flatten_grad_f, group['lmbda'])
                flatten_subgrad_psi = self.get_momentum_grad(group, 'momentum_buffer_half_space', group['momentum'], flatten_subgrad_psi)

                flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
                zero_group_idxes = flatten_x_norm == 0.0

                # compute trial iterate
                flatten_hat_x = self.grad_descent_update(flatten_x, group['lr'], flatten_subgrad_psi)

                # do half space projection
                flatten_hat_x =  self.half_space_project(flatten_hat_x, flatten_x, group['epsilon'], group['upper_group_sparsity'])

                # fixed non_free variables
                flatten_hat_x[zero_group_idxes, ...] = 0.0

                # recover shape
                flatten_hat_x = self.reverse_reorg_multi_head(flatten_hat_x, num_groups, group['num_heads'])

                self.update_xs_given_new_xs(group['params'], xs, flatten_hat_x)
                
            elif group['group_type'] == 3:
                """Group for standard linear layer"""
                xs, grad_fs = self.get_xs_grad_fs(group['params'])
                flatten_x = torch.cat(xs, dim = 1)
                flatten_grad_f = torch.cat(grad_fs, dim = 1)
                
                flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
                zero_group_idxes = flatten_x_norm == 0.0
                
                # compute gradient of psi
                flatten_subgrad_psi = self.mixed_l1_l2_subgrad_psi(flatten_x, flatten_grad_f, group['lmbda'])
                flatten_subgrad_psi = self.get_momentum_grad(group, 'momentum_buffer_half_space', group['momentum'], flatten_subgrad_psi)

                # compute trial iterate
                flatten_hat_x = self.grad_descent_update(flatten_x, group['lr'], flatten_subgrad_psi)

                # do half space projection
                flatten_hat_x = self.half_space_project(flatten_hat_x, flatten_x, group['epsilon'], group['upper_group_sparsity'])

                # fixed non_free variables
                flatten_hat_x[zero_group_idxes, ...] = 0.0

                self.update_xs_given_new_xs(group['params'], xs, flatten_hat_x)

            elif group['group_type'] == 4:
                """Group for Conv layer"""

                xs, grad_fs = self.get_xs_grad_fs(group['params'])
                flatten_x = torch.cat(xs, dim = 1)
                flatten_grad_f = torch.cat(grad_fs, dim = 1)
                
                flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
                zero_group_idxes = flatten_x_norm == 0.0

                # compute gradient of psi
                flatten_subgrad_psi = self.mixed_l1_l2_subgrad_psi(flatten_x, flatten_grad_f, group['lmbda'])
                flatten_subgrad_psi = self.get_momentum_grad(group, 'momentum_buffer_half_space', group['momentum'], flatten_subgrad_psi)
                
                # compute trial iterate
                flatten_hat_x = self.grad_descent_update(flatten_x, group['lr'], flatten_subgrad_psi)

                # do half space projection
                flatten_hat_x = self.half_space_project(flatten_hat_x, flatten_x, group['epsilon'], group['upper_group_sparsity'])

                # fixed non_free variables
                flatten_hat_x[zero_group_idxes, ...] = 0.0

                self.update_xs_given_new_xs(group['params'], xs, flatten_hat_x, shapes=group['shapes'])
            else:
                raise("some parameters are not in any group type, please check")
        
        return loss

    def compute_group_sparsity_omega(self):
        total_num_groups = torch.zeros(5) + 1e-6
        total_num_zero_groups = torch.zeros(5)

        omega = 0.0
        for group in self.param_groups:
            if group['group_type'] == 0:
                """Params that do not need update"""
                total_num_zero_groups[group['group_type']] = 0
                pass
            elif group['group_type'] == 1:
                """Params that are not included in the regularization"""
                total_num_zero_groups[group['group_type']] = 0
                pass
            elif group['group_type'] == 2:
                xs = self.get_xs(group['params'])
                flatten_x = torch.cat(xs, dim = 1)

                num_groups = flatten_x.shape[0] // group['num_heads']
                flatten_x = self.reorg_multi_head(flatten_x, num_groups, group['num_heads'])

                flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
                zero_group_idxes = flatten_x_norm == 0.0
                total_num_groups[group['group_type']] += num_groups
                total_num_zero_groups[group['group_type']] += torch.sum(zero_group_idxes).cpu()
                omega += torch.sum(flatten_x_norm)
            elif group['group_type'] == 3:
                """Group for standard linear layer"""
                xs = self.get_xs(group['params'])
                flatten_x = torch.cat(xs, dim = 1)                       

                flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
                zero_group_idxes = flatten_x_norm == 0.0
                total_num_groups[group['group_type']] += flatten_x.shape[0]
                total_num_zero_groups[group['group_type']] += torch.sum(zero_group_idxes).cpu()
                omega += torch.sum(flatten_x_norm)
            elif group['group_type'] == 4:
                xs = self.get_xs(group['params'])
                flatten_x = torch.cat(xs, dim = 1)                       

                flatten_x_norm = torch.norm(flatten_x, p=2, dim=1)
                zero_group_idxes = flatten_x_norm == 0.0
                total_num_groups[group['group_type']] += flatten_x.shape[0]
                total_num_zero_groups[group['group_type']] += torch.sum(zero_group_idxes).cpu()
                omega += torch.sum(flatten_x_norm)

        
        overall_group_sparsity = torch.sum(total_num_zero_groups) / torch.sum(total_num_groups)
        group_sparsities = total_num_zero_groups = total_num_zero_groups / total_num_groups
        return total_num_zero_groups.cpu().numpy(), total_num_groups.cpu().numpy(), group_sparsities.cpu().numpy(), overall_group_sparsity.cpu().numpy(), omega.cpu().numpy()
    
    def adapt_epsilon(self, adapt_epsilons, upper_group_sparsities, prev_group_sparsity):
        total_num_zero_groups, total_num_groups, group_sparsities, overall_group_sparsity, omega = self.compute_group_sparsity_omega()
        if overall_group_sparsity > prev_group_sparsity + 0.01:
            return False, overall_group_sparsity, None
        updated_epsilons = [0.0] * 5
        for group in self.param_groups:
            if group['group_type'] == 0:
                """Params that do not need update"""
                pass
            elif group['group_type'] == 1:
                """Params that are not included in the regularization"""
                pass
            elif group['group_type'] >= 2:
                if adapt_epsilons[group['group_type']]:
                    if group_sparsities[group['group_type']] < upper_group_sparsities[group['group_type']]:
                        if group['epsilon'] < 0.999:
                            group['epsilon'] += 0.1
                        group['epsilon'] = min(0.999, group['epsilon'])
                    else:
                        group['epsilon'] = 0.0
                    updated_epsilons[group['group_type']] = group['epsilon']
        return True, overall_group_sparsity, updated_epsilons
