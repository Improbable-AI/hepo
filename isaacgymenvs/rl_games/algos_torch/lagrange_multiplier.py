import torch
import numpy as np
from params_proto import PrefixProto
from torch import optim

class HEPO_Args(PrefixProto):
    alpha_lr = 0.001
    alpha_g_clip = 0.02
    alpha_max_clip = None
    alpha_min_clip = 0.0
    alpha_bsz = 8

class LagrangianMultiplier:
    def __init__(self, alpha, lmbd, use_hepo, use_switch, ref_scheme, alpha_lr, **kwargs):
        self.lmbd = lmbd
        self.use_hepo = use_hepo
        self.use_switch = use_switch
        self.ref_scheme = ref_scheme
        self.switch_count = 0
        self.model_list = ['hepo', 'ref']
        self.model_type_c = {'hepo': 'ref', 'ref': 'hepo'}
        self.rollout_policy = 'hepo'
        self.alpha = {}
        self.opt = {}
        self.alpha_grad = {}
        self.grad_buf_pi = {}
        self.grad_buf_ref = {}
        for model_type in self.model_list:
            self.grad_buf_pi[model_type] = []
            self.grad_buf_ref[model_type] = []
            self.alpha[model_type] = torch.Tensor([alpha])
            self.alpha[model_type].requires_grad_(True)
            self.opt[model_type] = optim.Adam([self.alpha[model_type]], \
                                              lr=alpha_lr)

    @ torch.no_grad()
    def count_performance(self):
        if self.alpha_grad['hepo'] < 0:
            self.switch_count += 1
        else:
            self.switch_count = 0
        if self.switch_count >= HEPO_Args.alpha_bsz:
            target_policy = self.model_type_c[self.rollout_policy]            
            print(f'switch {self.rollout_policy} to {target_policy}!')
            self.rollout_policy = target_policy
            self.switch_count = 0
            for model_type in self.model_list:
                self.grad_buf_pi[model_type] = []
        
    @ torch.no_grad()
    def compute_advantages(self, advantages, advantages_int, 
                           rewards, model_type=None):
        if self.use_hepo:
            beta = self.alpha['hepo'].item() / (1 + self.alpha['hepo'].item())
            advantages_mixed = beta * advantages + self.lmbd * (1 - beta) * advantages_int

            if self.ref_scheme == 'human':
                advantages_ref = advantages_int
            elif self.ref_scheme == 'sparse':
                advantages_ref = advantages
            elif self.ref_scheme == 'const_lmbd':
                advantages_ref = advantages + self.lmbd * advantages_int
            elif self.ref_scheme == 'hepo':
                beta = self.alpha['ref'].item() / (1 + self.alpha['ref'].item())
                advantages_ref = beta * advantages + self.lmbd * (1 - beta) * advantages_int
            return advantages_mixed, advantages_ref
        else:
            advantages_mixed = advantages + self.lmbd * advantages_int
            return advantages_mixed

    def update_alpha_values(self, advantages_dict):
        if not self.use_switch:
            half_bsz = advantages_dict['hepo'].shape[1] // 2
            for model_type in advantages_dict:
                self.grad_buf_pi[model_type].append(advantages_dict[model_type][:, :half_bsz].mean().item())
                self.grad_buf_ref[model_type].append(advantages_dict[model_type][:, half_bsz:].mean().item())
                if len(self.grad_buf_pi[model_type]) > HEPO_Args.alpha_bsz:
                    self.grad_buf_pi[model_type] = self.grad_buf_pi[model_type][1:]
                if len(self.grad_buf_ref[model_type]) > HEPO_Args.alpha_bsz:
                    self.grad_buf_ref[model_type] = self.grad_buf_ref[model_type][1:]
            for model_type in self.model_list:
                self._update_alpha_values(model_type)
        else:
            for model_type in self.model_list:
                if self.rollout_policy == 'hepo':
                    self.grad_buf_pi[model_type].append(advantages_dict[model_type].mean().item())
                else:
                    self.grad_buf_pi[model_type].append(-advantages_dict[model_type].mean().item())
                if len(self.grad_buf_pi[model_type]) > HEPO_Args.alpha_bsz:
                    self.grad_buf_pi[model_type] = self.grad_buf_pi[model_type][1:]
            self._update_alpha_values('hepo')
            self.count_performance()

    def _update_alpha_values(self, model_type):
        self.opt[model_type].zero_grad() 
        if model_type == 'hepo':
            grads = torch.Tensor(self.grad_buf_pi['hepo']) + torch.Tensor(self.grad_buf_pi['ref'])
            self.alpha_grad[model_type] = torch.median(0.5 * grads).unsqueeze(0)
        else:
            grads = torch.Tensor(self.grad_buf_ref['hepo']) + torch.Tensor(self.grad_buf_ref['ref'])
            self.alpha_grad[model_type] = torch.median(-0.5 * grads).unsqueeze(0)

        self.alpha[model_type].grad = torch.clamp(self.alpha_grad[model_type], 
                                    min=-HEPO_Args.alpha_g_clip, 
                                    max=HEPO_Args.alpha_g_clip)
        self.opt[model_type].step()
        self.alpha[model_type].data.clamp_(HEPO_Args.alpha_min_clip, HEPO_Args.alpha_max_clip)