import torch
import numpy as np
from params_proto import PrefixProto
from torch import optim

class EIPO_Args(PrefixProto):
    alpha_lr = 0.01
    alpha_g_clip = 1.0
    alpha_max_clip = None
    alpha_min_clip = -0.99

class LagrangianMultiplier:
    def __init__(self, alpha, lmbd, use_hepo, use_eps, alpha_lr, **kwargs):
        self.lmbd = lmbd
        self.use_hepo = use_hepo
        self.use_eps = use_eps
        self.model_list = ['hepo', 'ref']
        self.model_type_c = {'hepo': 'ref', 'ref': 'hepo'}
        self.rollout_policy = 'hepo'
        self.alpha = {}
        self.opt = {}
        self.alpha_grad = {}
        self.grad_buf = {}
        self.update_eipo = True
        for model_type in self.model_list:
            self.grad_buf[model_type] = []
            self.alpha[model_type] = torch.Tensor([alpha])
            self.alpha[model_type].requires_grad_(True)
            self.opt[model_type] = optim.Adam([self.alpha[model_type]], \
                                              lr=alpha_lr)

    def _update_record(self, rewards, advantages, model_type):
        rewards, advantages = rewards.clone(), advantages.clone()
        half_bsz = len(advantages) // 2
        if model_type == 'hepo':
            advantages = advantages[half_bsz:]
            rewards = rewards[:half_bsz]
        else:
            advantages = advantages[:half_bsz]
            rewards = rewards[half_bsz:]

        self.grad_buf[model_type].append(-advantages.mean().item())
        if len(self.grad_buf[model_type]) > 8:
            self.grad_buf[model_type] = self.grad_buf[model_type][1:]
        
    @ torch.no_grad()
    def compute_advantages(self, advantages, advantages_int, 
                           rewards, model_type=None):
        if self.use_hepo:
            advantages_mixed = advantages + self.lmbd / (1 + self.alpha['hepo'].item()) * advantages_int
            if model_type is not None and self.use_hepo:
                self._update_record(rewards, advantages, model_type)
            
            advantages_ref = advantages_int
            return advantages_mixed, advantages_ref
        else:
            advantages_mixed = advantages + self.lmbd * advantages_int
            return advantages_mixed

    @ torch.no_grad()
    def _update_alpha_values(self, model_type):
        grad = torch.Tensor(self.grad_buf[model_type]).clone()
        grad = grad - torch.Tensor(self.grad_buf[self.model_type_c[model_type]]).clone()
        grad = grad.median()
        self.alpha_grad[model_type] = torch.Tensor([grad])
        self.alpha[model_type].grad = torch.clamp(self.alpha_grad[model_type], 
                                    min=-EIPO_Args.alpha_g_clip, 
                                    max=EIPO_Args.alpha_g_clip)
        self.opt[model_type].step()
        self.alpha[model_type].clamp_(EIPO_Args.alpha_min_clip, EIPO_Args.alpha_max_clip) 

    def update_alpha_values(self, eps_leng=None, max_eps_leng=None):
        for model_type in self.model_list:
            self._update_alpha_values(model_type)