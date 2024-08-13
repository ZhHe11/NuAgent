import torch
import torch.nn as nn
from collections import OrderedDict
import copy
import torch


class AgentWrapper(object):
    """Wrapper for communicating the agent weights with the sampler."""

    def __init__(self, policies):
        for k, v in policies.items():
            setattr(self, k, v)

    def vec_norm(self, vec):
        return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)
        
    @torch.no_grad()
    def gen_z(self, sub_goal, obs, device="cpu", ret_emb: bool = False):
        traj_encoder = self.target_traj_encoder.to(device)
        goal_z = traj_encoder(sub_goal).mean
        target_cur_z = traj_encoder(obs).mean

        z = self.vec_norm(goal_z - target_cur_z)
        if ret_emb:
            return z, target_cur_z, goal_z
        else:
            return z
        
    @torch.no_grad()
    def gen_z_phi_g(self, phi_g, obs, device='cpu'):
        traj_encoder = self.target_traj_encoder.to(device)
        goal_z = phi_g
        target_cur_z = traj_encoder(obs).mean
    
        z = self.vec_norm(goal_z - target_cur_z)
        return z
        
        
    def get_param_values(self):
        param_dict = {}
        for k, v in self.__dict__.items():
            param_dict[k] = v.state_dict() if hasattr(v, "state_dict") else v.get_param_values()

        return param_dict

    def set_param_values(self, state_dict):
        for k, v in state_dict.items():
            net = getattr(self, k)
            net.load_state_dict(v)

    def eval(self):
        for v in self.__dict__.values():
            v.eval()

    def train(self):
        for v in self.__dict__.values():
            v.train()

    def reset(self):
        self.default_policy.reset()



def copy_init_policy(policy, qf1, qf2):
    policy = copy.deepcopy(policy)
    qf1 = copy.deepcopy(qf1)
    qf2 = copy.deepcopy(qf2)
    target_qf1 = copy.deepcopy(qf1)
    target_qf2 = copy.deepcopy(qf2)

    return policy, qf1, qf2, target_qf1, target_qf2
    
    
    
    
     
    
    
    

















