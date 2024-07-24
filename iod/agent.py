import torch
import torch.nn as nn
from collections import OrderedDict


class AgentWrapper(object):
    """Wrapper for communicating the agent weights with the sampler."""

    def __init__(self, policies):
        # assert isinstance(policies, dict) and "default_policy" in policies
        self.default_policy = policies["default_policy"]
        self.exploration_policy = policies.get("exploration_policy", None)
        self.traj_encoder = policies.get("traj_encoder", None)
        
        
    def get_param_values(self):
        param_dict = {}
        default_param_dict = self.default_policy.get_param_values()
        for k in default_param_dict.keys():
            param_dict[f"default_{k}"] = default_param_dict[k].detach()

        if self.exploration_policy:
            exploration_param_dict = self.exploration_policy.get_param_values()
            for k in exploration_param_dict.keys():
                param_dict[f"exploration_{k}"] = exploration_param_dict[k].detach()
        
        if self.traj_encoder:
            # traj_encoder_dict = self.traj_encoder.get_param_values()
            # traj_encoder_dict = OrderedDict((name, param.data) for name, param in self.traj_encoder.named_parameters())
            traj_encoder_dict = self.traj_encoder.state_dict()
            for k in traj_encoder_dict.keys():
                param_dict[f"traj_encoder_{k}"] = traj_encoder_dict[k].detach()

        return param_dict

    def set_param_values(self, state_dict):
        default_state_dict = {}
        exploration_state_dict = {}
        traj_encoder_dict = {}

        for k, v in state_dict.items():
            k: str
            if k.startswith("default_"):
                default_state_dict[k.replace("default_", "", 1)] = v
            elif k.startswith("exploration_"):
                exploration_state_dict[k.replace("exploration_", "", 1)] = v
            elif k.startswith("traj_encoder_"):
                traj_encoder_dict[k.replace("traj_encoder_", "", 1)] = v
            else:
                raise ValueError(f"Unknown key: {k}")
            
        self.default_policy.set_param_values(default_state_dict)
        if self.exploration_policy:
            self.exploration_policy.set_param_values(exploration_state_dict)
        if self.traj_encoder:
            self.traj_encoder.load_state_dict(traj_encoder_dict)
            # self.traj_encoder.set_param_values(traj_encoder_dict)

    def eval(self):
        self.default_policy.eval()
        if self.exploration_policy:
            self.exploration_policy.eval()
        if self.traj_encoder:
            self.traj_encoder.eval()

    def train(self):
        self.default_policy.train()
        if self.exploration_policy:
            self.exploration_policy.train()
        if self.traj_encoder:
            self.traj_encoder.train()

    def reset(self):
        self.default_policy.reset()
        # if self.exploration_policy:
        #     self.exploration_policy.reset()
        # if self.traj_encoder:
        #     self.traj_encoder.reset()



    
    
    
    

















