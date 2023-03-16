from typing import Dict, Any, Tuple, Callable, List

import logging
import grpc
import gym
import numpy as np
import torch

from core.service import env_server_pb2
from core.service import env_server_pb2_grpc

from core.agent_manager import AgentManager, Agent


class Policy:

    def __init__(self):
        pass

    def compute_action(self, observation: torch.Tensor, context: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class GymAgent(Agent):

    def __init__(self, obs_space: gym.Space, act_space: gym.Space, policy: Any, obs_preprocessor: Callable[[Any, gym.Space], np.array] = None, act_preprocessor: Callable[[np.array, gym.Space], Any] = None):
        super().__init__(obs_space, act_space, policy, obs_preprocessor, act_preprocessor)

    def act(self, observation: List[Any]):
        raise NotImplementedError


class EnvClient:

    def __init__(self, hostname: str, port: int):
        channel = grpc.insecure_channel(target=f"{hostname}:{port}")
        self.stub = env_server_pb2_grpc.EnvServerStub(channel)
        self.channel = channel
        self.agent_manager = AgentManager()

    def request_for_envs(self, env_id: int, env_num: int):
        return self.stub.RequestEnvs(env_server_pb2.VecEnvDesc(env_id=env_id, env_num=env_num))

    def run(self, env_instance_ids: List[str]):
        while True:
            env_reply: env_server_pb2.EnvReply = self.stub.GetStateAndObs(env_server_pb2.EnvRequest(instance_ids=env_instance_ids))
            if env_reply.all_done:
                break
            actions = self.agent_manager.step(env_reply.states, env_reply.observations)
            # TODO(ming): do serielization for actions
            self.stub.Step(env_server_pb2.EnvRequest(instance_ids=env_reply.instance_ids, actions=actions))
        self.channel.close()


def run(hostname: str, port: int, agent_policy_mapping: Dict[str, str], policies: Dict[str, Policy], env_id: str, eval_episodes: int = 3):
    client = EnvClient(hostname, port)
    
    env = gym.make(env_id)

    if not hasattr(env, "agents"):
        agent_profiles = {"default": (env.action_space, env.observation_space)}
    else:
        agent_profiles = {k: (env.action_space(k), env.observation_space(k)) for k in env.agents}

    for agent_id, (act_space, obs_space) in agent_profiles.items():
        client.agent_manager.register(
            name=agent_id,
            agent=GymAgent(obs_space,act_space))
        
    env.close()
    # request for environments creating
    env_instance_ids = client.request_for_envs(env_id, env_num=eval_episodes)
    client.run(env_instance_ids)
