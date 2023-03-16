from typing import Dict, Any, List, Type

import pickle
import grpc
import gym

from service import env_server_pb2
from service import env_server_pb2_grpc

from .agent_manager import AgentManager


class EnvClient:
    def __init__(self, hostname: str, port: int):
        channel = grpc.insecure_channel(target=f"{hostname}:{port}")
        self.stub = env_server_pb2_grpc.EnvServerStub(channel)
        self.channel = channel
        self.agent_manager = AgentManager()

    def request_for_envs(self, env_id: str, env_num: int):
        return self.stub.RequestEnvs(
            env_server_pb2.VecEnvDesc(env_id=env_id, env_num=env_num)
        )

    def run(self, env_instance_ids: List[str]):
        while True:
            env_reply: env_server_pb2.EnvReply = self.stub.GetStateAndObs(
                env_server_pb2.EnvRequest(instance_ids=env_instance_ids)
            )
            if env_reply.all_done:
                break
            states = pickle.loads(env_reply.b_states)
            observations = pickle.loads(env_reply.b_observations)
            actions = self.agent_manager.act(states, observations)
            actions = pickle.dumps(actions)
            self.stub.Step(
                env_server_pb2.EnvRequest(
                    instance_ids=env_reply.instance_ids, actions=actions
                )
            )
        self.channel.close()


def run(
    hostname: str,
    port: int,
    agent_policy_mapping: Dict[str, str],
    policies: Dict[str, Any],
    env_id: str,
    agent_cls: Type,
    eval_episodes: int = 3,
):
    client = EnvClient(hostname, port)

    env = gym.make(env_id)

    if not hasattr(env, "agents"):
        # single agent environment use 'default' as the agent name.
        agent_profiles = {"default": (env.action_space, env.observation_space)}
    else:
        assert hasattr(
            env, "agents"
        ), "for multi-agent environment, `agents` should be an property."
        assert callable(
            env.action_space
        ), "for multi-agent environment, the action_space should be callable."
        assert callable(
            env.observation_space
        ), "for multi-agent environment, the observation_space should be callable."
        agent_profiles = {
            k: (env.action_space(k), env.observation_space(k)) for k in env.agents
        }

    for agent_id, (act_space, obs_space) in agent_profiles.items():
        cid = agent_policy_mapping[agent_id]
        policy = policies[cid]
        client.agent_manager.register(
            name=agent_id, agent=agent_cls(obs_space, act_space, policy)
        )

    env.close()
    # request for environments creating
    env_instance_ids = client.request_for_envs(env_id, env_num=eval_episodes)
    client.run(env_instance_ids)
