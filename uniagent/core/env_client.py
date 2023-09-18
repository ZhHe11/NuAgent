import logging
from typing import Dict, Any, List, Type, Callable

import pickle
import grpc
import gym
import numpy as np

from service import env_server_pb2
from service import env_server_pb2_grpc

from core.agent import Agent
from core.agent_manager import AgentManager


class EnvClient:
    """
    A EnvClient is responsible for the coordination with EnvServer.
    """

    def __init__(self, hostname: str, port: int):
        """Construct a EnvClient instance.

        Args:
            hostname: Hostname of environment server.
            port: Port of environment server.
        """

        channel = grpc.insecure_channel(target=f"{hostname}:{port}")
        self.stub = env_server_pb2_grpc.EnvServerStub(channel)
        self.channel = channel
        self.agent_manager = AgentManager()

    def request_for_evaluation(
        self, env_id: str, env_num: int, max_episode_steps: int
    ) -> List[str]:
        """Call it after building connection with environment server, requesting `env_num` \
        environments for simulation.

        Args:
            env_id: Registered environment id.
            env_num: The number of environment instances.
            max_episode_steps: The maximum length of an episode.

        Returns:
            A list of environment instance ids.
        """

        if max_episode_steps is None:
            max_episode_steps = -1
        else:
            assert (
                max_episode_steps > 0
            ), "step limit should be larger than 0 or None for infinite."

        env_reply = self.stub.RequestEvaluation(
            env_server_pb2.EnvRequest(
                vec_env_desc=env_server_pb2.VecEnvDesc(
                    env_id=env_id, env_num=env_num, max_episode_steps=max_episode_steps
                )
            )
        )
        env_instance_ids = list(env_reply.instance_ids)
        return env_instance_ids

    def run(self, env_instance_ids: List[str]):
        """Execute simulation loop here.

        Args:
            env_instance_ids: A list of environment instance ids in the environment server.
        """

        while True:
            logging.debug("request for new observations...")
            env_reply: env_server_pb2.EnvReply = self.stub.GetStateAndObs(
                env_server_pb2.EnvRequest(instance_ids=env_instance_ids)
            )
            actual_env_instance_ids = env_reply.instance_ids
            logging.debug(
                f"got observations from remote server, num_envs={len(actual_env_instance_ids)}"
            )
            if env_reply.all_done:
                logging.debug("* simulation done for all episodes")
                break
            states = pickle.loads(env_reply.b_states)
            observations = pickle.loads(env_reply.b_observations)
            actions: Dict[str, List[Any]] = self.agent_manager.act(states, observations)
            logging.debug(f"decision results as: {actions}")
            b_actions = pickle.dumps(actions)
            env_reply = self.stub.Step(
                env_server_pb2.EnvRequest(
                    instance_ids=env_reply.instance_ids, b_actions=b_actions
                )
            )
        self.channel.close()
        logging.debug("*channel closed")


def run(
    hostname: str,
    port: int,
    agent_policy_mapping: Dict[str, str],
    policy_funcs: Dict[str, Any],
    env_id: str,
    agent_cls: Type = Agent,
    eval_episodes: int = 3,
    max_episode_steps: int = None,
    obs_preprocessor: Dict[str, Callable[[Any, gym.Space], np.ndarray]] = None,
    act_preprocessor: Dict[str, Callable[[np.ndarray, gym.Space], Any]] = None,
):
    """Execute evaluation request.

    Args:
        hostname: The hostname of the remote environment server.
        port: Port of the remote environment server.
        agent_policy_mapping: A dict that mapping from agent ids to policy ids.
        policy_funcs: A dict of policies.
        env_id: Environment id you wanna used for evaluation.
        agent_cls: Agent class type, `Agent` by default.
        eval_episodes: The rounds of evaluation.
        max_episode_steps: The maximum length of an episode, infinite by default.
        obs_preprocessor: A dict of agent observation preprocessors.
        act_preprocessor: A dict of agent action preprocessors.
    """

    client = EnvClient(hostname, port)
    logging.info(f"Connected to environment server at {hostname}:{port}")

    env = gym.make(env_id)

    if not hasattr(env, "agents"):
        # single agent environment use 'default' as the agent name.
        logging.info(f"Request environment is a single-agent case, env_id={env_id}")
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
        logging.info(f"Request environment is a multi-agent case, env_id={env_id}")
        agent_profiles = {
            k: (env.action_space(k), env.observation_space(k)) for k in env.agents
        }

    if obs_preprocessor is None:
        obs_preprocessor = dict.fromkeys(agent_profiles.keys(), None)

    if act_preprocessor is None:
        act_preprocessor = dict.fromkeys(agent_profiles.keys(), None)

    for agent_id, (act_space, obs_space) in agent_profiles.items():
        cid = agent_policy_mapping[agent_id]
        policy_func = policy_funcs[cid]
        client.agent_manager.register(
            name=agent_id,
            agent=agent_cls(
                obs_space,
                act_space,
                policy_func,
                obs_preprocessor[agent_id],
                act_preprocessor[agent_id],
            ),
        )
        logging.info(
            f"agent={agent_id} has been registered to manager successfully, with policy_id={cid}."
        )

    env.close()
    # request for environments creating
    logging.info(f"request for {eval_episodes} environments ...")
    env_instance_ids = client.request_for_evaluation(
        env_id, env_num=eval_episodes, max_episode_steps=max_episode_steps
    )
    logging.info(f"created {eval_episodes} environments, starting simulation ...")
    client.run(env_instance_ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run(
        hostname="localhost",
        port=50051,
        agent_policy_mapping={"default": "default"},
        policy_funcs={"default": None},
        env_id="CartPole-v1",
        eval_episodes=4,
        max_episode_steps=200,
    )
