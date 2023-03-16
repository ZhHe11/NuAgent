from typing import Any, Dict, List, Tuple

import uuid
import copy
import gym

from readerwriterlock import rwlock


class EnvStatus:
    def __init__(self) -> None:
        self.state = None
        self.obs = None
        self.rews = None
        self.dones = None
        self.infos = None
        self.truncated = None
        self.marker = rwlock.RWLockFair()

    def write(self, state, obs, rews, dones, truncated, infos):
        with self.marker.gen_wlock():
            self.state = state
            self.obs = obs
            self.rews = rews
            self.dones = dones
            self.truncated = truncated
            self.infos = infos

    def read(self):
        with self.marker.gen_rlock():
            state = copy.deepcopy(self.state)
            obs = copy.deepcopy(self.obs)
        return state, obs


class EnvManager:
    """
    An environment manager is responsible for the environment creation, stepping, and status analysis.
    """

    def __init__(self):
        self.envs: Dict[str, Any] = {}
        self.status_buffer: Dict[str, EnvStatus] = {}

    def register_env(self, name: str, env: Any):
        """
        Register existing environment instance. If the given name is already used, raise KeyError

        Args:
            name: Registered instance id.
            env: Environment instance.
        """
        if name in self.envs:
            raise KeyError(f"existing environment key: {name}")
        self.envs[name] = env
        env.reset()

    def create_envs(
        self, env_id: str, env_num: int, max_episode_steps: int
    ) -> List[str]:
        """Create a list of registered environment class that is named as `env_id`.
        The number is determined by `env_num`.

        Args:
            env_id: Registered environment id.
            env_num: Number of new added environment instances.
            max_episode_steps: Maximum length of an episode.

        Returns:
            List[str]: A list of new added instance ids.
        """

        instance_ids = []
        for _ in range(env_num):
            env_name = str(uuid.uuid4())
            while env_name in self.envs:
                env_name = str(uuid.uuid4())
            env = gym.make(env_id, max_episode_steps=max_episode_steps)
            self.envs[env_name] = env
            self.status_buffer[env_name] = EnvStatus()
            instance_ids.append(env_name)
            observation = env.reset()
            if hasattr(env, "get_state"):
                state = env.get_state()
            else:
                state = None
            self.status_buffer[env_name].write(
                state=state,
                obs=observation,
                rews=None,
                dones=False,
                truncated=False,
                infos=None,
            )
        return instance_ids

    def episode_analyze(self, state, obses, rews, dones, trucated, infos):
        pass

    def step(self, instance_id: List[str], actions_list: List[Dict[str, Any]]):
        """ Multiple environment stepping. Given a list of environment instance ids, and the corresponding \
        actions list.

        Args:
            instance_id: A list of environment instance ids.
            actions_list: A list of agent actions.
        """

        done_envs = []
        for eid, actions in zip(instance_id, actions_list):
            env = self.envs[eid]
            observations, rews, dones, truncated, infos = env.step(actions)
            if hasattr(env, "get_state"):
                state = env.get_state()
            else:
                state = None
            # TODO(ming): please make sure the single-agent environment behaves like multi-agent in keys
            if isinstance(dones, bool) and dones:
                done_envs.append(eid)
            elif isinstance(dones, Dict):
                is_env_done = dones["__all__"] or all(dones.values())
                if is_env_done:
                    done_envs.append(eid)
            # TODO(ming): record episode information here
            self.episode_analyze(state, observations, rews, dones, truncated, infos)
            self.status_buffer[eid].write(
                state, observations, rews, dones, truncated, infos
            )
        return {"dones": done_envs}

    def get_env_states_and_obs(
        self, instance_ids: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Retrieve a list of state-and-observation tuples, each is corresponding to an element in instance_ids.

        Args:
            instance_ids: A list of environment instance ids.

        Returns:
            Tuple[List[Any], List[Any]]: A tuple of states list and observation lis.
        """

        states_list = []
        observations_list = []
        for eid in instance_ids:
            states, observations = self.status_buffer[eid].read()
            states_list.append(states)
            observations_list.append(observations)
        return states_list, observations_list
