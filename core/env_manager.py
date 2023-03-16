from typing import Any, Dict, List, Tuple

import uuid
import copy
import gym

from readerwriterlock import rwlock


# TODO(ming): implement rw safe
class EnvStatus:
    def __init__(self) -> None:
        self.states = None
        self.obs = None
        self.rews = None
        self.dones = None
        self.infos = None
        self.truncated = None
        self.marker = rwlock.RWLockFair()

    def write(self, states, obs, rews, dones, truncated, infos):
        with self.marker.gen_wlock():
            self.states = states
            self.obs = obs
            self.rews = rews
            self.dones = dones
            self.truncated = truncated
            self.infos = infos

    def read(self):
        with self.marker.gen_rlock():
            states = copy.deepcopy(self.states)
            obs = copy.deepcopy(self.obs)
        return states, obs


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

    def create_envs(self, env_id: str, env_num: int) -> List[str]:
        """Create a list of registered environment class that is named as `env_id`.
        The number is determined by `env_num`.

        Args:
            env_id: Registered environment id.
            env_num: Number of new added environment instances.

        Returns:
            List[str]: A list of new added instance ids.
        """

        instance_ids = []
        for _ in range(env_num):
            env_name = str(uuid.uuid4())
            while env_name in self.envs:
                env_name = str(uuid.uuid4())
            env = gym.make(env_id)
            self.envs[env_name] = env
            self.status_buffer[env_name] = EnvStatus()
            instance_ids.append(instance_ids)
        return instance_ids

    def episode_analyze(self, states, obses, rews, dones, trucated, infos):
        pass

    def check_all_done(self) -> bool:
        raise NotImplementedError

    def step(self, instance_id: List[str], actions_list: List[Dict[str, Any]]):
        """ Multiple environment stepping. Given a list of environment instance ids, and the corresponding \
        actions list.

        Args:
            instance_id: A list of environment instance ids.
            actions_list: A list of agent actions.
        """

        for eid, actions in zip(instance_id, actions_list):
            states, observations, rews, dones, truncated, infos = self.envs[eid].step(
                actions
            )
            # TODO(ming): record episode information here
            self.episode_analyze(states, observations, rews, dones, truncated, infos)
            # lock it here
            self.status_buffer[eid].write(
                states, observations, rews, dones, truncated, infos
            )

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
            # TODO(ming): do serialization here
            states_list.append(states)
            observations_list.append(observations)
        return states_list, observations_list
