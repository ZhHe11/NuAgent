from typing import Any, Dict, List


# TODO(ming): implement rw safe
class EnvStatus:
    
    def __init__(self) -> None:
        pass

    def write(self, states, obs, rews, dones, infos):
        pass

    def read(self):
        return (self.states, self.obs)


class EnvManager:
    def __init__(self):
        self.envs: Dict[str, Any] = {}
        self.status_buffer: Dict[str, EnvStatus] = {}

    def register_env(self, name: str, env: Any):
        if name in self.envs:
            raise KeyError(f"existing environment key: {name}")
        self.envs[name] = env
        env.reset()

    def episode_anlyze(self, states, obses, rews, dones, trucated, infos):
        pass

    def check_all_done(self) -> bool:
        raise NotImplementedError

    def step(self, instance_id: List[str], actions_list: List[Any]):
        for eid, actions in zip(instance_id, actions_list):
            states, obses, rews, dones, trucated, infos = self.envs[eid].step(actions)
            # TODO(ming): record episode information here
            self.episode_anlyze(states, obses, rews, dones, trucated, infos)
            # lock it here
            self.status_buffer[eid].write(states, obses, rews, dones, trucated)
    
    def get_env_states_and_obs(self, instance_ids: List[str]):
        states_list = []
        observations_list = []
        for eid in instance_ids:
            states, obses  = self.status_buffer[eid].read()
            # TODO(ming): do seriealization here
            states_list.append(states)
            observations_list.append(obses)
        return states_list, observations_list
