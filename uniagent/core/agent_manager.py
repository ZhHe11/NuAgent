from typing import Dict, Any, List, Union

import gym

from core.agent import Agent


class AgentManager:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}

    def register(self, name: str, agent: Agent):
        assert name not in self.agents, f"detected existing agent={name}"
        self.agents[name] = agent

    def coordinates(
        self,
        env_states: List[Any],
        observations: Union[List[Any], Dict[str, List[Any]]],
    ) -> Dict[str, List[Any]]:
        """Implement agent coordination here. Return the original observations by default.

        Args:
            env_states (Any): A list of environment states.
            observations (Dict[str, List[Any]]): A dict of agent observations.

        Returns:
            Union[List[Any], Dict[str, List[Any]]]: A dict of agent observations, maybe coordinated.
        """
        return observations

    def act(
        self, states: List[Any], observations: Union[List[Any], Dict[str, List[Any]]]
    ):
        # merge agent observations
        observations = self.coordinates(states, observations)
        if isinstance(observations, Dict):
            actions = {}
            for agent_id, obs_list in observations.items():
                actions[agent_id] = self.agents[agent_id].act(obs_list)
        elif isinstance(observations, List):
            actions = self.agents["default"].act(observations)
        else:
            raise TypeError(f"Unexpected observation type: {type(observations)}")
        return actions
