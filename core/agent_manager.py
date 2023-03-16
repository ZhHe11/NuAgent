from typing import Dict, Any, List

import gym

from core.agent import Agent


class AgentManager:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}

    def register(self, name: str, agent: Agent):
        raise NotImplementedError
    
    def coordinates(self, env_states: List[Any], observations: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Implement agent coordination here. Return the original observations by default.

        Args:
            env_states (Any): A list of environment states.
            observations (Dict[str, List[Any]]): A dict of agent observations.

        Returns:
            Dict[str, List[Any]]: A dict of agent observations, maybe coordinated.
        """
        return observations

    def step(self, states: List[Any], observations: Dict[str, List[Any]]):
        # merge agent observations
        actions = {}
        observations = self.coordinates(states, observations)
        for agent_id, obs_list in observations.items():
            actions[agent_id] = self.agents[agent_id].act(obs_list)
        return actions
