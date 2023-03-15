from typing import Dict

import gym

from core.agent import Agent


class AgentManager:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}

    def register(self, name: str, agent: Agent):
        raise NotImplementedError

    def run(self):
        pass
