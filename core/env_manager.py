from typing import Any, Dict


class EnvManager:
    def __init__(self):
        self.envs: Dict[str, Any] = {}

    def register_env(self, name: str, env: Any):
        if name in self.envs:
            raise KeyError(f"existing environment key: {name}")
        self.envs[name] = env
        env.reset()

    def run(self):
        raise NotImplementedError
