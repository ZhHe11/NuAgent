from application.openai_gym import GymAgent

from uniagent.core.env_client import EnvClient, run
from uniagent.policy_wrapper.tianshou import (
    policy_func_wrapper as tianshou_pfunc_wrapper,
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Start environment client for classic control tasks."
    )

    args = parser.parse_args()

    # load policy
    run(
        hostname=args.hostname,
        port=args.port,
        agent_policy_mapping={"default": "default"},
        policies={"default": tianshou_pfunc_wrapper(policy)},
        env_id=args.env_id,
        agent_cls=GymAgent,
        eval_episodes=args.eval_episodes,
        max_episode_steps=200,
    )
