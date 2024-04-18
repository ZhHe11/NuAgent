from argparse import Namespace


def make_env(args: Namespace, max_path_length: int):
    if args.env_name == "maze":
        from .maze_env import MazeEnv

        env = MazeEnv(
            max_path_length=max_path_length,
            action_range=0.2,
        )
    elif args.env_name == "half_cheetah":
        from .custom_mujoco.half_cheetah_env import HalfCheetahEnv

        env = HalfCheetahEnv(render_hw=100)
    elif args.env_name == "ant":
        from .custom_mujoco.ant_env import AntEnv

        env = AntEnv(render_hw=100)
    elif args.env_name.startswith("dmc"):
        from .custom_dmc_tasks import dmc
        from .custom_dmc_tasks.pixel_wrappers import RenderWrapper

        assert args.encoder  # Only support pixel-based environments
        if args.env_name == "dmc_cheetah":
            env = dmc.make(
                "cheetah_run_forward_color",
                obs_type="states",
                frame_stack=1,
                action_repeat=2,
                seed=args.seed,
            )
            env = RenderWrapper(env)
        elif args.env_name == "dmc_quadruped":
            env = dmc.make(
                "quadruped_run_forward_color",
                obs_type="states",
                frame_stack=1,
                action_repeat=2,
                seed=args.seed,
            )
            env = RenderWrapper(env)
        elif args.env_name == "dmc_humanoid":
            env = dmc.make(
                "humanoid_run_color",
                obs_type="states",
                frame_stack=1,
                action_repeat=2,
                seed=args.seed,
            )
            env = RenderWrapper(env)
        else:
            raise NotImplementedError
    elif args.env_name == "kitchen":
        import sys
        from .lexa.mykitchen import MyKitchenEnv

        sys.path.append("lexa")
        assert args.encoder  # Only support pixel-based environments
        env = MyKitchenEnv(log_per_goal=True)
    else:
        raise NotImplementedError


from typing import Dict


def eval_episodes(env) -> Dict[str, float]:
    raise NotImplementedError
