from argparse import Namespace
from .consistent_normalized_env import consistent_normalize
from .plot import get_normalizer_preset
import gym


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

    if args.frame_stack is not None:
        from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper

        env = FrameStackWrapper(env, args.frame_stack)

    normalizer_type = args.normalizer_type
    normalizer_kwargs = {}

    if normalizer_type == "off":
        env = consistent_normalize(env, normalize_obs=False, **normalizer_kwargs)
    elif normalizer_type == "preset":
        normalizer_name = args.env_name
        normalizer_mean, normalizer_std = get_normalizer_preset(
            f"{normalizer_name}_preset"
        )
        env = consistent_normalize(
            env,
            normalize_obs=True,
            mean=normalizer_mean,
            std=normalizer_std,
            **normalizer_kwargs,
        )

    return env
