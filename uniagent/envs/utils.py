from .gym_control import create_gym_control
from .atari import create_atari_env


def make_env_wrapper(args):
    def make_env():
        if args.task_type == "gym_control":
            return create_gym_control(args.env_name)
        elif args.task_type == "atari":
            return create_atari_env(args.env_name, scale_obs=True)
        else:
            raise NotImplementedError

    return make_env
