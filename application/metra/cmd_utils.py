from typing import Sequence
from argparse import ArgumentParser, Namespace

import os

from uniagent.utils.wandb import default_wandb_config


def get_exp_name(args: Namespace, global_start_time: int):
    exp_name = ""
    exp_name += f"sd{args.seed:03d}_"
    if "SLURM_JOB_ID" in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if "SLURM_PROCID" in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name_prefix = exp_name
    if "SLURM_RESTART_COUNT" in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    exp_name += f"{global_start_time}"

    exp_name += "_" + args.env_name
    exp_name += "_" + args.algo

    return exp_name, exp_name_prefix


EXP_DIR = "exp"
if os.environ.get("START_METHOD") is not None:
    START_METHOD = os.environ["START_METHOD"]
else:
    START_METHOD = "spawn"


def get_log_dir(args: Namespace):
    exp_name, exp_name_prefix = get_exp_name()
    assert len(exp_name) <= os.pathconf("/", "PC_NAME_MAX")
    # Resolve symlinks to prevent runs from crashing in case of home nfs crashing.
    log_dir = os.path.realpath(os.path.join(EXP_DIR, args.run_group, exp_name))
    assert not os.path.exists(log_dir), f"The following path already exists: {log_dir}"

    return log_dir


import sys
import random
import numpy as np
import warnings


def set_seed(seed):
    """Set the process-wide random seed.

    Args:
        seed (int): A positive integer

    """
    seed %= 4294967294
    # pylint: disable=global-statement
    global seed_
    global seed_stream_
    seed_ = seed
    random.seed(seed)
    np.random.seed(seed)
    if "tensorflow" in sys.modules:
        raise RuntimeError("Do not support tensorflow")
    if "torch" in sys.modules:
        warnings.warn(
            "Enabeling deterministic mode in PyTorch can have a performance "
            "impact when using GPU."
        )
        import torch  # pylint: disable=import-outside-toplevel

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_command_parser(args: Sequence[str] = None):
    parser = ArgumentParser("Training METRA")

    parser.add_argument("--env-name", type=str, help="environment name", required=True)
    parser.add_argument("--width", type=int, default=200, help="window size, the width")
    parser.add_argument(
        "--height", type=int, default=200, help="window size, the height"
    )

    parser.add_argument(
        "--save-dir", type=str, default="exp/", help="experiment logging directory"
    )
    parser.add_argument("--restore-path", type=str, default=None)
    parser.add_argument(
        "--run-group", type=str, default="debug", help="naming experiment group"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--num-video-episodes", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100000)
    parser.add_argument("--save-interval", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--total-steps", type=int, default=1000000)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-hidden-dim", type=int, default=512)
    parser.add_argument("--value-num-layers", type=int, default=3)
    parser.add_argument("--actor-hidden-dim", type=int, default=512)
    parser.add_argument("--actor-num-layers", type=int, default=3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--expectile", type=float, default=0.95)
    parser.add_argument("--use-layer-norm", type=int, default=1)
    parser.add_argument("--option-dim", type=int, default=32)
    parser.add_argument("--skill-expectile", type=float, default=0.9)
    parser.add_argument("--skill-temperature", type=float, default=10.0)
    parser.add_argument("--skill-discount", type=float, default=0.99)
    parser.add_argument("--p-currgoal", type=float, default=0.0)
    parser.add_argument("--p-trajgoal", type=float, default=0.625)
    parser.add_argument("--p-randomgoal", type=float, default=0.375)

    parser.add_argument("--planning-num-recursions", type=int, default=0)
    parser.add_argument("--planning-num_states", type=int, default=50000)
    parser.add_argument("--planning-num-knns", type=int, default=50)

    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--p-aug", type=float, default=None)
    parser.add_argument("--use-rnd", type=int, default=0)

    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args(args)

    args.wandb = default_wandb_config()
    args.value_hidden_dims = tuple([args.value_hidden_dim] * args.value_num_layers)
    args.actor_hidden_dims = tuple([args.actor_hidden_dim] * args.actor_num_layers)

    return args
