from typing import Sequence
from argparse import ArgumentParser, Namespace

import os

from uniagent.utils.wandb import default_wandb_config

import datetime

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


def get_log_dir(args: Namespace, EXP_DIR=EXP_DIR):
    exp_name, exp_name_prefix = get_exp_name(args, int(datetime.datetime.now().timestamp()))
    assert len(exp_name) <= os.pathconf("/", "PC_NAME_MAX")
    # Resolve symlinks to prevent runs from crashing in case of home nfs crashing.
    log_dir = os.path.realpath(os.path.join(EXP_DIR, args.run_group, exp_name))
    assert not os.path.exists(log_dir), f"The following path already exists: {log_dir}"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


import sys
import random
import numpy as np
import warnings
import torch


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

    # ---------------logging settings----------------
    parser.add_argument(
        "--run-group", type=str, default="debug", help="naming experiment group"
    )
    parser.add_argument(
        "--save-dir", type=str, default="exp/", help="experiment logging directory"
    )

    parser.add_argument(
        "--video_log_dir", type=str, default="exp/", help="experiment logging directory"
    )

    parser.add_argument(
        "--debug", type=int, default=1, help="debug mode or not, if debug, will save the information of the test traj."
    )


    parser.add_argument(
        "--use-wandb", type=int, default=1, help="enabling wandb or not"
    )
    parser.add_argument("--render", type=int, default=1)

    # ------------------general exp settings--------------
    parser.add_argument("--algo", type=str, default="metra", choices=["metra", "dads"])
    parser.add_argument(
        "--normalizer-type",
        type=str,
        default="off",
        choices=["off", "preset"],
        help="indicates ",
    )
    parser.add_argument("--restore-path", type=str, default=None)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--save-interval", type=int, default=5000)

    # ------------------task settings--------------
    parser.add_argument(
        "--env-name",
        type=str,
        default="maze",
        choices=[
            "maze",
            "half_cheetah",
            "ant",
            "dmc_cheetah",
            "dmc_quadruped",
            "dmc_humanoid",
            "kitchen",
        ],
        help="environment name",
    )
    parser.add_argument("--frame-stack", type=int, default=None)
    parser.add_argument("--max-path-length", type=int, default=200)

    # ---------------------training resource settings----------
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed, defaults to magic number 42"
    )
    parser.add_argument("--n-thread", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="default is cpu, otherwise random cuda when it is available",
    )

    # -------------------training hyper-parameters settings----------
    parser.add_argument("--n-epochs", type=int, default=1000000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,              
        help="batch size for training",
    )
    parser.add_argument(
        "--use-inner-product",
        type=int,
        default=1,
        choices={0, 1},
        help="whether use inner product for reward calculation",
    )
    parser.add_argument(
        "--discrete-option",
        type=int,
        default=1,
        choices={0, 1},
        help="whether the option representation is a discrete mode or not",
    )
    parser.add_argument(
        "--unit-length",
        type=int,
        choices={0, 1},
        help="activated for only continuous options",
    )
    parser.add_argument("--common-lr", type=float, default=1e-4)
    parser.add_argument("--lr-op", type=float, default=None)
    parser.add_argument("--lr-te", type=float, default=None)
    parser.add_argument("--lr-dual", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--dual-reg", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dual-lam", type=float, default=30)
    parser.add_argument("--dual-slack", type=float, default=1e-3)
    parser.add_argument(
        "--dual-dist", type=str, default="one", choices=["l2", "s2_from_s", "one"]
    )

    # ----------- sac training settings
    parser.add_argument("--sac-tau", type=float, default=5e-3)
    parser.add_argument("--sac-lr-q", type=float, default=None)
    parser.add_argument("--sac-lr-a", type=float, default=None)
    parser.add_argument("--sac-discount", type=float, default=0.99)
    parser.add_argument("--sac-scale-reward", type=float, default=1.0)
    parser.add_argument("--sac-target-coef", type=float, default=1.0)
    parser.add_argument("--buffer-size", type=int, default=300000)

    # ----------------model settings-----------------
    parser.add_argument(
        "--option-dim", type=int, default=2, help="option dimension size"
    )
    parser.add_argument(
        "--dim_option", type=int, default=2, help="option dimension size"
    )
    parser.add_argument("--value-hidden-dim", type=int, default=512)
    parser.add_argument("--value-num-layers", type=int, default=3)
    parser.add_argument("--actor-hidden-dim", type=int, default=512)
    parser.add_argument("--actor-num-layers", type=int, default=3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--use-layer-norm", type=int, default=1)
    parser.add_argument(
        "--use-dist-predictor",
        type=int,
        default=0,
        choices={0, 1},
        help="use distance predictor or not",
    )
    parser.add_argument(
        "--use-option-planner",
        type=int,
        default=0,
        choices={0, 1},
        help="enable option planner model or not",
    )

    # --------------evaluation settings----------
    parser.add_argument(
        "--num-eval-trajectories",
        type=int,
        default=48,
        help="how many episodes you wanna evaluation for each time",
    )
    parser.add_argument("--eval-plot-axis", type=float, default=None, nargs="*")

    parser.add_argument("--discrete_goal", type=int, default=0, choices={0, 1})
    parser.add_argument("--discrete_option", type=int, default=0, choices={0, 1})
    # ------------for video record when evaluation-----------
    parser.add_argument(
        "--eval-record-video",
        type=int,
        default=1,
        choices={0, 1},
        help="indicating whether use video rendering for evaluation",
    )
    parser.add_argument(
        "--video-skip-frames", type=int, default=1, help="frame skipping, defaults to 4"
    )
    parser.add_argument(
        "--num-video-repeats",
        type=int,
        default=2,
        help="how many videos for saving, defaults to 2",
    )

    args = parser.parse_args(args)
    args.wandb = default_wandb_config()
    args.value_hidden_dims = tuple([args.value_hidden_dim] * args.value_num_layers)
    args.actor_hidden_dims = tuple([args.actor_hidden_dim] * args.actor_num_layers)

    args.video_log_dir = get_log_dir(args, EXP_DIR=os.path.join(EXP_DIR, "video"))
    print("[video_log_dir] : ", args.video_log_dir)

    if torch.cuda.is_available() and "cuda" not in args.device:
        args.device = "cuda"

    if args.lr_te is None:
        args.lr_te = args.common_lr

    if args.lr_op is None:
        args.lr_op = args.common_lr

    if args.sac_lr_q is None:
        args.sac_lr_q = args.common_lr

    if args.sac_lr_a is None:
        args.sac_lr_a = args.common_lr

    if args.lr_dual is None:
        args.lr_dual = args.common_lr

    return args
