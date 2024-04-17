import os
import time
import glob
import pickle
import datetime

from functools import partial
from argparse import ArgumentParser, Namespace

import numpy as np
import wandb
import tqdm
import tree

from torch import nn
from uniagent.utils.wandb import setup_wandb, default_wandb_config


from .learner import MetraAgent


def main(args: Namespace):
    raise NotImplementedError


from typing import Sequence
from application.hilp.cli import get_command_parser as hilp_command_parser


def get_command_parser(args: Sequence[str] = None):
    parser = ArgumentParser("Training HILP")

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
