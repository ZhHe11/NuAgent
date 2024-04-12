import sys
import argparse
import pytest
import os

from hilp.learner import HILPAgent
from hilp.dataset_utils import get_env_and_dataset
from hilp.cli import get_command_parser


def test_prepare():
    args = "--env-name antmaze-large-diverse-v2 --device cuda".split()
    args = get_command_parser(args)
    env, dataset, aux_env, goal_info = get_env_and_dataset(args)
