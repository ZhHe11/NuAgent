import os
import tempfile
import wandb

from tests.GetArgparser import get_argparser


def get_exp_name():
    exp_name = ''
    exp_name += f'sd{args.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name_prefix = exp_name
    if 'SLURM_RESTART_COUNT' in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    exp_name += "test"

    exp_name += '_' + args.env
    exp_name += '_' + args.algo

    return exp_name, exp_name_prefix



args = get_argparser().parse_args()

# if 'WANDB_API_KEY' in os.environ:
wandb_output_dir = tempfile.mkdtemp()
wandb.init(group="test", name=get_exp_name()[0],
            config=vars(args), dir=wandb_output_dir)



