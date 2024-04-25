from application.metra.cli import main
from application.metra.cmd_utils import get_command_parser


if __name__ == "__main__":
    args = get_command_parser()
    main(args)

# PYTHONPATH=. python examples/metra_gcrl.py --eval-plot-axis -50 50 -50 50 --use-wandb 0
