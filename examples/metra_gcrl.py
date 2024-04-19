from application.metra.cli import main
from application.metra.cmd_utils import get_command_parser


if __name__ == "__main__":
    args = get_command_parser()
    main(args)
