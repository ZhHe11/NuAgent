from application.hilp.cli import main, get_command_parser

if __name__ == "__main__":
    args = get_command_parser()
    main(args)
