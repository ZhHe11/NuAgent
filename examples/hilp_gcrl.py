from application.hilp.cli import main, get_command_parser

if __name__ == "__main__":
    args = get_command_parser()
    main(args)


# run as:
# docker run -d --gpus device=0 -v /home/zhouming/projects/universal-agent/:/hilp -v /home/zhouming/.d4rl:/root/.d4rl --name gcrl hilp /bin/bash -c "PYTHONPATH=. python examples/hilp_gcrl.py --env-name antmaze-large-diverse-v2 --use-rnd 1 --device cuda"
