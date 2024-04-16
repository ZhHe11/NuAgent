from application.hilp.cli import main, get_command_parser

if __name__ == "__main__":
    args = get_command_parser()
    main(args)


# run as:
# docker run -d --gpus device=0 -v /home/zhouming/projects/universal-agent/:/hilp -v /home/zhouming/.d4rl:/root/.d4rl --name gcrl hilp /bin/bash -c "PYTHONPATH=. python example/hilp_gcrl.py --env_name antmaze-large-diverse-v2 --use_rnd --device cuda"