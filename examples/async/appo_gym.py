import os
import shutup

shutup.please()

import torch
import torch.multiprocessing as mp

from uniagent.envs.utils import make_env_wrapper
from uniagent.trainers.parameter_server import run_parameter_server

from application.a3c_gym.cli import run_worker, get_actor_critic_cls
from application.appo_gym.async_agent import AsyncAgent
from application.appo_gym.cli import command_args


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""  # if not torch.cuda.is_available() else "0"
    mp.set_start_method("spawn")

    args = command_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    args.device = (
        torch.device(f"cuda:{args.device_idx}" if torch.cuda.is_available() else "cpu")
        if args.use_cuda
        else torch.device("cpu")
    )

    torch.manual_seed(args.seed)

    env = make_env_wrapper(args)()

    print(
        f"env: {args.env_name}\nobservation_space: {env.observation_space}\naction_space: {env.action_space}"
    )

    processes = []

    counter = mp.Value("i", 0)
    lock = mp.Lock()

    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join("runs/test", current_time + "_" + socket.gethostname())
    ps_name = "parameter_server"
    num_processes = args.num_workers + 2

    p = mp.Process(target=run_parameter_server, args=(0, num_processes, ps_name))
    p.start()
    processes.append(p)

    for rank in range(1, num_processes):
        p = mp.Process(
            target=run_worker,
            args=(
                args,
                rank,
                num_processes,
                ps_name,
                get_actor_critic_cls(args),
                env.observation_space,
                env.action_space,
                counter,
                lock,
                log_dir,
                AsyncAgent,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
