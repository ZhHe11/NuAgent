from argparse import ArgumentParser, Namespace


def command_args() -> Namespace:
    parser = ArgumentParser(description="A3C for Gym control")
    parser.add_argument(
        "--lr",
        type=float,
        default=2.5e-4,  # try LogUniform(1e-4.5, 1e-3.5)
        help="learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="worker discount factor for rewards",
    )
    parser.add_argument(
        "--llambda", type=float, default=0.95, help="parameter for GAE (worker only)"
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (also called beta)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.25,
        help="worker value loss coefficient",
    )
    parser.add_argument("--lr-decay", type=bool, default=True)
    parser.add_argument(
        "--eps-clip", type=float, default=0.1, help="epsilon for clipping in PPO"
    )
    parser.add_argument(
        "--dual-clip", type=float, default=0, help="whether use dual clip in PPO"
    )
    parser.add_argument(
        "--value-clip",
        action="store_true",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--repeat", type=int, default=4, help="replicas for training over a batch."
    )
    parser.add_argument(
        "--norm-adv", action="store_true", help="whether normalize advantage"
    )
    parser.add_argument(
        "--reward-norm", action="store_true", help="whether enable reward normalization"
    )

    parser.add_argument(
        "--recompute-adv",
        action="store_true",
        help="whether recompute advantage for each replica",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5, help="value loss coefficient"
    )
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="how many training processes to use",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="number of forward steps to collect data",
    )
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=1000000,
        help="maximum length of an episode",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        help="environment to train on",
    )
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--master-addr", default="localhost")
    parser.add_argument("--master-port", default="29500")
    parser.add_argument("--optimizer", default="sgd")
    parser.add_argument("--task-type", default="gym_control")
    parser.add_argument("--use-lstm", action="store_true")
    parser.add_argument("--device-idx", default=0, type=int)

    return parser.parse_args()
