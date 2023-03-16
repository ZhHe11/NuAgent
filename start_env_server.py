"""Scripts for starting an environment server."""

import logging

from core.env_server import serve


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Environment server starter.")
    parser.add_argument("-p", "--port", type=str, help="service port.", default="50051")
    parser.add_argument(
        "--max-workers", type=int, help="maximum of workers.", default=10
    )

    args = parser.parse_args()
    logging.basicConfig()

    serve(args.port, args.max_workers)
