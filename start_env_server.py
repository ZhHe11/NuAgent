"""Scripts for starting an environment server."""

from concurrent import futures

import logging
import grpc

from core.env_server import EnvServer, env_server_pb2_grpc


def serve(port: str = "50051", max_workers: int = 10):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    env_server_pb2_grpc.add_EnvServerServicer_to_server(EnvServer(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Environment server started, listening on " + port)
    server.wait_for_termination()


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
