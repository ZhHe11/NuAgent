"""Scripts for starting an agent server."""


from concurrent import futures

import logging
import grpc

from core.agent_server import AgentServer
from service import agent_server_pb2_grpc


def serve(port: str = "50052", max_workers: int = 10):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    agent_server_pb2_grpc.add_AgentServerServicer_to_server(AgentServer(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Agent server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Agent server starter.")
    parser.add_argument("-p", "--port", type=str, help="service port.", default="50052")
    parser.add_argument(
        "--max-workers", type=int, help="maximum of workers.", default=10
    )

    args = parser.parse_args()
    logging.basicConfig()

    serve(args.port, args.max_workers)
