import pickle
import grpc

from concurrent import futures

from service import env_server_pb2_grpc
from service import env_server_pb2

from .env_manager import EnvManager


class EnvServer(env_server_pb2_grpc.EnvServerServicer):
    def __init__(self):
        super().__init__()
        self.env_manager = EnvManager()

    def GetStateAndObs(self, request, context):
        """Handling client request for getting states and observations. Then return the
        serialized states and observations, cooperating with a list of environment instance ids.
        """

        instance_ids = request.instance_ids
        states, observations = self.env_manager.get_env_states_and_obs(instance_ids)
        # state and observation serialization here,
        #   currently we use pickle proxy.
        b_states = pickle.dumps(states)
        b_observations = pickle.dumps(observations)
        all_done = self.env_manager.check_all_done()
        return env_server_pb2.EnvReply(
            instance_ids=instance_ids,
            b_states=b_states,
            b_observations=b_observations,
            all_done=all_done,
        )

    def Step(self, request, context):
        """Handling environment stepping."""

        actions = pickle.loads(request.actions)
        self.env_manager.step(request.instance_ids, actions)

    def RequestEnvs(self, request, context):
        """Handling client request for create multiple environments. Then return
        the list of environment instance ids to the corresponding client."""

        env_id = request.env_id
        env_num = request.env_num
        # a list of instance id
        env_list = self.env_manager.create_envs(env_id, env_num)
        return env_server_pb2.EnvReply(instance_ids=env_list, all_done=False)


def serve(port: str = "50051", max_workers: int = 10):
    """Start an environment server.
    Args:
        port: Server port.
        max_workers: Maximum of workers.
    """

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    env_server_pb2_grpc.add_EnvServerServicer_to_server(EnvServer(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Environment server started, listening on " + port)
    server.wait_for_termination()
