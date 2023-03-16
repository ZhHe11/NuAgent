from typing import Dict, List, Any

import copy
import logging
import pickle
import grpc

from concurrent import futures
from readerwriterlock import rwlock

from service import env_server_pb2_grpc
from service import env_server_pb2

from core.env_manager import EnvManager


def _get_client_identification(context: grpc.ServicerContext):
    peer_identification = context.peer().split(":")
    client_host = peer_identification[1]
    client_port = peer_identification[2]
    client_identification = f"{client_host}:{client_port}"
    return client_identification


class Session:
    def __init__(self, env_id: str, env_num: int, env_list: List[str]):
        self.env_id = env_id
        self.env_num = env_num
        self.env_list = env_list
        self.dead_envs = set()
        self.marker = rwlock.RWLockFair()

    def get(self, key: str):
        with self.marker.gen_rlock():
            res = copy.deepcopy(getattr(self, key))
        return res

    def _add_dead_env(self, env_ids: List[str]):
        self.dead_envs.update(env_ids)

    def update(self, key: str, value: Any):
        with self.marker.gen_wlock():
            if key == "dead_envs":
                self._add_dead_env(value)
            else:
                raise KeyError(f"unexpected key: {key}")


class EnvServer(env_server_pb2_grpc.EnvServerServicer):
    def __init__(self):
        super().__init__()
        self.env_manager = EnvManager()
        self.session_buffer: Dict[str, Session] = {}

    def GetStateAndObs(self, request, context):
        """Handling client request for getting states and observations. Then return the
        serialized states and observations, cooperating with a list of environment instance ids.
        """

        expected_instance_ids = request.instance_ids
        client_id = _get_client_identification(context)
        session = self.session_buffer[client_id]

        # filter instance_id that has been tagged as done
        dead_envs = session.get("dead_envs")
        len_all_envs = len(session.get("env_list"))
        len_dead = len(dead_envs)
        instance_ids = []
        for eid in expected_instance_ids:
            if eid in dead_envs:
                continue
            instance_ids.append(eid)

        logging.debug(f"dead envs: {dead_envs}")
        logging.info(f"task completed ratio: {len_dead}/{len_all_envs}")

        if len(instance_ids) == 0:
            logging.info(f"All simulation tasks for client {client_id} have been done.")
            return env_server_pb2.EnvReply(all_done=True)

        states, observations = self.env_manager.get_env_states_and_obs(instance_ids)
        # state and observation serialization here,
        #   currently we use pickle proxy.
        b_states = pickle.dumps(states)
        b_observations = pickle.dumps(observations)
        return env_server_pb2.EnvReply(
            instance_ids=instance_ids,
            b_states=b_states,
            b_observations=b_observations,
            all_done=False,
        )

    def Step(self, request, context):
        """Handling environment stepping."""

        actions = pickle.loads(request.b_actions)
        info = self.env_manager.step(request.instance_ids, actions)
        client_id = _get_client_identification(context)
        self.session_buffer[client_id].update("dead_envs", info["dones"])
        return env_server_pb2.EnvReply(all_done=False)

    def RequestEvaluation(
        self, request: env_server_pb2.EnvRequest, context: grpc.ServicerContext
    ):
        """Handling client request for create multiple environments. Then return
        the list of environment instance ids to the corresponding client."""
        client_identification = _get_client_identification(context)

        vec_env_desc = request.vec_env_desc
        env_id = vec_env_desc.env_id
        env_num = vec_env_desc.env_num
        episode_length = vec_env_desc.max_episode_steps
        logging.info(
            f"Building connection with {client_identification}, env_id={env_id}, env_num={env_num}"
        )

        # a list of instance id
        env_list = self.env_manager.create_envs(env_id, env_num, episode_length)
        self.session_buffer[client_identification] = Session(
            env_id=env_id, env_num=env_num, env_list=env_list
        )
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    serve()
