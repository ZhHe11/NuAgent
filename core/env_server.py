import uuid
from service import env_server_pb2_grpc
from service import env_server_pb2
from core.env_manager import EnvManager


class EnvServer(env_server_pb2_grpc.EnvServerServicer):
    def __init__(self):
        super().__init__()
        self.env_manager = EnvManager()

    def GetStateAndObs(self, request, context):
        # TODO(ming): retrieve states from manager with given env list
        instance_ids = request.instance_ids
        states, observations = self.env_manager.get_env_states_and_obs(instance_ids)
        all_done = self.env_manager.check_all_done()
        # TODO(ming): seriealize as string here
        return env_server_pb2.EnvReply(
            instance_ids=instance_ids,
            states=states,
            observations=observations,
            all_done=all_done)

    def Step(self, request, context):
        # TODO(ming): do deserilization here for actions
        self.env_manager.step(request.instance_ids, request.actions)

    def RequestEnvs(self, request, context):
        env_id = request.env_id
        env_num = request.env_num
        # a list of instance id
        env_list = self.env_manager.create_envs(env_id, env_num)
        return env_server_pb2.EnvReply(instance_ids=env_list, all_done=False)
