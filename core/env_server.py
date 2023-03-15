import uuid
from service import env_server_pb2_grpc
from service import env_server_pb2
from core.env_manager import EnvManager


class EnvServer(env_server_pb2_grpc.EnvServerServicer):
    def __init__(self):
        super().__init__()
        self.env_manager = EnvManager()

    # def SayHello(self, request, context):
    #     return env_server_pb2.HelloReply(message="Hello, %s" % request.name)

    def Connect(self, request, context):
        raise NotImplementedError

    def RegisterEnv(self, request, context):
        env_name = uuid.uuid4()
        self.env_manager.register_env(name=env_name)
        return env_server_pb2.EnvReply(message={"env_name": env_name})

    def GetObs(self, request, context):
        return env_server_pb2.EnvObs(message=None)

    def Step(self, request, context):
        raise NotImplementedError
