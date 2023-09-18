from service import agent_server_pb2_grpc

from core.agent_manager import AgentManager


class AgentServer(agent_server_pb2_grpc.AgentServerServicer):
    def __init__(self):
        super().__init__()
        self.agent_manager = AgentManager()

    def Action(self, request, context):
        raise NotImplementedError

    def Connect(self, request, context):
        raise NotImplementedError
