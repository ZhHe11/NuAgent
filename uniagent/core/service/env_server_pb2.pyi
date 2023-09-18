from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

class AgentActions(_message.Message):
    __slots__ = ["key", "value"]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(
        self, key: _Optional[str] = ..., value: _Optional[str] = ...
    ) -> None: ...

class EnvReply(_message.Message):
    __slots__ = ["all_done", "b_observations", "b_states", "instance_ids"]
    ALL_DONE_FIELD_NUMBER: _ClassVar[int]
    B_OBSERVATIONS_FIELD_NUMBER: _ClassVar[int]
    B_STATES_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_IDS_FIELD_NUMBER: _ClassVar[int]
    all_done: bool
    b_observations: bytes
    b_states: bytes
    instance_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        b_states: _Optional[bytes] = ...,
        b_observations: _Optional[bytes] = ...,
        instance_ids: _Optional[_Iterable[str]] = ...,
        all_done: bool = ...,
    ) -> None: ...

class EnvRequest(_message.Message):
    __slots__ = ["b_actions", "instance_ids", "vec_env_desc"]
    B_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_IDS_FIELD_NUMBER: _ClassVar[int]
    VEC_ENV_DESC_FIELD_NUMBER: _ClassVar[int]
    b_actions: bytes
    instance_ids: _containers.RepeatedScalarFieldContainer[str]
    vec_env_desc: VecEnvDesc
    def __init__(
        self,
        instance_ids: _Optional[_Iterable[str]] = ...,
        b_actions: _Optional[bytes] = ...,
        vec_env_desc: _Optional[_Union[VecEnvDesc, _Mapping]] = ...,
    ) -> None: ...

class Observations(_message.Message):
    __slots__ = ["agent_id", "observation"]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    agent_id: str
    observation: str
    def __init__(
        self, agent_id: _Optional[str] = ..., observation: _Optional[str] = ...
    ) -> None: ...

class VecEnvDesc(_message.Message):
    __slots__ = ["env_id", "env_num", "max_episode_steps"]
    ENV_ID_FIELD_NUMBER: _ClassVar[int]
    ENV_NUM_FIELD_NUMBER: _ClassVar[int]
    MAX_EPISODE_STEPS_FIELD_NUMBER: _ClassVar[int]
    env_id: str
    env_num: int
    max_episode_steps: int
    def __init__(
        self,
        env_id: _Optional[str] = ...,
        env_num: _Optional[int] = ...,
        max_episode_steps: _Optional[int] = ...,
    ) -> None: ...
