from argparse import Namespace
from gym import Space

import torch
import torch.nn as nn

from uniagent.models.mingpt.tokenizer.vision import VisionEmbedding
from uniagent.models.mingpt.tokenizer.scalar import ContinuousScalarTokenizer
from uniagent.models.mingpt.outer_query_gpt import GPT, OuterQueryGPT
from uniagent.models.fun.gpt_feudal import GPTFeudal, FeudalState


class GPTFeudalVision(GPTFeudal):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        backbone: str = "gpt2",
        d: int = 256,
        k: int = 16,
        c: int = 10,
        channel_first: bool = True,
        device: torch.DeviceObjType = ...,
    ) -> None:
        super().__init__(
            observation_space, action_space, backbone, d, k, c, channel_first, device
        )

    def create_perception(self) -> nn.Module:
        if len(self.observation_space.shape) == 3:
            return nn.ModuleDict(
                dict(
                    vision=VisionEmbedding(self.args),
                    scalar=ContinuousScalarTokenizer(self.args),
                )
            )
        elif len(self.observation_space.shape) == 1:
            return ContinuousScalarTokenizer(self.args)
        else:
            raise NotImplementedError

    # TODO(ming): do not forget apply cache for inference mode
    def init_memory(self):
        raise NotImplementedError

    def merge_with_memory(self, step_token_embedding_batch: torch.Tensor):
        # shape of step_token: [batch_size, 50, embedding_size]
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, last_action: torch.Tensor, feudal_state: FeudalState
    ):
        # [batch_size, 49, embedding_size]
        vision_token_embedding_batch = self.perception.vision(x)
        # [batch_size, 1, embedding_size]
        action_token_embedding_batch = self.perception.scalar(last_action)
        # [batch_size, 50, embedding_size]
        step_token_embedding_batch = torch.concat(
            [vision_token_embedding_batch, action_token_embedding_batch], dim=1
        )
        # [batch_size, time_step * 50, embedding_size]
        token_seq_embedding_batch = self.merge_with_memory(step_token_embedding_batch)

        # each ele: [batch_size, seq_len, embed_dim]
        values_manager, queries, states_M = self.manager(
            token_seq_embedding_batch, feudal_state.manager_state
        )

        # here, goals play as query to compute the corresponding values and policy
        (
            values_worker,
            logits,
            states_W,
        ) = self.worker(
            step_token_embedding_batch, queries.detach(), feudal_state.worker_state
        )

        # keep dim
        values = [values_manager, values_worker]
        logits = [queries, logits]
        states = FeudalState(
            states_M, states_W, feudal_state.state_seg, feudal_state.goal_seg
        )
        if not self.training:
            values = [v[:, -1] for v in values]
            logits = [v[:, -1] for v in logits]

        return values, logits, states
