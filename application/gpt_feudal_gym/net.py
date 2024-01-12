from argparse import Namespace
from gym import Space

import torch
import torch.nn as nn

from uniagent.models.mingpt.tokenizer.vision import VisionEmbedding
from uniagent.models.mingpt.tokenizer.scalar import DiscreateScalarTokenizer
from uniagent.models.mingpt.outer_query_gpt import GPT, OuterQueryGPT
from uniagent.models.fun.gpt_feudal import GPTFeudal, FeudalState


class GPTFeudalVision(GPTFeudal):
    def create_perception(self) -> nn.Module:
        if len(self.observation_space.shape) == 3:
            return nn.ModuleDict(
                dict(
                    vision=VisionEmbedding(self.observation_space, self.config),
                    scalar=DiscreateScalarTokenizer(self.action_space.n, self.config),
                )
            )
        elif len(self.observation_space.shape) == 1:
            return DiscreateScalarTokenizer(self.action_space.n, self.config)
        else:
            raise NotImplementedError
        
    def init_state(self, batch_size: int, device=None) -> FeudalState:
        return FeudalState(
            self.manager.init_state(batch_size),
            self.worker.init_state(batch_size),
            [],
            []
        )

    # TODO(ming): do not forget apply cache for inference mode
    def init_memory(self, batch_size: int):
        self.time_step = 0
        shape = (batch_size, self.config.traj_len, 50, self.config.n_embed)
        self.memory = torch.zeros(shape, device=self.config.device)

    def merge_with_memory(self, step_token_embedding_batch: torch.Tensor):
        # shape of step_token: [batch_size, 50, embedding_size]
        self.memory[self.time_step] = step_token_embedding_batch.unsqueeze(1).clone()
        if self.time_step + 1 == self.config.traj_len:
            # shift
            self.memory = self.memory.roll(-1, 0)
        self.time_step = min(self.time_step + 1, self.config.traj_len)

        return self.memory[:, : self.time_step].view(
            -1, self.time_step * 50, self.config.n_embed
        )

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
            token_seq_embedding_batch, queries.detach(), feudal_state.worker_state
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
