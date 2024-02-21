"""
This file implements a GPTFeudalVision that is backboned with GPT, and the input is vision.
To achieve that, GPTFeudalVision comprises three key components:

1) shared preception module: see `create_preception`, the preception module is responsible for encoding an input trajectory
    that is formed as (s_0, a_0, ..., s_t, a_t), where `s_t` the image-based observation and `a_t` the discrete action. As
    there are two kinds of tensors (observations and actions) that differ in the data type, the preception module is 
    implemented as a module dict including `VisionEmbedding` and `DiscreteScalarTokenizer` for the tokenization of 
    observations and actions, respectively.

2) Manager: the Manager is implemented as an instance of `uniagent.models.fun.manager::TransformerManager`, which a 
    Transformer accepts outer token sequence that is given by the preception module, for goal generation

3) Worker: the Worker is implemented as an instance of `uniagent.models.fun.worker::TransformerWorker`, which a GPT
    accepts outer queries that is given by the Manager, a sequence of goals, each relates to an observation
"""

import torch
import torch.nn as nn

from uniagent.models.mingpt.tokenizer.vision import VisionEmbedding
from uniagent.models.mingpt.tokenizer.scalar import DiscreteScalarTokenizer
from uniagent.models.fun.gpt_feudal import GPTFeudal, FeudalState


class GPTFeudalVision(GPTFeudal):
    def create_perception(self) -> nn.Module:
        if len(self.observation_space.shape) == 3:
            return nn.ModuleDict(
                dict(
                    vision=VisionEmbedding(self.observation_space, self.config),
                    scalar=DiscreteScalarTokenizer(self.action_space.n, self.config),
                )
            )
        elif len(self.observation_space.shape) == 1:
            return DiscreteScalarTokenizer(self.action_space.n, self.config)
        else:
            raise NotImplementedError

    def init_state(self, batch_size: int, device=None) -> FeudalState:
        return FeudalState(
            self.manager.init_state(batch_size),
            self.worker.init_state(batch_size),
            [],
            [],
        )

    # TODO(ming): do not forget apply cache for inference mode
    def init_memory(self, batch_size: int):
        self.time_step = 0
        shape = (batch_size, self.config.traj_len, 50, self.config.n_embed)
        self.memory = torch.zeros(shape, device=self.config.device)

    def merge_with_memory(
        self, step_token_embedding_batch: torch.Tensor
    ) -> torch.Tensor:
        """Concatenating a given batch of token to a segment of memory, and return such a concatnation \
        as a batch of token sequences, shaped as `[batch_size, time_step * size_of_one_step, n_embed]`

        Args:
            step_token_embedding_batch (torch.Tensor): A batch of token of a timestep, shaped as \
                `[batch_size, size_of_one_step, n_embed]`

        Returns:
            torch.Tensor: A batch of token sequences
        """

        # shape of step_token: [batch_size, 50, embedding_size]
        assert (
            len(step_token_embedding_batch.shape) > 1
        ), step_token_embedding_batch.shape
        self.memory[:, self.time_step] = step_token_embedding_batch.unsqueeze(1).clone()
        if self.time_step + 1 >= self.config.traj_len:
            # shiftting the memory as its capacity has been achieved
            self.memory = self.memory.roll(-1, 1)
        self.time_step = min(self.time_step + 1, self.config.traj_len - 1)
        batch_size, size_of_one_step = step_token_embedding_batch.size()[:2]

        token_seq = self.memory[:, : self.time_step].view(
            batch_size, self.time_step * size_of_one_step, self.config.n_embed
        )

        return token_seq

    def forward(
        self, x: torch.Tensor, last_action: torch.Tensor, feudal_state: FeudalState
    ):
        # [batch_size, 7 * 7, embedding_size]
        vision_token_embedding_batch = self.perception.vision(x)
        # [batch_size, 1, embedding_size]
        action_token_embedding_batch = self.perception.scalar(last_action).unsqueeze(1)
        # [batch_size, 50, embedding_size]
        step_token_embedding_batch = torch.concat(
            [vision_token_embedding_batch, action_token_embedding_batch], dim=1
        )
        # [batch_size, time_step * 50, embedding_size]
        token_seq_embedding_batch = self.merge_with_memory(step_token_embedding_batch)

        # each ele: [batch_size, seq_len, embed_dim]
        # TODO(ming): should check whether the gradient flow whether break
        values_manager, queries, states_M = self.manager(
            token_seq_embedding_batch, feudal_state.manager_state
        )

        # here, goals play as query to compute the corresponding values and policy
        # convert queries to token_seq_embedding_batch
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
