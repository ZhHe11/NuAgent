from argparse import Namespace

import torch
import torch.nn as nn

from uniagent.models.mingpt.tokenizer.vision import VisionEmbedding
from uniagent.models.mingpt.tokenizer.scalar import ContinuousScalarTokenizer
from uniagent.models.mingpt.outer_query_gpt import GPT, OuterQueryGPT
from uniagent.models.fun.gpt_feudal import GPTFeudal, FeudalState


class GPTFeudalVision(GPTFeudal):
    def create_perception(self) -> nn.Module:
        if len(self.observation_space.shape) == 3:
            config = Namespace(
                **dict(
                    fp16=False,
                    vision_patch_size=16,
                    vision_num_input_channels=self.observation_space.shape[0]
                    if self.channel_first
                    else self.observation_space.shape[-1],
                    n_embed=256,
                    vision_position_vocab_size=256,
                    vision_hidden_dropout_prob=0.1,
                )
            )
            return VisionEmbedding(config)
        elif len(self.observation_space.shape) == 1:
            raise NotImplementedError
            # return ContinuousScalarTokenizer(config)
        else:
            raise NotImplementedError

    # TODO(ming): do not forget apply cache for inference mode
    def forward(
        self, x: torch.Tensor, feudal_state: FeudalState, reset_value_grad: bool = True
    ):
        token_embedding_batch = self.perception(x)
        # each ele: [batch_size, seq_len, embed_dim]
        values_manager, queries, states_M = self.manager.compute_with_outer_embedding(
            token_embedding_batch, feudal_state.manager_state
        )

        # here, goals play as query to compute the corresponding values and policy
        (
            values_worker,
            logits,
            states_W,
        ) = self.worker.compute_with_outer_embedding(
            token_embedding_batch, queries.detach(), feudal_state.worker_state
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
