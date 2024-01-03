from argparse import Namespace
import os
import torch

print("os path", os.environ["PYTHONPATH"])

from uniagent.envs.utils import make_env_wrapper
from uniagent.models.mingpt.tokenizer import VisionEmbedding


args = Namespace(
    **dict(
        task_type="atari",
        env_name="PongNoFrameskip-v4",
    )
)
env = make_env_wrapper(args)()

print("observation space:", env.observation_space)
print("action space:", env.action_space)

config = Namespace(
    **dict(
        fp16=False,
        vision_patch_size=12,  # note 84 / 7 = 12
        vision_num_input_channels=env.observation_space.shape[0],
        n_embed=768,
        vision_position_vocab_size=128,
        vision_hidden_dropout_prob=0.5,
    )
)
vision_embedding = VisionEmbedding(config)
vision_embedding.print_model_size()
embeddings = vision_embedding(
    torch.from_numpy(env.observation_space.sample()).unsqueeze(0)
)
patch_size = (env.observation_space.shape[1] // 12) ** 2
assert embeddings.size() == (1, patch_size, config.n_embed), embeddings.size()
