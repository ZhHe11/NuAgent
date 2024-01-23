import torch

from torch.distributions import Categorical

from application.gpt_feudal_gym.net import GPTFeudalVision
from application.gpt_feudal_gym.cli import command_args
from uniagent.envs.utils import make_env_wrapper

args = command_args()

args.device = (
    torch.device(f"cuda:{args.device_idx}" if torch.cuda.is_available() else "cpu")
    if args.use_cuda
    else torch.device("cpu")
)

env = make_env_wrapper(args)()
args.vision_num_input_channels = env.observation_space.shape[0]
feudal = GPTFeudalVision(env.observation_space, env.action_space, args)
feudal.report_num_of_parameters()

done = False
obs, _ = env.reset()
feudal.eval()
feudal.init_memory(1)
feudal_state = feudal.init_state(1)
last_action = torch.LongTensor([env.action_space.sample()]).to(args.device)

while not done:
    obs = torch.from_numpy(obs).to(args.device)
    with torch.no_grad():
        values, logits, feudal_state = feudal(
            obs.unsqueeze(0), last_action, feudal_state
        )
        # compute action
        manager_values, worker_values = values
        goals, worker_logits = logits
        dist = Categorical(logits=worker_logits)
        action = dist.sample()
    obs, reward, done, truncated, info = env.step(action.cpu().numpy()[0, 0])
    done = done or truncated
