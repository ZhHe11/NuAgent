from iod.viz_utils import *

env = MazeWrapper("antmaze-medium-diverse-v0", random_init=False)

policy_path = "/mnt/nfs2/zhanghe/NuAgent/exp/MazeSZN/PSZP-6-tanh150sd000_1730015883_ant_maze_PSZP/option_policy600.pt"
traj_encoder_path = policy_path.replace("option_policy", "traj_encoder")
# SZN_path = "/mnt/nfs2/zhanghe/NuAgent/exp/MazeSZN/PSZP-1-k_3sd000_1729773482_ant_maze_PSZP/wandb/latest-run/filesSampleZPolicy-1500.pt"

load_option_policy_base = torch.load(policy_path)
load_traj_encoder_base = torch.load(traj_encoder_path)
# load_SZN_path_base = torch.load(SZN_path)

model_name = policy_path.split('/')[-4]
path = './test/' + model_name   
dim_option = 2
device = 'cuda'

if "target_traj_encoder" in load_traj_encoder_base.keys():
    agent_traj_encoder = load_traj_encoder_base['target_traj_encoder'].eval()
else:
    agent_traj_encoder = load_traj_encoder_base['traj_encoder'].eval()


policy = load_option_policy_base['policy']

obs0 = env.reset()
s0 = torch.tensor(obs0).to(device).float()
fig, ax = plt.subplots(2,2)
fig.subplots_adjust(wspace=0.4, hspace=0.4) 
env.draw(ax[0,0])
ax[0,0].set_title('State of Traj. in Maze')
ax[0,1].set_axis_off()
ax[0,1].set_title('Estimate Value in Z Space')

max_path_length=300
isCover=0

def Psi(phi_x, phi_x0=None):
    # if phi_x0 is None:
    #     x0 = self.s0        # [1, dim_obs]; phi_x: [batch, dim_z]
    #     phi_x0 = self.traj_encoder(x0).mean     # [1, dim_z]
    return torch.tanh(1/150 * (phi_x))

ax[0,0], FinallDistanceList, All_Repr_obs_list, All_Goal_obs_list, All_trajs_list, FinallDistanceList, ArriveList, All_Cover_list = eval_cover_rate(env, agent_traj_encoder, policy, dim_option, device, Psi=Psi, freq=2, ax=ax[0,0], max_path_length=max_path_length)
ax[0,0] = plot_trajectories(env, All_trajs_list, fig, ax[0,0])
ax[1,0] = PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length, is_goal=True, ax=ax[1,0])

filepath = path + "-Maze_traj.png"
plt.savefig(filepath) 
print(filepath)

eval_metrics = calc_eval_metrics(All_Cover_list, is_option_trajectories=True)
print('[eval_metrics]:', eval_metrics)

