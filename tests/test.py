from iod.viz_utils import *

@torch.no_grad()
def viz_Regert_in_Psi(base1, base2, state, Repr_goal_array=None, State_goal_array=None, ax=None, cmap=None, color=None, num_samples=10, device='cpu', path='./',  base3=None):
    def get_fuctions(base):
        return base['qf1'], base['qf2'], base['alpha'], base['policy'] 
    
    density = 200
    x = np.linspace(-1, 1, density)
    y = np.linspace(-1, 1, density)
    X, Y = np.meshgrid(x,y)
    fig = plt.figure(figsize=(18, 12), facecolor='w')
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    pos = torch.tensor(pos).to(device)
    pos_flatten = pos.view(-1,2)
    option = pos_flatten
    state_batch = state.unsqueeze(0).repeat(option.shape[0], 1)
    
    # value 1:
    qf1, qf2, alpha, policy = get_fuctions(base1)
    V1 = EstimateValue(policy, alpha, qf1, qf2, option, state_batch, num_samples=10)
    V1 = V1.view(pos.shape[0],pos.shape[1])
    
    # value 2:
    qf1, qf2, alpha, policy = get_fuctions(base2)
    V2 = EstimateValue(policy, alpha, qf1, qf2, option, state_batch, num_samples=10)
    V2 = V2.view(pos.shape[0],pos.shape[1])
    

    if base3 is not None:
        qf1, qf2, alpha, policy = get_fuctions(base3)
        Vk__ = EstimateValue(policy, alpha, qf1, qf2, option, state_batch, num_samples=10)
        Vk__ = Vk__.view(pos.shape[0],pos.shape[1])
        print("base3 involved")
        Regret = 1*(V2/V2.max() - V1/V1.max()) + 0*(V1 - Vk__)

    else:
        # Regret:
        Regret = V2 - V1

    print('Regret:', Regret.max(), Regret.min())
    
    # Special Points:
    if Repr_goal_array is None:
        Repr_goal_array = np.load('/mnt/nfs2/zhanghe/NuAgent/test/PSZP-6-cal_softmaxsd000_1730088991_ant_maze_PSZP-Repr_goal_list.npy')

        Repr_obs_array = np.load('/mnt/nfs2/zhanghe/NuAgent/test/PSZP-6-cal_softmaxsd000_1730088991_ant_maze_PSZP-Repr_obs_list.npy')
    
    x_points = Repr_goal_array[:, 0]
    y_points = Repr_goal_array[:, 1]
    
    
    # xy_points = torch.tensor(list(zip(x_points, y_points))).to(device)
    # state_batch_for_points = state.unsqueeze(0).repeat(xy_points.shape[0], 1)
    # # 计算对应的 V1 和 V2 值
    # qf1, qf2, alpha, policy = get_fuctions(base1)
    # V1_points = EstimateValue(policy, alpha, qf1, qf2, xy_points, state_batch_for_points, num_samples=10)

    # qf1, qf2, alpha, policy = get_fuctions(base2)
    # V2_points = EstimateValue(policy, alpha, qf1, qf2, xy_points, state_batch_for_points, num_samples=10)

    # # 计算对应的 Regret 值
    # Regret_points = (V2_points - V1_points).cpu().numpy()

    if ax is None:  
        fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制 2D 平面图，颜色表示 Regret 值
    c = ax.contourf(X, Y, Regret.cpu().numpy(), levels=50, cmap='viridis')
    zero_contour = ax.contour(X, Y, Regret.cpu().numpy(), levels=[0], colors='black', linewidths=1.5)
    # 添加颜色条，用于表示 Regret 的数值大小
    fig.colorbar(c, ax=ax, label='Regret')
    
    # 计算 Regret 的梯度
    Regret_np = Regret.cpu().numpy()
    grad_x, grad_y = np.gradient(Regret_np)

    # 计算梯度大小
    magnitude = np.sqrt(grad_x**2 + grad_y**2)


    # 调整颜色映射和线条宽度以增强对比
    # ax.contour(X, Y, magnitude, levels=10, colors='black', linewidths=1, alpha=0.8)

    # 使用颜色渐变表示梯度大小，并选择更高对比度的颜色映射
    contour_grad = ax.contour(X, Y, magnitude, levels=10, cmap='hot', linewidths=1.2, alpha=0.8)

    # 添加颜色条显示梯度大小
    plt.colorbar(contour_grad, ax=ax, label='Gradient Magnitude')
    
    # 用矢量场显示梯度方向并增强颜色显示
    # quiver = ax.quiver(X, Y, grad_x, grad_y, magnitude, scale=100, cmap='plasma', alpha=0.8)
    
    # 绘制特定点，颜色与前面不同以便区分
    ax.scatter(x_points, y_points, c=color, cmap=cmap, s=50, edgecolor='k')

    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    # plt.savefig(path + '-Regret-2D' + '.png')
    # print('save at: ' + path + '-Regret-2D' + '.png')




import argparse
parser = argparse.ArgumentParser(description="A simple example of argparse usage")
parser.add_argument('-e', '--epoch_num', type=int)
args = parser.parse_args()

env = MazeWrapper("antmaze-medium-diverse-v0", random_init=False)
epoch_num = args.epoch_num
policy_path = "/mnt/nfs2/zhanghe/NuAgent/exp/MazeSZN/Baselinesd000_1730702994_ant_maze_PSZP/wandb/run-20241104_144955-3fuytij1/filesoption_policy-" + str(epoch_num) +'.pt'
# policy_path1 = "/mnt/nfs2/zhanghe/NuAgent/exp/MazeSZN/PSZP-8-GMM1-Deque10-std1_3e1_1e1sd000_1730290115_ant_maze_PSZP/wandb/latest-run/filesoption_policy-" + str(epoch_num-100) +'.pt'
# policy_path2 = "/mnt/nfs2/zhanghe/NuAgent/exp/MazeSZN/PSZP-8-GMM1-Deque10-std1_3e1_1e1sd000_1730290115_ant_maze_PSZP/wandb/latest-run/filesoption_policy-" + str(epoch_num-200) +'.pt'


traj_encoder_path = policy_path.replace("option_policy", "traj_encoder")
SZN_path = policy_path.replace("option_policy", "SampleZPolicy")

load_option_policy_base = torch.load(policy_path)
load_traj_encoder_base = torch.load(traj_encoder_path)
load_SZN_path_base = torch.load(SZN_path)

model_name = policy_path.split('/')[-4]
path = './test/' + model_name   
dim_option = 2
device = 'cuda:0'

if "target_traj_encoder" in load_traj_encoder_base.keys():
    agent_traj_encoder = load_traj_encoder_base['target_traj_encoder'].eval()
else:
    agent_traj_encoder = load_traj_encoder_base['traj_encoder'].eval()


qf1 = load_option_policy_base['qf1']
qf2 = load_option_policy_base['qf2']
alpha = load_option_policy_base['alpha']
policy = load_option_policy_base['policy']

window = load_SZN_path_base['window']

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
baseline=1

def Psi(phi_x, phi_x0=None):
    # if phi_x0 is None:
    #     x0 = self.s0        # [1, dim_obs]; phi_x: [batch, dim_z]
    #     phi_x0 = self.traj_encoder(x0).mean     # [1, dim_z]
    # return torch.tanh(1/150 * (phi_x))
    return phi_x

# def Psi(phi_x, phi_x0):
#     # if phi_x0 is None:
#     #     x0 = self.s0        # [1, dim_obs]; phi_x: [batch, dim_z]
#     #     phi_x0 = self.traj_encoder(x0).mean     # [1, dim_z]
#     return torch.tanh((phi_x - phi_x0))


# FD, AR, eval_metrics = PlotMazeTrajWindowDist(env, DistWindow, self.target_traj_encoder, self.qf1, self.qf2, self.log_alpha, self.option_policy, self.device, Psi=partial(self.Psi), dim_option=self.dim_option, max_path_length=self.max_path_length, path=path)




# # Traj. Map:
run_env = 1
if run_env:
    All_Goal_obs_list = []
    ax[0,0], FinallDistanceList, All_Repr_obs_list, All_Goal_obs_list, All_trajs_list, FinallDistanceList, ArriveList, All_Cover_list = eval_cover_rate(env, agent_traj_encoder, policy, dim_option, device, Psi=Psi, freq=2, ax=ax[0,0], max_path_length=max_path_length, option_type='baseline')
    ax[0,0] = plot_trajectories(env, All_trajs_list, fig, ax[0,0])
    ax[1,0] = PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length, is_goal=True, ax=ax[1,0])
    # eval_metrics
    eval_metrics = calc_eval_metrics(All_Cover_list, is_option_trajectories=True)
    print('[eval_metrics]:', eval_metrics)

    # Value Map: 
    fig = viz_Value_in_Psi(policy, alpha, qf1, qf2, state=s0, num_samples=10, device=device, path=path, fig=fig)
    
    # save special points
    filepath = path + "-Repr_obs_list.npy"
    np.save(filepath, np.array(All_Repr_obs_list))
    filepath = path + "-Repr_goal_list.npy"
    np.save(filepath, np.array(All_Goal_obs_list)[:,0,:])
    print("save as ", filepath)


    fig.suptitle("Policy Epoch: " + str(epoch_num), fontsize=16)
    filepath = path + "-RegertMap.png"
    fig.savefig(filepath) 
    print(filepath)

# # # Window Dist：

# filepath = path + "-Maze_traj.png"
# plt.savefig(filepath) 
# print(filepath)

# eval_metrics = calc_eval_metrics(All_Cover_list, is_option_trajectories=True)
# print('[eval_metrics]:', eval_metrics)


# def draw_goal_map(State_goal_array, ax):
#     ax.scatter(State_goal_array, color='red', )


# Regret Map：
def RegretMap(ax, ReprGoalPath=None, All_Goal_obs_list=None): 
    base1 = torch.load(policy_path1)
    base2 = load_option_policy_base
    # base3 = torch.load(policy_path2)
    base3 = None
    State_goal_array = np.load('/mnt/nfs2/zhanghe/NuAgent/AnalysisData/goal_list.npy')
    colors = np.arange(len(State_goal_array))   
    cmap = plt.get_cmap("tab20", len(State_goal_array))  # 使用 tab20 调色板，并指定 26 个离散颜色
    ax[0,0].scatter(State_goal_array[:,0], State_goal_array[:,1], c=colors, cmap=cmap)
    
    if ReprGoalPath is not None:
        Repr_goal_array = np.load(ReprGoalPath)
        viz_Regert_in_Psi(base1, base2, state=s0, num_samples=10, device=device, path=path, Repr_goal_array=Repr_goal_array, State_goal_array=State_goal_array, ax=ax[1,1], color=colors, cmap=cmap, base3=base3)
        
    else:
        np.save('/mnt/nfs2/zhanghe/NuAgent/tests/savenp/testRepr_goal_array.npy', np.array(All_Goal_obs_list)[:,0,:])
        viz_Regert_in_Psi(base1, base2, state=s0, num_samples=10, device=device, path=path, Repr_goal_array=np.array(All_Goal_obs_list)[:,0,:], State_goal_array=State_goal_array, ax=ax[1,1], color=colors, cmap=cmap, base3=base3)
        
    ax[1,1] = viz_dist_circle(window, psi_z=None, ax=ax[1,1])

    fig.suptitle("Policy Epoch: " + str(epoch_num), fontsize=16)
    filepath = path + "-RegertMap.png"
    fig.savefig(filepath) 
    print(filepath)


if __name__ == '__main__':
    
    # ReprGoalPath = '/mnt/nfs2/zhanghe/NuAgent/AnalysisData/PSZP-6-PopDeque_windowsize10-softmax1-w101w25sd000_1730116659_ant_maze_PSZP-Repr_goal_list.npy'    
    
    if len(All_Goal_obs_list) == 0:
        ReprGoalPath = "/mnt/nfs2/zhanghe/NuAgent/tests/savenp/testRepr_goal_array.npy"
    else:
        ReprGoalPath = None
    if not baseline:
        RegretMap(ax, ReprGoalPath=ReprGoalPath, All_Goal_obs_list=All_Goal_obs_list)
