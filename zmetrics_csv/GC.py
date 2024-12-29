from iod.viz_utils import *


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--epoch', type=int, default='0')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--eval_type', type=str, default='random')
    args = parser.parse_args()
    device = 'cuda:3'
    max_path_length = 300
    args.eval_num = 16
    # args.model_path = '/mnt/nfs2/zhanghe/NuAgent/exp/LM-ready/TheBestsd000_1735206880_lm_SZPC'
    # args.eval_type = 'Projection_psi'

    args.model_path = '/mnt/nfs2/zhanghe/NuAgent/exp/MazeReady/Baseline-dim4sd000_1733193086_ant_maze_metra_bl'
    args.eval_type = 'baseline'
    
    eval_type = args.eval_type
    args.epoch_list = ['7500']
    
    for epoch in args.epoch_list:
        # 1. define the env:
        # Ant;
        from envs.mujoco.ant_env import AntEnv
        from iod.utils import get_normalizer_preset
        from garagei.envs.consistent_normalized_env import consistent_normalize

        # env = AntEnv(render_hw=100)
        # normalizer_name = 'ant'
        # normalizer_kwargs = {}
        # normalizer_mean, normalizer_std = get_normalizer_preset(f'{normalizer_name}_preset')
        # env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std, **normalizer_kwargs)      
        
        # AntMaze
        from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze
        args.env = 'ant_maze'
        env = MazeWrapper("antmaze-medium-diverse-v0", random_init=False)
        # args.env = 'ant_large_maze'
        # from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze
        # env = MazeWrapper("antmaze-large-diverse-v0", random_init=False)
        
        # # LM
        # from envs.AntMazeEnv import MazeWrapper, GoalReachingMaze
        # args.env = 'lm'
        # env = MazeWrapper("maze2d-large-v1", random_init=False)
        
        normalizer_kwargs = {}
        env = consistent_normalize(env, normalize_obs=False, **normalizer_kwargs)
        
        obs = env.reset()
        policy_path = args.model_path + '/wandb/latest-run/filesoption_policy-' + str(epoch) + '.pt'
        traj_encoder_path = policy_path.replace('option_policy', 'traj_encoder')
        load_option_policy_base = torch.load(policy_path, map_location=device)
        load_traj_encoder_base = torch.load(traj_encoder_path, map_location=device)
        agent_policy = load_option_policy_base['policy'].eval()
        dim_option = load_traj_encoder_base['dim_option']
        agent_traj_encoder = load_traj_encoder_base['traj_encoder'].eval()
        model_name = policy_path.split('/')[-4]
        path = './test/' + model_name   

        if 'psi' in eval_type:
            SZN_path =policy_path.replace('option_policy', 'SampleZPolicy')
            load_SZN_path_base = torch.load(SZN_path, map_location=device)
            SZN = load_SZN_path_base['goal_sample_network'].eval()
            input_token = load_SZN_path_base['input_token']
            qf1 = load_option_policy_base['qf1']
            qf2 = load_option_policy_base['qf2']
            alpha = load_option_policy_base['alpha']

        # 4. interaction:
        def __Psi(phi_x, phi_x0=None):
            if 'psi' in eval_type:
                return torch.tanh(2/max_path_length * (phi_x))
            else:
                return phi_x

        # Plot GMM:
        fig, ax = plt.subplots(1,2, figsize=(20,8))
        
        ax[0], FinallDistanceList, All_Repr_obs_list, All_Goal_obs_list, All_trajs_list, FinallDistanceList, ArriveList, All_Cover_list = eval_cover_rate(env, agent_traj_encoder, agent_policy, dim_option, device, ax=ax[0], max_path_length=max_path_length, Psi=__Psi, option_type=args.eval_type)
        
        FinallDistance = np.array(FinallDistanceList).mean()
        ArriveRate = np.array(ArriveList).mean()
        
        
        plot_trajectories(env, All_trajs_list, fig, ax[0])
        PCA_plot_traj(All_Repr_obs_list, All_Goal_obs_list, path, path_len=max_path_length, is_goal=True, ax=ax[1])
        
        info = f"FD: {FinallDistance}; AR: {ArriveRate}"
        ax[0].set_title(info)
        print(info)
        
        save_path = '/mnt/nfs2/zhanghe/NuAgent/zmetrics_csv/AntLargeMaze'
        filepath = save_path + '/GoalCondition' + str(epoch) + str(args.eval_type) + '.pdf'
        plt.savefig(filepath, format='pdf', bbox_inches='tight')
        print(filepath)
        