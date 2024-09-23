export CUDA_VISIBLE_DEVICES=6
export MUJOCO_GL="osmesa"


# regret
python tests/main.py --run_group Quadruped --env dmc_quadruped --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --frame_stack 3 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo SZN --n_epochs_per_log 10 --n_epochs_per_eval 20 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 4 --encoder 1 --sample_cpu 0 --trans_optimization_epochs 1 \
    --exp_name Contrastive_v2-bs1024 --phi_type contrastive --policy_type baseline --explore_type baseline --sample_type baseline --num_her 0 --_trans_phi_optimization_epochs 200 --_trans_policy_optimization_epochs 200 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 

# # ours
#     --phi_type contrastive --policy_type her_reward --explore_type baseline --sample_type contrastive --num_her 0 --_trans_phi_optimization_epochs 1 --trans_optimization_epochs 200 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 
# # baseline
#     --phi_type baseline --policy_type baseline --explore_type baseline --sample_type baseline --num_her 0 --_trans_phi_optimization_epochs 1 --trans_optimization_epochs 200 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 


# python tests/main.py --run_group Maze --exp_name theta-dim2 --env ant_maze --max_path_length 300 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type off --sac_max_buffer_size 100000 --algo metra --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 100 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --sac_scale_reward 1 --trans_minibatch_size 1024 --is_wandb 0 --phi_type contrastive --policy_type her_reward --explore_type theta --sample_type contrastive --num_her 0 --_trans_phi_optimization_epochs 1 --target_theta 1e-3 --is_wandb 1

# baseline
# python tests/main.py --run_group Quadruped --env dmc_quadruped --exp_name baseline --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --frame_stack 3 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra_bl --trans_optimization_epochs 200 --n_epochs_per_log 5 --n_epochs_per_eval 10 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 4 --encoder 1 --sample_cpu 0 --is_wandb 1 --seed 20240910


