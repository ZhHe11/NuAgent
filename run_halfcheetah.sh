export CUDA_VISIBLE_DEVICES=7
export MUJOCO_GL="osmesa"


# because the discrete is 1, wait for debug

# regret
python tests/main.py --run_group half_cheetah --env half_cheetah --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type preset --trans_optimization_epochs 50 --n_epochs_per_log 10 --n_epochs_per_eval 20 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --discrete 1 --dim_option 16


# python tests/main.py --run_group Maze --exp_name theta-dim2 --env ant_maze --max_path_length 300 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type off --sac_max_buffer_size 100000 --algo metra --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 100 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --sac_scale_reward 1 --trans_minibatch_size 1024 --is_wandb 0 --phi_type contrastive --policy_type her_reward --explore_type theta --sample_type contrastive --num_her 0 --_trans_phi_optimization_epochs 1 --target_theta 1e-3 --is_wandb 1


# baseline
python tests/main.py --run_group Debug --env half_cheetah --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --discrete 1 --dim_option 16

