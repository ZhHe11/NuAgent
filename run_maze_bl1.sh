export CUDA_VISIBLE_DEVICES=6
export D4RL_SUPPRESS_IMPORT_ERROR=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
export MUJOCO_GL="osmesa"

# # # baseline
python tests/main.py --run_group MazeReady --env ant_maze --max_path_length 300 --seed 0 --traj_batch_size 16 --n_parallel 2 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 50 --n_epochs_per_eval 100 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 4 --sac_scale_reward 1 \
    --algo metra_bl --exp_name LSD --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 75 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --model_master_num_layers 2 --n_epochs 8000 --dual_reg 0 --spectral_normalization 1 --discrete 0

python tests/main.py --run_group MazeReady --env ant_maze --max_path_length 300 --seed 2 --traj_batch_size 16 --n_parallel 2 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 50 --n_epochs_per_eval 100 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 4 --sac_scale_reward 1 \
    --algo metra_bl --exp_name LSD --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 75 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --model_master_num_layers 2 --n_epochs 8000 --dual_reg 0 --spectral_normalization 1 --discrete 0

python tests/main.py --run_group MazeReady --env ant_maze --max_path_length 300 --seed 4 --traj_batch_size 16 --n_parallel 2 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 50 --n_epochs_per_eval 100 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 4 --sac_scale_reward 1 \
    --algo metra_bl --exp_name LSD --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 75 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --model_master_num_layers 2 --n_epochs 8000 --dual_reg 0 --spectral_normalization 1 --discrete 0



