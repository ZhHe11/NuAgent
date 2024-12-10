export CUDA_VISIBLE_DEVICES=5
export D4RL_SUPPRESS_IMPORT_ERROR=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
export MUJOCO_GL="osmesa"

# # # baseline
# python tests/main.py --run_group MazeReady --env ant_maze --max_path_length 300 --seed 4 --traj_batch_size 16 --n_parallel 2 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 50 --n_epochs_per_eval 100 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 4 --sac_scale_reward 1 \
#     --algo metra_bl --exp_name Baseline-dim4 --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 75 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --model_master_num_layers 2 --n_epochs 8000

python tests/main.py --run_group Large --env ant_maze_large --max_path_length 300 --seed 0 --traj_batch_size 16 --n_parallel 4 --normalizer_type off --trans_optimization_epochs 75 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --sac_max_buffer_size 1000000 --algo metra_bl --discrete 0 --dim_option 4 --trans_minibatch_size 1024



# # # regret
# python tests/main.py --run_group MazeReady --env ant_maze_large --max_path_length 300 --seed 0 --traj_batch_size 16 --n_parallel 2 --normalizer_type off --sac_max_buffer_size 1000000  --n_epochs_per_log 50 --n_epochs_per_eval 200 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 4 --sac_scale_reward 1 \
#     --algo SZPC --exp_name w10_3-win15_2-Rmax10-dual_slack --phi_type Projection --explore_type SZN --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 75 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 \
#     --common_lr 1e-4 --lr_te 1e-3 --dual_lr 1e-4 \
#     --SZN_w2 10 --SZN_w3 3 --SZN_window_size 15 --SZN_repeat_time 3 \
#     --Repr_temperature 0 --z_unit 0 --model_master_num_layers 2 --n_epochs 8000 --Repr_max_step 10 \
#     --dual_slack 1e-3

