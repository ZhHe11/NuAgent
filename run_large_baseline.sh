export CUDA_VISIBLE_DEVICES=5
export D4RL_SUPPRESS_IMPORT_ERROR=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
export MUJOCO_GL="osmesa"

# # baseline
python tests/main.py --run_group Large --env ant_maze_large --max_path_length 300 --seed 2 --traj_batch_size 16 --n_parallel 4 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 100 --n_epochs_per_eval 500 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 4 --sac_scale_reward 1 \
    --algo metra_bl --exp_name DIAYN --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 75 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 --model_master_num_layers 2 --n_epochs 20000 --inner 0 --unit_length 0 --dual_reg 0 --discrete 0 --common_lr 1e-5

python tests/main.py --run_group Large --env ant_maze_large --max_path_length 300 --seed 4 --traj_batch_size 16 --n_parallel 8 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 100 --n_epochs_per_eval 500 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 4 --sac_scale_reward 1 \
    --algo dads --exp_name dads --trans_optimization_epochs 150 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --model_master_num_layers 2 --n_epochs 20000 --inner 0 --unit_length 0 --dual_reg 0
