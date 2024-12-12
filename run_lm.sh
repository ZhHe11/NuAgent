export CUDA_VISIBLE_DEVICES=7
export D4RL_SUPPRESS_IMPORT_ERROR=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
export MUJOCO_GL="osmesa"


# regret
python tests/main.py --run_group LM-ready --env lm --max_path_length 300 --seed 0 --traj_batch_size 16 --n_parallel 4 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 100 --n_epochs_per_eval 50 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --sac_scale_reward 1 \
    --algo SZPC --exp_name cic-t1 --phi_type Projection --explore_type SZN --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --lr_te 1e-3 --dual_lr 1e-4 --dual_lam 20 --SZN_w2 5 --SZN_w3 3 --SZN_window_size 15 --SZN_repeat_time 3 --Repr_temperature 1 --Repr_max_step 5 --z_unit 0 --model_master_num_layers 2 --n_epochs 1100 --wandb_note '' --save_pt_step 100  --seed 0 --dual_slack 1e-4 --SZN_std_min 1e-1

# python tests/main.py --run_group LM-ready --env lm --max_path_length 300 --seed 0 --traj_batch_size 16 --n_parallel 2 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 100 --n_epochs_per_eval 50 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --sac_scale_reward 1 \
#     --algo SZPC --exp_name Ours-w15r3 --phi_type Projection --explore_type SZN --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --lr_te 1e-3 --dual_lr 1e-4 --dual_lam 20 --SZN_w2 5 --SZN_w3 1 --SZN_window_size 15 --SZN_repeat_time 3 --Repr_temperature 0 --Repr_max_step 5 --z_unit 0 --model_master_num_layers 2 --n_epochs 1100 --wandb_note '' --save_pt_step 100  --seed 2 --dual_slack 1e-1 --SZN_std_min 1e-1

# python tests/main.py --run_group LM-ready --env lm --max_path_length 300 --seed 0 --traj_batch_size 16 --n_parallel 2 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 100 --n_epochs_per_eval 50 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --sac_scale_reward 1 \
#     --algo SZPC --exp_name Ours-w15r3 --phi_type Projection --explore_type SZN --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --lr_te 1e-3 --dual_lr 1e-4 --dual_lam 20 --SZN_w2 5 --SZN_w3 1 --SZN_window_size 15 --SZN_repeat_time 3 --Repr_temperature 0 --Repr_max_step 5 --z_unit 0 --model_master_num_layers 2 --n_epochs 1100 --wandb_note '' --save_pt_step 100  --seed 4 --dual_slack 1e-1 --SZN_std_min 1e-1

