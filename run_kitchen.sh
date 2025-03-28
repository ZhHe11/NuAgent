export CUDA_VISIBLE_DEVICES=6
export D4RL_SUPPRESS_IMPORT_ERROR=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
export MUJOCO_GL="osmesa"


# # # baseline
# python tests/main.py --run_group kitchen --env kitchen --max_path_length 50 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --num_video_repeats 1 --frame_stack 3 --sac_max_buffer_size 100000 --algo metra_bl --sac_lr_a -1 --trans_optimization_epochs 100 --n_epochs_per_log 25 --n_epochs_per_eval 25 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 1 --dim_option 24 --encoder 1 --sample_cpu 0 --is_wandb 1


# Policy
python tests/main.py --run_group kitchen --env kitchen --max_path_length 50 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --sac_max_buffer_size 100000  --n_epochs_per_log 25 --seed 2 --n_epochs_per_eval 25 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 1 --dim_option 24 --encoder 1 --sample_cpu 0 --sac_lr_a -1  --frame_stack 3  \
    --algo metra_bl_ours --exp_name ours3-kl_debug-no_softmax-sample8-w2_05-win_8_1 --phi_type baseline --explore_type SZN --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256  --SZN_w2 0.5 --SZN_w3 1 --SZN_window_size 8 --SZN_repeat_time 1 --Repr_temperature 0 --Repr_max_step 300 --z_unit 0 --model_master_num_layers 2 --n_epochs 2000 --wandb_note '' --save_pt_step 100  --SZN_std_min 1e-1 


    --algo metra_bl --exp_name metra_bl --sac_lr_a -1 --trans_optimization_epochs 100 --n_epochs_per_log 25 --n_epochs_per_eval 25 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 1 --dim_option 24 --encoder 1 --sample_cpu 0 --is_wandb 1




    --algo SZPC --exp_name kitchen-w2_5-window_5_2 --phi_type baseline --explore_type SZN --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 --common_lr 1e-4 --lr_te 1e-4 --dual_lr 1e-4 --SZN_w2 5 --SZN_w3 1 --SZN_window_size 5 --SZN_repeat_time 2 --Repr_temperature 0 --Repr_max_step 300 --z_unit 0 --model_master_num_layers 2 --n_epochs 2000 --wandb_note '' --save_pt_step 100  --seed 0  --SZN_std_min 1e-1


    --algo SZPC --exp_name kitchen-baseline --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 --common_lr 1e-4 --lr_te 1e-4 --dual_lr 1e-4 --SZN_w2 5 --SZN_w3 1 --SZN_window_size 15 --SZN_repeat_time 3 --Repr_temperature 0 --Repr_max_step 300 --z_unit 0 --model_master_num_layers 2 --n_epochs 2000 --wandb_note '' --save_pt_step 100  --seed 0 --SZN_std_min 1e-1








# python tests/main.py --run_group kitchen_debug  --env kitchen --max_path_length 50 --seed 0 --n_parallel 2 --normalizer_type off --num_video_repeats 1 --frame_stack 3 --sac_max_buffer_size 100000 --sac_lr_a -1 --n_epochs_per_log 100 --n_epochs_per_eval 100 --n_epochs_per_save 250 --n_epochs_per_pt_save 250 --encoder 1 --sample_cpu 0 \
#     --algo SZPC --exp_name Ours-lr1e4-w15r3 --phi_type Projection --explore_type SZN --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --lr_te 1e-4 --dual_lr 1e-4 --dual_lam 20 --SZN_w2 5 --SZN_w3 1 --SZN_window_size 15 --SZN_repeat_time 3 --Repr_temperature 0 --Repr_max_step 5 --z_unit 0 --model_master_num_layers 2 --n_epochs 1100  --save_pt_step 100  --seed 0 --dual_slack 1e-4  --dim_option 4


