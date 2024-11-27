export CUDA_VISIBLE_DEVICES=4
export MUJOCO_GL="osmesa"

# # baseline
# python tests/main.py --run_group Debug --env kitchen --max_path_length 50 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --num_video_repeats 1 --frame_stack 3 --sac_max_buffer_size 100000 --algo metra_bl --sac_lr_a -1 --trans_optimization_epochs 100 --n_epochs_per_log 25 --n_epochs_per_eval 250 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 1 --dim_option 24 --encoder 1 --sample_cpu 0 --is_wandb 1


python tests/main.py --run_group kitchen_debug  --env kitchen --max_path_length 50 --seed 0 --n_parallel 4 --normalizer_type off --num_video_repeats 1 --frame_stack 3 --sac_max_buffer_size 100000 --sac_lr_a -1 --n_epochs_per_log 25 --n_epochs_per_eval 100 --n_epochs_per_save 100 --n_epochs_per_pt_save 100 --encoder 1 --sample_cpu 0 \
    --algo SZPC --exp_name SZPC-d1-SZN --phi_type baseline --explore_type SZN --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 --common_lr 1e-4 --SZN_w2 5 --SZN_w3 2 --SZN_window_size 24 --SZN_repeat_time 1 --Repr_temperature 1 --Repr_max_step 100 --save_pt_step 500 --z_unit 0 --discrete 1 --traj_batch_size 8 --dim_option 24  --wandb_note 'discrete=1; baseline, I don;t believe the regret can not improve the sample effiency.' 











    # d=0; baseline
    --algo PSZP --exp_name PSZP-d0-baseline --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --SZN_w2 3 --SZN_w3 2 --SZN_window_size 10 --SZN_repeat_time 5 --Repr_temperature 0.5 --Repr_max_step 100 --save_pt_step 500 --z_unit 0 --traj_batch_size 16 --discrete 0 --dim_option 4  --wandb_note 'd=0;baseline, I want to prove our improvement in d=0' 



    # d=1; baseline
    --algo PSZP --exp_name PSZP-d1-baseline --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 --common_lr 1e-4 --SZN_w2 5 --SZN_w3 2 --SZN_window_size 24 --SZN_repeat_time 1 --Repr_temperature 1 --Repr_max_step 100 --save_pt_step 500 --z_unit 0 --discrete 1 --traj_batch_size 8 --dim_option 24  --wandb_note 'discrete=1; baseline, I don;t believe the regret can not improve the sample effiency.' 

    # d=1; ours
    --algo PSZP --exp_name PSZP-d1-2 --phi_type Projection --explore_type SZN --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 --common_lr 1e-4 --SZN_w2 5 --SZN_w3 2 --SZN_window_size 24 --SZN_repeat_time 1 --Repr_temperature 1 --Repr_max_step 100 --save_pt_step 500 --z_unit 0 --discrete 1 --traj_batch_size 8 --dim_option 24  --wandb_note 'discrete=1; SZN_repeat_time=1' 


    # d=0; ours
    --algo PSZP --exp_name PSZP-d0-7 --phi_type Projection --explore_type SZN --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --SZN_w2 3 --SZN_w3 2 --SZN_window_size 10 --SZN_repeat_time 5 --Repr_temperature 0.5 --Repr_max_step 100 --save_pt_step 500 --z_unit 0 --traj_batch_size 16 --discrete 0 --dim_option 4  --wandb_note '0.1 * cst_penalty_2; lr=1e-4;' 


















# ours
    # --algo SZN --phi_type contrastive --policy_type baseline --explore_type SZN --sample_type baseline --num_her 0 --_trans_phi_optimization_epochs 1 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 

# baseline:
    # --algo SZN --phi_type baseline --policy_type baseline --explore_type baseline --sample_type baseline --num_her 0 --_trans_phi_optimization_epochs 1 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 



# # baseline:
# python tests/main.py --run_group Baseline --env kitchen --exp_name ori-metra --max_path_length 50 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --num_video_repeats 1 --frame_stack 3 --sac_max_buffer_size 100000 --algo metra --sac_lr_a -1 --trans_optimization_epochs 100 --n_epochs_per_log 2 --n_epochs_per_eval 10 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 1 --dim_option 24 --encoder 1 --sample_cpu 0 --is_wandb 1


# # SGN-D;
# python tests/main.py --run_group SGN --exp_name SGN_E \
#     --env kitchen --num_video_repeats 1 --frame_stack 3  --max_path_length 50 --sac_lr_a -1 --encoder 1 \
#     --seed 0 --traj_batch_size 8 --n_parallel 4 \
#     --normalizer_type off \
#     --sac_max_buffer_size 100000 \
#     --algo metra --trans_optimization_epochs 100 \
#     --n_epochs_per_log 10 --n_epochs_per_eval 100 \
#     --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 \
#     --discrete 0 --dim_option 8 --unit_length 1 \
#     --sac_scale_reward 1 --trans_minibatch_size 1024 --is_wandb 1 \
#     --phi_type contrastive --policy_type her_reward --explore_type theta \
#     --sample_type contrastive --num_her 0 \

# # baseline:
# python tests/main.py --run_group kitchen --env kitchen --max_path_length 50 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --num_video_repeats 1 --frame_stack 3 --sac_max_buffer_size 100000 --algo metra --sac_lr_a -1 --trans_optimization_epochs 100 --n_epochs_per_log 25 --n_epochs_per_eval 250 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 1 --dim_option 24 --encoder 1 --sample_cpu 0 --is_wandb 1 --phi_type baseline --policy_type baseline --explore_type baseline --sample_type baseline --num_her 0

