export CUDA_VISIBLE_DEVICES=
export D4RL_SUPPRESS_IMPORT_ERROR=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
export MUJOCO_GL="osmesa"


python tests/main.py --run_group LittleMaze_Exp --env lm --max_path_length 300 --seed 0 --traj_batch_size 16 --n_parallel 2 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 100 --n_epochs_per_eval 50 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --sac_scale_reward 1 \
    --algo metra_bl --exp_name baseline --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --lr_te 1e-4 --n_epochs 2000 --wandb_note 'Epsilon_Decay' --save_pt_step 100


python tests/main.py --run_group LM-ready --env lm --max_path_length 300 --seed 2 --traj_batch_size 16 --n_parallel 4 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 100 --n_epochs_per_eval 50 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --sac_scale_reward 1 \
    --algo metra_bl --exp_name baseline --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --lr_te 1e-4 --n_epochs 2000 --wandb_note 'Epsilon_Decay' --save_pt_step 100


python tests/main.py --run_group LM-ready --env lm --max_path_length 300 --seed 4 --traj_batch_size 16 --n_parallel 4 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 100 --n_epochs_per_eval 50 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --sac_scale_reward 1 \
    --algo metra_bl --exp_name baseline --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --lr_te 1e-4 --n_epochs 2000 --wandb_note 'Epsilon_Decay' --save_pt_step 100


python tests/main.py --run_group LM-ready --env lm --max_path_length 300 --seed 8 --traj_batch_size 16 --n_parallel 4 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 100 --n_epochs_per_eval 50 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --sac_scale_reward 1 \
    --algo metra_bl --exp_name baseline --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --lr_te 1e-4 --n_epochs 2000 --wandb_note 'Epsilon_Decay' --save_pt_step 100


python tests/main.py --run_group LM-ready --env lm --max_path_length 300 --seed 16 --traj_batch_size 16 --n_parallel 4 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 100 --n_epochs_per_eval 50 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --sac_scale_reward 1 \
    --algo metra_bl --exp_name baseline --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --lr_te 1e-4 --n_epochs 2000 --wandb_note 'Epsilon_Decay' --save_pt_step 100




# regret
# python tests/main.py --run_group LM-ready --env lm --max_path_length 300 --seed 0 --traj_batch_size 16 --n_parallel 2 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 100 --n_epochs_per_eval 100 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --sac_scale_reward 1 \
#     --algo SZPC --exp_name Ours-OnlyRegretScale-lr_te_1e_3-wod --phi_type Projection --explore_type SZN --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --lr_te 1e-3 --dual_lr 1e-4 --dual_lam 20 --SZN_w2 5 --SZN_w3 3 --SZN_window_size 50 --SZN_repeat_time 25 --Repr_temperature 0 --Repr_max_step 5 --z_unit 0 --model_master_num_layers 2 --n_epochs 2000 --wandb_note '' --save_pt_step 100  --seed 8




    --algo SZPC --exp_name baseline-lr1e4 --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --lr_te 1e-4 --SZN_w2 10 --SZN_w3 3 --SZN_window_size 10 --SZN_repeat_time 5 --Repr_temperature 2 --z_unit 0 --model_master_num_layers 2 --n_epochs 2000 --wandb_note 'Epsilon_Decay' --save_pt_step 100




    --algo PSZP --exp_name baseline --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 50 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --model_master_num_layers 2 






