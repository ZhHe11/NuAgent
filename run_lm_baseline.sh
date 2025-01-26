export CUDA_VISIBLE_DEVICES=5 
export D4RL_SUPPRESS_IMPORT_ERROR=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
export MUJOCO_GL="osmesa"

python tests/main.py --run_group LM-ready --env lm --max_path_length 300 --seed 2 --traj_batch_size 16 --n_parallel 4 --normalizer_type off --sac_max_buffer_size 3000000 --n_epochs_per_log 100 --n_epochs_per_eval 50 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --sac_scale_reward 1 --algo dads --exp_name dads --trans_optimization_epochs 100 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --n_epochs 2000 --save_pt_step 100 --inner 0 --unit_length 0 --dual_reg 0 --discrete 0 --dim_option 2

python tests/main.py --run_group LM-ready --env lm --max_path_length 300 --seed 0 --traj_batch_size 16 --n_parallel 4 --normalizer_type off --sac_max_buffer_size 3000000 --n_epochs_per_log 100 --n_epochs_per_eval 50 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --sac_scale_reward 1 --algo metra_bl --exp_name DIAYN --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 100 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --lr_te 1e-4 --n_epochs 2000 --save_pt_step 100 --inner 0 --unit_length 0 --dual_reg 0 --discrete 0 --dim_option 2

