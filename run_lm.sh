export CUDA_VISIBLE_DEVICES=2
export D4RL_SUPPRESS_IMPORT_ERROR=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
export MUJOCO_GL="osmesa"

# regret
python tests/main.py --run_group LM-AS --env lm --max_path_length 300 --seed 0 --traj_batch_size 16 --n_parallel 4 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 20 --n_epochs_per_eval 20 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --sac_scale_reward 1 \
    --algo SZPC --exp_name Epsilon-ready-1 --phi_type Projection --explore_type SZN --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 50 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-3 --SZN_w2 0 --SZN_w3 0 --SZN_window_size 10 --SZN_repeat_time 5 --Repr_temperature 0 --z_unit 1 --model_master_num_layers 2 --wandb_note 'epsilon=0.1; z_unit=1; No w2, w3; t=0' 




    --algo PSZP --exp_name baseline --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 50 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --model_master_num_layers 2 






