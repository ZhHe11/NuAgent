export CUDA_VISIBLE_DEVICES=4
export D4RL_SUPPRESS_IMPORT_ERROR=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
export MUJOCO_GL="osmesa"

# regret
python tests/main.py --run_group LittleMaze --env lm --max_path_length 300 --seed 0 --traj_batch_size 16 --n_parallel 4 --normalizer_type off --sac_max_buffer_size 3000000  --n_epochs_per_log 10 --n_epochs_per_eval 50 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --sac_scale_reward 1 \
    --algo PSZP --exp_name SZN-t05-no_g_dir --phi_type Projection --explore_type SZN --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 50 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-3 --model_master_num_layers 2 --wandb_note 'last on t = 01 this time 05'


    --algo PSZP --exp_name baseline --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 50 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 --common_lr 1e-4 --model_master_num_layers 2 













