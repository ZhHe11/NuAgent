export CUDA_VISIBLE_DEVICES=7
export MUJOCO_GL="osmesa"


# # regret
python tests/main.py --run_group humanoid --exp_name theta-onlybiaoding --env dmc_humanoid --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --frame_stack 3 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra --n_epochs_per_log 5 --n_epochs_per_eval 5 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --encoder 1 --sample_cpu 0 \
    --phi_type contrastive --policy_type her_reward --explore_type theta --sample_type contrastive --num_her 0 --_trans_phi_optimization_epochs 50 --trans_optimization_epochs 1 --target_theta 2e-5 --is_wandb 1 --trans_minibatch_size 1024 

# baseline
# python tests/main.py --run_group humanoid --env dmc_humanoid --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --frame_stack 3 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 10 --n_epochs_per_eval 20 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --encoder 1 --sample_cpu 0 \
#  --is_wandb 1
