export CUDA_VISIBLE_DEVICES=7
export MUJOCO_GL="osmesa"

# # baseline 
python tests/main.py --run_group ant --env ant --max_path_length 200 --seed 0 --traj_batch_size 16 --n_parallel 2 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --discrete 0 --dim_option 2 \
    --algo metra_bl --exp_name LSD --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 --common_lr 1e-4 --n_epochs 8000 --save_pt_step 100 --dual_reg 0 --spectral_normalization 1 --discrete 0 --dim_option 2

