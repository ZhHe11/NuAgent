export CUDA_VISIBLE_DEVICES=7
export MUJOCO_GL="osmesa"

# # baseline 
python tests/main.py --run_group ant --env ant --max_path_length 200 --seed 4 --traj_batch_size 16 --n_parallel 2 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --discrete 0 --dim_option 2 \
    --algo dads --exp_name dads --num_her 0 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 --common_lr 1e-4 --n_epochs 6000 --save_pt_step 250 --inner 0 --unit_length 0 --dual_reg 0 --discrete 0 --dim_option 2

