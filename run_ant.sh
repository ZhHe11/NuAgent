export CUDA_VISIBLE_DEVICES=6
export MUJOCO_GL="osmesa"

# baseline 
# python tests/main.py --run_group ant --env ant --max_path_length 50 --seed 0 --traj_batch_size 16 --n_parallel 2 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --discrete 0 --dim_option 2 \
#     --algo metra_bl --exp_name baseline --phi_type baseline --explore_type baseline --policy_type baseline --sample_type baseline --num_her 0 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 --common_lr 1e-4 --n_epochs 8000 --save_pt_step 100 

# # # # SZN
python tests/main.py --run_group ant --env ant --max_path_length 50 --seed 0 --traj_batch_size 16 --n_parallel 2 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --n_epochs_per_log 100 --n_epochs_per_eval 100 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --discrete 0 --dim_option 2 \
    --algo SZPC --exp_name Ours-1e-43e-3 --phi_type Projection --explore_type SZN --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 50 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 --common_lr 1e-4 --lr_te 1e-4 --dual_lr 3e-3 --dual_lam 20 --SZN_w2 10 --SZN_w3 0 --SZN_window_size 10 --SZN_repeat_time 2 --Repr_temperature 0 --Repr_max_step 300 --z_unit 0 --model_master_num_layers 5 --n_epochs 8000 --save_pt_step 100  --seed 0



