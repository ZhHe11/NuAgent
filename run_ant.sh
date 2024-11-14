export CUDA_VISIBLE_DEVICES=2
export MUJOCO_GL="osmesa"

# baseline 
# python tests/main.py --run_group ant --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra_bl --discrete 0 --dim_option 2 --is_wandb 1


# # SZN
python tests/main.py --run_group ant --env ant --max_path_length 50 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --n_epochs_per_log 100 --n_epochs_per_eval 500 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --discrete 0 --dim_option 2 \
    --algo PSZP --exp_name SZN-tune --phi_type Projection --explore_type SZN --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 50 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 --common_lr 1e-3 --wandb_note 'unchange params' --SZN_w2 5 --SZN_w3 1 --SZN_window_size 5 --SZN_repeat_time 2 --Repr_temperature 0.5 --Repr_max_step 5 --save_pt_step 500



# # ours
#     --phi_type contrastive --policy_type her_reward --explore_type baseline --sample_type contrastive --num_her 0 --_trans_phi_optimization_epochs 1 --trans_optimization_epochs 200 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 
# # baseline
#     --phi_type baseline --policy_type baseline --explore_type baseline --sample_type baseline --num_her 0 --_trans_phi_optimization_epochs 1 --trans_optimization_epochs 200 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 


# python tests/main.py --run_group ant --env ant --exp_name baseline --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 10 --n_epochs_per_eval 20 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --discrete 0 --dim_option 2 --is_wandb 1 





