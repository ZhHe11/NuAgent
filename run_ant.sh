export CUDA_VISIBLE_DEVICES=7
export MUJOCO_GL="osmesa"


# SZN
python tests/main.py --run_group Debug --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 1 --n_epochs_per_log 25 --n_epochs_per_eval 50 --n_epochs_per_save 5000 --sac_max_buffer_size 1000000 --algo SZN --discrete 0 --dim_option 2 \
    --exp_name SZN_v2_epoch_t3-C2 --phi_type contrastive --policy_type baseline --explore_type SZN --sample_type baseline --num_her 0 --_trans_phi_optimization_epochs 50 --target_theta 1 --is_wandb 1 --trans_minibatch_size 1024 

# # ours
#     --phi_type contrastive --policy_type her_reward --explore_type baseline --sample_type contrastive --num_her 0 --_trans_phi_optimization_epochs 1 --trans_optimization_epochs 200 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 
# # baseline
#     --phi_type baseline --policy_type baseline --explore_type baseline --sample_type baseline --num_her 0 --_trans_phi_optimization_epochs 1 --trans_optimization_epochs 200 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 


# python tests/main.py --run_group ant --env ant --exp_name baseline --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 10 --n_epochs_per_eval 20 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --discrete 0 --dim_option 2 --is_wandb 1 

