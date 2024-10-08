export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL="osmesa"

# regret
python tests/main.py --run_group Maze --env ant_maze --max_path_length 100 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type off --sac_max_buffer_size 300000  --n_epochs_per_log 10 --n_epochs_per_eval 50 --n_epochs_per_save 100 --n_epochs_per_pt_save 100 --discrete 0 --dim_option 2 --sac_scale_reward 1 \
    --algo SZN_Z --exp_name SZN_Z-Cv5 --phi_type contrastive_v5 --explore_type SZN --policy_type baseline --sample_type baseline --num_her 0 --trans_optimization_epochs 50 --target_theta 1 --is_wandb 1 --trans_minibatch_size 256 

# baseline
# python tests/main.py --run_group Debug --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --discrete 0 --dim_option 2


# soft_update
# python tests/main.py --run_group Debug_baseline --exp_name soft_update_3 \
#     --env ant_maze --max_path_length 300 \
#     --seed 0 --traj_batch_size 8 --n_parallel 4 \
#     --normalizer_type off \
#     --sac_max_buffer_size 100000 \
#     --algo metra --trans_optimization_epochs 50 \
#     --n_epochs_per_log 50 --n_epochs_per_eval 100 \
#     --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 \
#     --discrete 0 --dim_option 2 --unit_length 1 \
#     --sac_scale_reward 1 --trans_minibatch_size 1024 --is_wandb 1 \
#     --phi_type soft_update --policy_type her_reward --explore_type freeze \
#     --sample_type her_reward \
    

# contrastive learning;
# python tests/main.py --run_group Debug_baseline --exp_name contrastive_1 \
#     --env ant_maze --max_path_length 300 \
#     --seed 0 --traj_batch_size 8 --n_parallel 4 \
#     --normalizer_type off \
#     --sac_max_buffer_size 100000 \
#     --algo metra --trans_optimization_epochs 50 \
#     --n_epochs_per_log 50 --n_epochs_per_eval 100 \
#     --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 \
#     --discrete 0 --dim_option 2 --unit_length 1 \
#     --sac_scale_reward 1 --trans_minibatch_size 1024 --is_wandb 1 \
#     --phi_type contrastive --policy_type her_reward --explore_type freeze \
#     --sample_type contrastive \

# sample goal to explore;
# python tests/main.py --run_group Debug_baseline --exp_name psro-R_sum_phi_g-Wait \
#     --env ant_maze --max_path_length 300 \
#     --seed 0 --traj_batch_size 8 --n_parallel 4 \
#     --normalizer_type off \
#     --sac_max_buffer_size 100000 \
#     --algo metra --trans_optimization_epochs 50 \
#     --n_epochs_per_log 50 --n_epochs_per_eval 100 \
#     --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 \
#     --discrete 0 --dim_option 2 --unit_length 1 \
#     --sac_scale_reward 1 --trans_minibatch_size 1024 --is_wandb 1 \
#     --phi_type contrastive --policy_type her_reward --explore_type psro \
#     --sample_type contrastive \


### for sample method type
# # determined sample baseline;
# python tests/main.py --run_group Debug_baseline --exp_name SGN-psro \
#     --env ant_maze --max_path_length 300 \
#     --seed 0 --traj_batch_size 8 --n_parallel 4 \
#     --normalizer_type off \
#     --sac_max_buffer_size 100000 \
#     --algo metra --trans_optimization_epochs 50 \
#     --n_epochs_per_log 50 --n_epochs_per_eval 100 \
#     --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 \
#     --discrete 0 --dim_option 2 --unit_length 1 \
#     --sac_scale_reward 1 --trans_minibatch_size 1024 --is_wandb 1 \
#     --phi_type contrastive --policy_type her_reward --explore_type psro \
#     --sample_type contrastive 

# # determined sample baseline;
# python tests/main.py --run_group SGN_dim4 --exp_name SGN_D_large_neg-more_high_mean-low_std_negweight1 \
#     --env ant_maze --max_path_length 300 \
#     --seed 0 --traj_batch_size 16 --n_parallel 4 \
#     --normalizer_type off \
#     --sac_max_buffer_size 100000 \
#     --algo metra --trans_optimization_epochs 50 \
#     --n_epochs_per_log 100 --n_epochs_per_eval 100 \
#     --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 \
#     --discrete 0 --dim_option 4 --unit_length 1 \
#     --sac_scale_reward 1 --trans_minibatch_size 1024 --is_wandb 1 \
#     --phi_type contrastive --policy_type her_reward --explore_type theta \
#     --sample_type contrastive --num_her 0 \

 
# SGN-C;
# python tests/main.py --run_group SGN_dim4 --exp_name SGN_C_adaptive-direction \
#     --env ant_maze --max_path_length 300 \
#     --seed 0 --traj_batch_size 16 --n_parallel 4 \
#     --normalizer_type off \
#     --sac_max_buffer_size 100000 \
#     --algo metra --trans_optimization_epochs 50 \
#     --n_epochs_per_log 100 --n_epochs_per_eval 100 \
#     --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 \
#     --discrete 0 --dim_option 4 --unit_length 1 \
#     --sac_scale_reward 1 --trans_minibatch_size 1024 --is_wandb 1 \
#     --phi_type contrastive --policy_type her_reward --explore_type wait \
#     --sample_type contrastive --num_her 0 \


# # her_resample
# python tests/main.py --run_group Debug_baseline --exp_name her_resample \
#     --env ant_maze --max_path_length 300 \
#     --seed 0 --traj_batch_size 8 --n_parallel 4 \
#     --normalizer_type off \
#     --sac_max_buffer_size 100000 \
#     --algo metra --trans_optimization_epochs 50 \
#     --n_epochs_per_log 50 --n_epochs_per_eval 100 \
#     --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 \
#     --discrete 0 --dim_option 2 --unit_length 1 \
#     --phi_type soft_update --policy_type her_reward --explore_type freeze \
#     --sac_scale_reward 1 --trans_minibatch_size 1024 --is_wandb 1 \

# # # baseline
# python tests/main.py --run_group Debug_baseline --exp_name baseline_dim4_sample16 \
#     --env ant_maze --max_path_length 300 \
#     --seed 0 --traj_batch_size 16 --n_parallel 4 \
#     --normalizer_type off \
#     --sac_max_buffer_size 100000 \
#     --algo metra --trans_optimization_epochs 50 \
#     --n_epochs_per_log 50 --n_epochs_per_eval 100 \
#     --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 \
#     --discrete 0 --dim_option 4 --unit_length 1 \
#     --sac_scale_reward 1 --trans_minibatch_size 1024 --is_wandb 1 \
#     --phi_type baseline --policy_type baseline --explore_type baseline \
#     --sample_type baseline \
#     --seed 0 \
