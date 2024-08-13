export CUDA_VISIBLE_DEVICES=7
export MUJOCO_GL="osmesa"

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
python tests/main.py --run_group Debug_baseline --exp_name cl_her-buffer_sample_direction_goal \
    --env ant_maze --max_path_length 300 \
    --seed 0 --traj_batch_size 8 --n_parallel 4 \
    --normalizer_type off \
    --sac_max_buffer_size 100000 \
    --algo metra --trans_optimization_epochs 50 \
    --n_epochs_per_log 50 --n_epochs_per_eval 100 \
    --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 \
    --discrete 0 --dim_option 2 --unit_length 1 \
    --sac_scale_reward 1 --trans_minibatch_size 1024 --is_wandb 1 \
    --phi_type contrastive --policy_type her_reward --explore_type psro \
    --sample_type contrastive \




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


