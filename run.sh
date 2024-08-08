export CUDA_VISIBLE_DEVICES=7
export MUJOCO_GL="osmesa"
python tests/main.py --run_group Debug_baseline --exp_name contrastive-bs-norm \
    --env ant_maze --max_path_length 300 \
    --seed 0 --traj_batch_size 8 --n_parallel 4 \
    --normalizer_type off \
    --sac_max_buffer_size 100000 \
    --algo metra --trans_optimization_epochs 50 \
    --n_epochs_per_log 50 --n_epochs_per_eval 100 \
    --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 \
    --discrete 0 --dim_option 2 --unit_length 1 \
    --phi_type contrastive --policy_type baseline --explore_type freeze \
    --sac_scale_reward 1 --trans_minibatch_size 1024 \
