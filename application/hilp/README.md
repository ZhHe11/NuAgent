# Foundation Policies with Hilbert Representations (Goal-Conditioned RL)

## Overview
This codebase contains the official implementation of **[Hilbert Foundation Policies](https://seohong.me/projects/hilp/)** (**HILPs**) for **goal-conditioned RL**.
The implementation is based on [HIQL: Offline Goal-Conditioned RL with Latent States as Actions](https://github.com/seohongpark/HIQL).

## Requirements
* Python 3.8
* MuJoCo 2.1.0

## Installation
```bash
# run the following command, shuld be able to see libEGL_nvidia installed
# refer to a similar trouble shooting: https://svl.stanford.edu/gibson2/docs/issues.html
ldconfig -p | grep EGL

# if not, run nvidia-smi to check the driver version id as ${version_id}, then run installation as
sudo apt-get install libnvidia-gl-${version_id}

# create conda environment
conda create --name hilp_gcrl python=3.8
conda activate hilp_gcrl
pip install -r requirements.txt --no-deps
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# fix issue if it is needed, make sure the version of flax > 0.6
pip install --upgrade flax
```

## Examples
```
# for each running, please add 'XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1' before python

# HILP on antmaze-large-diverse
# make sure you've install mujoco-py's dependencies: https://github.com/openai/mujoco-py?tab=readme-ov-file
# docker run -d --gpus device=0 -v /home/zhouming/projects/hilp-contrast/hilp_gcrl:/hilp/hilp_gcrl -v /home/zhouming/.d4rl:/root/.d4rl --name gcrl hilp /bin/bash -c "cd hilp_gcrl; python main.py --run_group EXP --agent_name hilp --algo_name hilp --seed 0 --env_name antmaze-large-diverse-v2 --use_rnd"
D4RL_SUPPRESS_IMPORT_ERROR=1 python main.py --run_group EXP --agent_name hilp --algo_name hilp --seed 0 --env_name antmaze-large-diverse-v2

# HILP-Plan on antmaze-large-diverse
python main.py --run_group EXP --agent_name hilp --algo_name hilp --planning_num_recursions 3 --seed 0 --env_name antmaze-large-diverse-v2

# HILP on antmaze-ultra-diverse
python main.py --run_group EXP --agent_name hilp --algo_name hilp --seed 0 --env_name antmaze-ultra-diverse-v0

# HILP on kitchen-partial
python main.py --run_group EXP --agent_name hilp --algo_name hilp --seed 0 --env_name kitchen-partial-v0 --train_steps 500000 --eval_interval 50000 --save_interval 500000 

# HILP on visual-kitchen-partial
mkdir -p data/d4rl_kitchen_rendered
python dataset_render.py --env_name kitchen-partial-v0
python main.py --run_group EXP --agent_name hilp --algo_name hilp --seed 0 --env_name visual-kitchen-partial-v0 --train_steps 500000 --eval_interval 50000 --save_interval 500000 --expectile 0.7 --skill_expectile 0.7 --batch_size 256 --encoder impala_small --p_aug 0.5
```

## License

MIT