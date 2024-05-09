# METRA: Scalable Unsupervised RL with Metric-Aware Abstraction

This repository contains the official implementation of **METRA: Scalable Unsupervised RL with Metric-Aware Abstraction**.
The implementation is based on
[Lipschitz-constrained Unsupervised Skill Discovery](https://github.com/seohongpark/LSD).

## Examples

```python
# run maze env
PYTHONPATH=. MUJOCO_GL=egl python examples/metra_gcrl.py --eval-plot-axis -50 50 -50 50 --use-wandb 0 --normalizer-type off --env-name maze --render 1

# run ant env
PYTHONPATH=. MUJOCO_GL=egl python examples/metra_gcrl.py --eval-plot-axis -50 50 -50 50 --use-wandb 0 --normalizer-type preset
```