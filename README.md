# universal-agent
Universal implementation of an agent interface for RL evaluation, all agents (in one group) will behave as a big agent, 
and an evaluation framework on top of it is policy-/model-agnostic.


![Design](/assets/Diagram.png)

## Installation

```bash
conda crate -n univsersal-agent python=3.8 -y
conda activate universal-agent
pip install -r requirements.txt
make compile
```

## Quick Start

```bash
# start env server
python start_env_server.py -p 50052

# start multiple env clients, each with an universal-agent instance
python start_env_client.py --hostname localhost -p 50052 --env-id CartPole-v1
```

## TODOs

💬: in discussion

🕓: implemented but lightly tested.

✅ thoroughly-tested. In many cases, we verified against known values and/or reproduced results from papers.

Status | Desc | Use 
--- | --- | ---
💬 | Suite for episode-level evaluation | ...
🕓 | Multi-agent Application | ...
🕓 | torch-based Algo Library | tianshou, RLLib, etc.
