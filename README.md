# universal-agent
Universal implementation of agent interface for RL evaluation.

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