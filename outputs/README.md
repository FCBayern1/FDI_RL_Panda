# Experiment Outputs

This directory contains all experiment outputs organized by algorithm.

Author: joshua
Date: 2025-10-31

## Structure

```
outputs/
├── dqn/              # Deep Q-Network experiments
├── ddpg/             # Deep Deterministic Policy Gradient experiments
├── lstm_ddpg/        # LSTM-enhanced DDPG experiments
├── baseline/         # Baseline experiments
├── ppo/              # Proximal Policy Optimization (SB3)
├── a2c/              # Advantage Actor-Critic (SB3)
├── sac/              # Soft Actor-Critic (SB3)
├── td3/              # Twin Delayed DDPG (SB3)
└── misc/             # Miscellaneous outputs
```

## Experiment Organization

Each experiment follows this structure:

```
algorithm/
└── experiment_YYYYMMDD_HHMMSS/
    ├── models/              # Trained model files (.pth, .zip)
    ├── logs/                # Training logs and metrics
    ├── results/             # Experiment results and data
    └── experiment_info.txt  # Experiment metadata
```

## Algorithm Categories

### Custom Training Loop Algorithms
- **DQN**: Deep Q-Network with custom training loop
- **DDPG**: Deep Deterministic Policy Gradient
- **LSTM-DDPG**: LSTM-enhanced DDPG for temporal dependencies
- **Baseline**: Baseline experiments for comparison

### Stable-Baselines3 Algorithms
- **PPO**: Proximal Policy Optimization
- **A2C**: Advantage Actor-Critic
- **SAC**: Soft Actor-Critic (continuous actions)
- **TD3**: Twin Delayed DDPG (continuous actions)

## Loading Models

### Custom Training Loop Models (PyTorch)

```python
import torch

model_path = "outputs/ddpg/experiment_20251031_110819/networks/ddpg_models/actor_trafo_0_final.pth"
model = torch.load(model_path)
```

### SB3 Models

```python
from stable_baselines3 import PPO

model = PPO.load("outputs/ppo/case14_20251031_120000/networks/final_model")
```

## Creating New Experiments

When running new experiments, use timestamp format:

```python
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = f"outputs/my_algorithm/experiment_{timestamp}"
```

## Finding Experiments

```bash
# List all experiments for an algorithm
ls outputs/ddpg/

# View specific experiment details
ls outputs/ddpg/experiment_20251031_110819/

# View experiment metadata
cat outputs/ddpg/experiment_20251031_110819/experiment_info.txt

# Find all model files
find outputs/ddpg -name "*.pth"
```

## Git

This directory is in `.gitignore`. Do not commit large model files.

Author: joshua
Date: 2025-10-31
