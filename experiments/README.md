# Experiments

This directory contains all experimental code organized by algorithm.

Author: joshua
Date: 2025-10-31

## Structure

```
experiments/
├── dqn/              # Deep Q-Network experiments
├── ddpg/             # Deep Deterministic Policy Gradient experiments
├── lstm_ddpg/        # LSTM-enhanced DDPG experiments
├── baseline/         # Baseline experiments
├── sb3/              # Stable-Baselines3 experiments
└── plot_utils.py     # Shared plotting utilities
```

## Algorithm Categories

### Custom Training Loop Algorithms

These experiments use custom training loops with PyTorch:

- **DQN** (`dqn/`): Deep Q-Network implementation
  - `train.py`: Training script
  - `test.py`: Testing/evaluation script

- **DDPG** (`ddpg/`): Deep Deterministic Policy Gradient
  - `train.py`: Training script
  - `test.py`: Testing/evaluation script

- **LSTM-DDPG** (`lstm_ddpg/`): LSTM-enhanced DDPG for temporal dependencies
  - `train_v2.py`: Training script with LSTM architecture
  - `test_v2.py`: Testing/evaluation script

- **Baseline** (`baseline/`): Baseline experiments for comparison

### Stable-Baselines3 Algorithms

These experiments use the SB3 library (`sb3/`):

- `train_sb3.py`: General training script for PPO, A2C, DQN, SAC, TD3
- `test_gym_env.py`: Environment validation
- `quick_start.py`: Quick experiment launcher
- `config_example.py`: Predefined experiment configurations

## Running Experiments

### Custom Training Loop Experiments

```bash
# Train DQN
python experiments/dqn/train.py

# Train DDPG
python experiments/ddpg/train.py

# Train LSTM-DDPG
python experiments/lstm_ddpg/train_v2.py
```

### SB3 Experiments

```bash
# Test environment
python experiments/sb3/test_gym_env.py

# Quick training
python experiments/sb3/quick_start.py --scenario debug --mode train

# Full training
python experiments/sb3/quick_start.py --scenario basic --mode train

# Evaluate model
python experiments/sb3/quick_start.py --scenario basic --mode eval \
    --model_path outputs/ppo/case14_XXXXXX/models/final_model
```

## Output Locations

All experiment outputs are saved to the `outputs/` directory:

- Custom training loops: `outputs/{algorithm}/experiment_{timestamp}/`
- SB3 experiments: `outputs/{algorithm}/case14_{timestamp}/`

See `outputs/README.md` for more details.

## Documentation

- `docs/README_GYM_ENV.md`: Complete SB3 environment documentation
- `docs/MIGRATION_GUIDE.md`: Guide for migrating between implementations

Author: joshua
Date: 2025-10-31
