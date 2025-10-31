# FDI RL Panda

Reinforcement Learning for Power Grid Transformer Control with False Data Injection (FDI) Attack Detection

Author: joshua
Date: 2025-10-31

---

## Overview

This project implements multi-agent reinforcement learning algorithms to control substation transformer connections in power grids. The system learns to:
- Disconnect transformers when necessary to avoid overheating/overloading
- Detect and bypass False Data Injection (FDI) attacks on temperature measurements
- Optimize power grid operation under attack conditions

### Algorithms Implemented

**Custom Training Loop Algorithms**:
- DQN (Deep Q-Network)
- DDPG (Deep Deterministic Policy Gradient)
- LSTM-DDPG (LSTM-enhanced DDPG)

**Stable-Baselines3 Algorithms**:
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
- DQN (SB3 version)

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install as package (recommended)
pip install -e .
```

### Run Experiments

**SB3 Experiments** (Recommended for new users):
```bash
# Test environment
python experiments/sb3/test_gym_env.py

# Quick training (debug mode - 5 episodes)
python experiments/sb3/quick_start.py --scenario debug --mode train

# Full training
python experiments/sb3/quick_start.py --scenario basic --mode train

# Evaluate model
python experiments/sb3/quick_start.py --scenario basic --mode eval \
    --model_path outputs/ppo/case14_XXXXXX/models/final_model
```

**Custom Training Loop Experiments**:
```bash
# DQN
python experiments/dqn/train.py

# DDPG
python experiments/ddpg/train.py

# LSTM-DDPG
python experiments/lstm_ddpg/train_v2.py
```

---

## Project Structure

```
FDI_RL_Panda/
├── experiments/          # All experiment code
│   ├── dqn/             # DQN experiments
│   ├── ddpg/            # DDPG experiments
│   ├── lstm_ddpg/       # LSTM-DDPG experiments
│   ├── baseline/        # Baseline experiments
│   └── sb3/             # Stable-Baselines3 experiments
│
├── src/                 # Shared source code
│   ├── envs/           # Environment definitions
│   │   └── gym_substation_env.py  # Gymnasium environment
│   ├── models/         # Neural network models
│   ├── controllers/    # PandaPower controllers
│   ├── utils/          # Utility functions
│   └── plots/          # Plotting utilities
│
├── outputs/            # Experiment outputs (by algorithm)
│   ├── dqn/            # DQN outputs
│   ├── ddpg/           # DDPG outputs
│   ├── lstm_ddpg/      # LSTM-DDPG outputs
│   ├── ppo/            # PPO outputs
│   ├── a2c/            # A2C outputs
│   ├── sac/            # SAC outputs
│   └── td3/            # TD3 outputs
│
├── docs/               # Detailed documentation
├── configs/            # Configuration files
├── scripts/            # Utility scripts
├── requirements.txt    # Dependencies
└── setup.py           # Package setup
```

---

## Key Features

### Multi-Agent RL Environment

**Observation Space** (12 dimensions per transformer):
- Loading percentage
- Measured temperature (potentially under FDI attack)
- Actual temperature
- Temperature trend
- Temperature jump detection
- Voltage level
- Connected line loadings (4 lines)
- Local load percentage
- Connection state
- Time step
- FDI attack indicator

**Action Space**:
- 0: Disconnect transformer
- 1: Keep connected

**Reward Function**:
- Correct disconnection (overheating/overloading): +10
- Missed disconnection: -15
- False disconnection: -8
- FDI detection bonus: +5
- Normal operation: +1

### FDI Attack Simulation

- Configurable attack probability
- Random temperature value injection
- Detection based on IEEE temperature model
- Detection threshold: 5°C difference between measured and actual

### Temperature Model

Based on IEEE standards:
```python
T = T_ambient + T_rated * (loading / 100.0) ** n
```
- T_ambient: 25°C (default)
- T_rated: 65°C (default)
- n: 1.6 (typical for transformers)

---

## Usage Examples

### Training a Model

```python
from stable_baselines3 import PPO
from src.envs.gym_substation_env import make_substation_env

# Create environment
env = make_substation_env(
    case_name='case14',
    max_steps=200,
    fdi_attack_prob=0.1
)

# Train model
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Save model
model.save("outputs/ppo/my_model")
```

### Loading and Evaluating

```python
from stable_baselines3 import PPO
from src.envs.gym_substation_env import make_substation_env

# Load model
model = PPO.load("outputs/ppo/case14_20251031/networks/final_model")

# Create environment
env = make_substation_env(case_name='case14')

# Evaluate
obs, info = env.reset()
for _ in range(200):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Total reward: {info['cumulative_reward']}")
print(f"Correct disconnections: {info['stats']['correct_disconnections']}")
print(f"FDI detections: {info['stats']['fdi_detections']}")
```

### Custom Training Loop (DDPG Example)

```python
import torch
from src.networks.ddpg import Actor, Critic
from src.envs.gym_substation_env import make_substation_env

# Create environment
env = make_substation_env(case_name='case14')

# Initialize networks
actor = Actor(state_dim=12, action_dim=2, hidden_dim=256)
critic = Critic(state_dim=12, action_dim=2, hidden_dim=256)

# Training loop
for episode in range(num_episodes):
    obs, info = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        # Select action
        action = actor(torch.FloatTensor(obs))

        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        # Update networks
        # ... (training logic)

        if terminated or truncated:
            break
```

---

## Experiment Outputs

All experiments save outputs to `outputs/{algorithm}/experiment_{timestamp}/`:

```
outputs/ddpg/experiment_20251031_120000/
├── models/              # Trained models (.pth or .zip)
├── logs/                # Training logs and metrics
├── results/             # Experiment results
└── experiment_info.txt  # Metadata
```

### Loading Models

**Custom Training Loop Models**:
```python
import torch
model = torch.load("outputs/ddpg/experiment_XXXXXX/networks/ddpg_models/actor_trafo_0_final.pth")
```

**SB3 Models**:
```python
from stable_baselines3 import PPO
model = PPO.load("outputs/ppo/case14_XXXXXX/networks/final_model")
```

---

## Configuration

### SB3 Scenarios

See `experiments/sb3/config_example.py` for predefined scenarios:

- **debug**: 5 episodes, 10 steps (quick testing)
- **basic**: 500 episodes, 200 steps (standard training)
- **high_fdi**: High FDI attack probability
- **conservative**: Lower disconnect threshold
- **aggressive**: Higher disconnect threshold
- **large**: Long training (2000 episodes)
- **scheduled_fdi**: Scheduled FDI attacks at specific times
- **a2c**: A2C algorithm configuration
- **dqn**: DQN algorithm configuration

### Custom Configuration

```python
from src.envs.gym_substation_env import make_substation_env

env = make_substation_env(
    case_name='case14',
    max_steps=200,
    T_ambient=25.0,
    T_rated=65.0,
    max_temperature=90.0,
    fdi_attack_prob=0.1,
    fdi_temp_range=(50.0, 95.0),
    fdi_detection_threshold=5.0
)
```

---

## Documentation

### Quick References
- **This README**: Main documentation with quick start and examples
- **experiments/README.md**: Experiment organization and running instructions
- **outputs/README.md**: Output file structure and model loading

### Detailed Documentation (in `docs/`)
- **QUICK_REFERENCE.md**: Quick reference guide for common operations
- **FINAL_STRUCTURE.md**: Complete project structure overview
- **OUTPUTS_RESTRUCTURE_COMPLETE.md**: Output reorganization details
- **OUTPUTS_RESTRUCTURE_PLAN.md**: Restructuring design document

---

## Development

### Project Setup

```bash
# Clone repository
git clone <repository_url>
cd FDI_RL_Panda

# Install dependencies
pip install -r requirements.txt

# Install as editable package
pip install -e .
```

### Running Tests

```bash
# Test environment
python experiments/sb3/test_gym_env.py

# Test basic functionality
python -c "from src.envs.gym_substation_env import make_substation_env; env = make_substation_env('case14'); print('Environment created successfully')"
```

### Adding New Algorithms

1. Create experiment directory: `experiments/my_algorithm/`
2. Create output directory: `outputs/my_algorithm/`
3. Implement training script with timestamp-based output:
```python
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = f"outputs/my_algorithm/experiment_{timestamp}"
```

---

## Technical Details

### Power Grid Simulation
- Based on PandaPower library
- IEEE case14 network (default)
- Support for multiple transformer configurations

### RL Implementation
- Gymnasium-compatible environment
- Multi-agent setup (independent agents per transformer)
- Continuous/discrete action spaces supported
- Extensive logging and metrics tracking

### FDI Attack Model
- Random injection at configurable probability
- Temperature-based attacks (most critical parameter)
- Detection via temperature model comparison
- Rewards for successful detection

---

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'src'`:

```bash
# Solution 1: Install as package
pip install -e .

# Solution 2: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/FDI_RL_Panda"
```

### Finding Experiment Outputs

```bash
# List all experiments for an algorithm
ls outputs/ddpg/

# View specific experiment
ls outputs/ddpg/experiment_20251031_120000/

# View experiment metadata
cat outputs/ddpg/experiment_20251031_120000/experiment_info.txt
```

### Model Loading Issues

Ensure you're using the correct loading method:
- Custom models: `torch.load(path)`
- SB3 models: `Algorithm.load(path)` (e.g., `PPO.load(path)`)

---

## Performance Tips

1. **Start with debug scenario** for quick testing
2. **Use SB3 algorithms** for faster development
3. **Monitor TensorBoard** for training progress:
   ```bash
   tensorboard --logdir outputs/ppo/case14_XXXXXX/tensorboard
   ```
4. **Adjust FDI probability** based on your research needs
5. **Use checkpoints** for long training runs

---

## Citation

If you use this code in your research, please cite:

```
@software{fdi_rl_panda,
  author = {joshua},
  title = {FDI RL Panda: RL for Power Grid Control with FDI Detection},
  year = {2025},
  url = {https://github.com/your-repo/FDI_RL_Panda}
}
```

---

## License

[Specify your license here]

---

## Contact

For questions or issues, please open an issue on GitHub or contact the author.

Author: joshua
Last Updated: 2025-10-31
