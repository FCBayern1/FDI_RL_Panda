# Source Code

This directory contains all shared source code (libraries) used by experiments.

## Structure

- `envs/`: Environment definitions
- `models/`: Neural network models
- `controllers/`: PandaPower controllers
- `utils/`: Utility functions
- `plots/`: Plotting utilities

## Usage

Install as a package:

```bash
pip install -e .
```

Then import in experiments:

```python
from src.envs.gym_substation_env import make_substation_env
from src.networks.ddpg import Actor, Critic
```
