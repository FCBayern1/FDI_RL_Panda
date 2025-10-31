# Stable-Baselines3 Experiments

New experiments using Stable-Baselines3 library with standard Gym interface.

## Files

- `train_sb3.py`: Training and evaluation script
- `test_gym_env.py`: Environment testing
- `quick_start.py`: Quick start with predefined configs
- `config_example.py`: Configuration examples

## Quick Start

```bash
# Test environment
python test_gym_env.py

# Train with basic config
python quick_start.py --scenario basic --mode train

# Evaluate model
python quick_start.py --scenario basic --mode eval --model_path <path>
```

## Documentation

See `docs/README_GYM_ENV.md` for detailed documentation.
