"""

"""

# =============================================================================
#  1:  -  FDI 
# =============================================================================
BASIC_CONFIG = {
    'env': {
        'network_case': 'case14',
        'max_steps': 200,
        'fdi_attack_prob': 0.05,  # 5% FDI 
        'max_temperature': 90.0,
        'fdi_temp_range': (80.0, 120.0),
    },
    'training': {
        'algorithm': 'ppo',
        'total_timesteps': 100000,
        'n_envs': 4,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'gamma': 0.99,
    }
}

# =============================================================================
#  2:  FDI  -  FDI 
# =============================================================================
HIGH_FDI_CONFIG = {
    'env': {
        'network_case': 'case14',
        'max_steps': 200,
        'fdi_attack_prob': 0.3,  # 30% FDI 
        'max_temperature': 90.0,
        'fdi_temp_range': (70.0, 150.0),  # 
        #  FDI 
        'reward_fdi_detection': 10.0,
        'reward_correct_disconnect': 12.0,
        'penalty_missed_disconnect': 20.0,
    },
    'training': {
        'algorithm': 'ppo',
        'total_timesteps': 200000,  # 
        'n_envs': 8,
        'learning_rate': 1e-4,  # 
        'n_steps': 2048,
        'batch_size': 128,
        'gamma': 0.99,
    }
}

# =============================================================================
#  3:  - 
# =============================================================================
CONSERVATIVE_CONFIG = {
    'env': {
        'network_case': 'case14',
        'max_steps': 200,
        'fdi_attack_prob': 0.1,
        'max_temperature': 80.0,  # 
        'fdi_temp_range': (75.0, 110.0),
        # 
        'reward_correct_disconnect': 15.0,
        'penalty_missed_disconnect': 25.0,  # 
        'penalty_false_disconnect': 5.0,    # 
    },
    'training': {
        'algorithm': 'ppo',
        'total_timesteps': 150000,
        'n_envs': 6,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'gamma': 0.99,
    }
}

# =============================================================================
#  4:  - 
# =============================================================================
AGGRESSIVE_CONFIG = {
    'env': {
        'network_case': 'case14',
        'max_steps': 200,
        'fdi_attack_prob': 0.1,
        'max_temperature': 100.0,  # 
        'fdi_temp_range': (90.0, 130.0),
        # 
        'reward_stay_connected': 5.0,       # 
        'penalty_false_disconnect': 15.0,   # 
        'penalty_missed_disconnect': 10.0,  # 
    },
    'training': {
        'algorithm': 'ppo',
        'total_timesteps': 150000,
        'n_envs': 6,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'gamma': 0.99,
    }
}

# =============================================================================
#  5:  - Case30
# =============================================================================
LARGE_NETWORK_CONFIG = {
    'env': {
        'network_case': 'case30',  # 
        'max_steps': 300,          #  episode
        'fdi_attack_prob': 0.15,
        'max_temperature': 90.0,
        'fdi_temp_range': (80.0, 120.0),
    },
    'training': {
        'algorithm': 'ppo',
        'total_timesteps': 300000,  # 
        'n_envs': 8,                # 
        'learning_rate': 3e-4,
        'n_steps': 4096,            # 
        'batch_size': 128,
        'gamma': 0.99,
    }
}

# =============================================================================
#  6:  - 
# =============================================================================
DEBUG_CONFIG = {
    'env': {
        'network_case': 'case14',
        'max_steps': 50,           #  episode
        'fdi_attack_prob': 0.2,
        'max_temperature': 90.0,
        'fdi_temp_range': (85.0, 115.0),
        'render_mode': 'human',    # 
    },
    'training': {
        'algorithm': 'ppo',
        'total_timesteps': 10000,  # 
        'n_envs': 2,
        'learning_rate': 3e-4,
        'n_steps': 512,
        'batch_size': 32,
        'gamma': 0.99,
    }
}

# =============================================================================
#  7:  FDI  - 
# =============================================================================
SCHEDULED_FDI_CONFIG = {
    'env': {
        'network_case': 'case14',
        'max_steps': 100,
        'fdi_attack_prob': 0.0,  #  FDI
        'max_temperature': 90.0,
        #  FDI 
        'fdi_schedule': [
            (10, 0, 130.0),  #  10  0 130°C
            (20, 0, 140.0),  #  20  0 140°C
            (30, 1, 125.0),  #  30  1 125°C
            (40, 1, 135.0),  #  40  1 135°C
            (50, 0, 120.0),  #  50  0 120°C
            (60, 1, 115.0),  #  60  1 115°C
        ],
    },
    'training': {
        'algorithm': 'ppo',
        'total_timesteps': 50000,
        'n_envs': 4,
        'learning_rate': 3e-4,
        'n_steps': 1024,
        'batch_size': 64,
        'gamma': 0.99,
    }
}

# =============================================================================
#  8: A2C  - 
# =============================================================================
A2C_CONFIG = {
    'env': {
        'network_case': 'case14',
        'max_steps': 200,
        'fdi_attack_prob': 0.1,
        'max_temperature': 90.0,
        'fdi_temp_range': (80.0, 120.0),
    },
    'training': {
        'algorithm': 'a2c',
        'total_timesteps': 100000,
        'n_envs': 16,  # A2C 
        'learning_rate': 7e-4,
        'n_steps': 5,  # A2C 
        'gamma': 0.99,
    }
}

# =============================================================================
#  9: DQN  - 
# =============================================================================
DQN_CONFIG = {
    'env': {
        'network_case': 'case14',
        'max_steps': 200,
        'fdi_attack_prob': 0.1,
        'max_temperature': 90.0,
        'fdi_temp_range': (80.0, 120.0),
    },
    'training': {
        'algorithm': 'dqn',
        'total_timesteps': 100000,
        'n_envs': 1,  # DQN 
        'learning_rate': 1e-4,
        'batch_size': 32,
        'gamma': 0.99,
    }
}

# =============================================================================
# 
# =============================================================================

def get_config(scenario_name):
    """
    

    Args:
        scenario_name: 

    Returns:
        
    """
    configs = {
        'basic': BASIC_CONFIG,
        'high_fdi': HIGH_FDI_CONFIG,
        'conservative': CONSERVATIVE_CONFIG,
        'aggressive': AGGRESSIVE_CONFIG,
        'large': LARGE_NETWORK_CONFIG,
        'debug': DEBUG_CONFIG,
        'scheduled_fdi': SCHEDULED_FDI_CONFIG,
        'a2c': A2C_CONFIG,
        'dqn': DQN_CONFIG,
    }

    if scenario_name not in configs:
        raise ValueError(f"Unknown scenario: {scenario_name}. "
                         f"Available: {list(configs.keys())}")

    return configs[scenario_name]

def print_config(config):
    """"""
    print("\n" + "="*80)
    print("Configuration")
    print("="*80)
    print("\nEnvironment:")
    for key, value in config['env'].items():
        print(f"  {key}: {value}")
    print("\nTraining:")
    for key, value in config['training'].items():
        print(f"  {key}: {value}")
    print("="*80 + "\n")

# =============================================================================
# 
# =============================================================================

if __name__ == '__main__':
    # 
    scenarios = [
        'basic', 'high_fdi', 'conservative', 'aggressive',
        'large', 'debug', 'scheduled_fdi', 'a2c', 'dqn'
    ]

    for scenario in scenarios:
        print(f"\n{'#'*80}")
        print(f"# Scenario: {scenario.upper()}")
        print(f"{'#'*80}")
        config = get_config(scenario)
        print_config(config)
