"""
配置文件示例

这个文件展示了如何配置不同的实验场景
"""

# =============================================================================
# 场景 1: 基础训练 - 低 FDI 攻击率
# =============================================================================
BASIC_CONFIG = {
    'env': {
        'network_case': 'case14',
        'max_steps': 200,
        'fdi_attack_prob': 0.05,  # 5% FDI 攻击概率
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
# 场景 2: 高 FDI 攻击率 - 专注于 FDI 检测
# =============================================================================
HIGH_FDI_CONFIG = {
    'env': {
        'network_case': 'case14',
        'max_steps': 200,
        'fdi_attack_prob': 0.3,  # 30% FDI 攻击概率（很高！）
        'max_temperature': 90.0,
        'fdi_temp_range': (70.0, 150.0),  # 更大的虚假温度范围
        # 增加 FDI 检测奖励
        'reward_fdi_detection': 10.0,
        'reward_correct_disconnect': 12.0,
        'penalty_missed_disconnect': 20.0,
    },
    'training': {
        'algorithm': 'ppo',
        'total_timesteps': 200000,  # 更多训练步数
        'n_envs': 8,
        'learning_rate': 1e-4,  # 较低学习率以适应复杂任务
        'n_steps': 2048,
        'batch_size': 128,
        'gamma': 0.99,
    }
}


# =============================================================================
# 场景 3: 保守策略 - 更严格的过载阈值
# =============================================================================
CONSERVATIVE_CONFIG = {
    'env': {
        'network_case': 'case14',
        'max_steps': 200,
        'fdi_attack_prob': 0.1,
        'max_temperature': 80.0,  # 更低的温度阈值（更保守）
        'fdi_temp_range': (75.0, 110.0),
        # 调整奖励以鼓励保守行为
        'reward_correct_disconnect': 15.0,
        'penalty_missed_disconnect': 25.0,  # 严重惩罚漏断
        'penalty_false_disconnect': 5.0,    # 减少误断惩罚
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
# 场景 4: 激进策略 - 优先保持系统运行
# =============================================================================
AGGRESSIVE_CONFIG = {
    'env': {
        'network_case': 'case14',
        'max_steps': 200,
        'fdi_attack_prob': 0.1,
        'max_temperature': 100.0,  # 更高的温度阈值（允许更高温度）
        'fdi_temp_range': (90.0, 130.0),
        # 调整奖励以鼓励保持连接
        'reward_stay_connected': 5.0,       # 增加保持连接奖励
        'penalty_false_disconnect': 15.0,   # 严重惩罚误断
        'penalty_missed_disconnect': 10.0,  # 减少漏断惩罚
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
# 场景 5: 大规模网络 - Case30
# =============================================================================
LARGE_NETWORK_CONFIG = {
    'env': {
        'network_case': 'case30',  # 更大的网络
        'max_steps': 300,          # 更长的 episode
        'fdi_attack_prob': 0.15,
        'max_temperature': 90.0,
        'fdi_temp_range': (80.0, 120.0),
    },
    'training': {
        'algorithm': 'ppo',
        'total_timesteps': 300000,  # 更多训练步数
        'n_envs': 8,                # 更多并行环境
        'learning_rate': 3e-4,
        'n_steps': 4096,            # 更大的步数
        'batch_size': 128,
        'gamma': 0.99,
    }
}


# =============================================================================
# 场景 6: 快速测试 - 用于调试
# =============================================================================
DEBUG_CONFIG = {
    'env': {
        'network_case': 'case14',
        'max_steps': 50,           # 短 episode
        'fdi_attack_prob': 0.2,
        'max_temperature': 90.0,
        'fdi_temp_range': (85.0, 115.0),
        'render_mode': 'human',    # 启用渲染
    },
    'training': {
        'algorithm': 'ppo',
        'total_timesteps': 10000,  # 少量步数用于快速测试
        'n_envs': 2,
        'learning_rate': 3e-4,
        'n_steps': 512,
        'batch_size': 32,
        'gamma': 0.99,
    }
}


# =============================================================================
# 场景 7: 预定 FDI 攻击 - 用于特定场景测试
# =============================================================================
SCHEDULED_FDI_CONFIG = {
    'env': {
        'network_case': 'case14',
        'max_steps': 100,
        'fdi_attack_prob': 0.0,  # 关闭随机 FDI
        'max_temperature': 90.0,
        # 预定的 FDI 攻击
        'fdi_schedule': [
            (10, 0, 130.0),  # 第 10 步，变压器 0，虚假温度 130°C
            (20, 0, 140.0),  # 第 20 步，变压器 0，虚假温度 140°C
            (30, 1, 125.0),  # 第 30 步，变压器 1，虚假温度 125°C
            (40, 1, 135.0),  # 第 40 步，变压器 1，虚假温度 135°C
            (50, 0, 120.0),  # 第 50 步，变压器 0，虚假温度 120°C
            (60, 1, 115.0),  # 第 60 步，变压器 1，虚假温度 115°C
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
# 场景 8: A2C 算法 - 更快的训练
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
        'n_envs': 16,  # A2C 适合更多并行环境
        'learning_rate': 7e-4,
        'n_steps': 5,  # A2C 通常使用较小的步数
        'gamma': 0.99,
    }
}


# =============================================================================
# 场景 9: DQN 算法 - 离散动作空间
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
        'n_envs': 1,  # DQN 不支持并行环境
        'learning_rate': 1e-4,
        'batch_size': 32,
        'gamma': 0.99,
    }
}


# =============================================================================
# 辅助函数
# =============================================================================

def get_config(scenario_name):
    """
    根据场景名称获取配置

    Args:
        scenario_name: 场景名称

    Returns:
        配置字典
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
    """打印配置"""
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
# 使用示例
# =============================================================================

if __name__ == '__main__':
    # 打印所有配置
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
