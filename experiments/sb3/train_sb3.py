"""
使用 Stable-Baselines3 训练变电站变压器控制策略

支持多种算法：
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- DQN (Deep Q-Network) - 需要离散动作空间
- SAC (Soft Actor-Critic) - 需要连续动作空间

作者：重构版本
日期：2025-10-31
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Stable-Baselines3
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# 自定义环境
from src.envs.gym_substation_env import make_substation_env


class TensorboardCallback(BaseCallback):
    """
    自定义回调，用于记录额外的指标到 Tensorboard
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # 获取环境信息
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])

                # 记录统计信息
                if 'stats' in info:
                    stats = info['stats']
                    self.logger.record('stats/total_disconnections', stats['total_disconnections'])
                    self.logger.record('stats/correct_disconnections', stats['correct_disconnections'])
                    self.logger.record('stats/false_disconnections', stats['false_disconnections'])
                    self.logger.record('stats/missed_disconnections', stats['missed_disconnections'])

                    # 计算准确率
                    total_decisions = (stats['correct_disconnections'] +
                                       stats['false_disconnections'] +
                                       stats['missed_disconnections'])
                    if total_decisions > 0:
                        accuracy = stats['correct_disconnections'] / total_decisions
                        self.logger.record('stats/disconnect_accuracy', accuracy)

                    # FDI 检测率
                    if stats['total_fdi_attacks'] > 0:
                        fdi_detection_rate = stats['fdi_attacks_detected'] / stats['total_fdi_attacks']
                        self.logger.record('stats/fdi_detection_rate', fdi_detection_rate)

        return True


def make_env(env_kwargs, rank=0, seed=0):
    """
    创建环境的工厂函数（用于并行环境）
    """
    def _init():
        env = make_substation_env(**env_kwargs)
        env.seed(seed + rank)
        env = Monitor(env)
        return env
    return _init


def train(
    algorithm='ppo',
    network_case='case14',
    total_timesteps=100000,
    n_envs=4,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    max_steps_per_episode=200,
    fdi_attack_prob=0.1,
    max_temperature=90.0,
    save_dir='./models_sb3',
    log_dir='./logs_sb3',
    eval_freq=10000,
    seed=42,
):
    """
    训练函数

    Args:
        algorithm: 算法名称 ('ppo', 'a2c', 'dqn')
        network_case: 电网案例
        total_timesteps: 总训练步数
        n_envs: 并行环境数量
        learning_rate: 学习率
        n_steps: PPO/A2C 的步数
        batch_size: 批次大小
        n_epochs: PPO 的 epoch 数
        gamma: 折扣因子
        max_steps_per_episode: 每个 episode 的最大步数
        fdi_attack_prob: FDI 攻击概率
        max_temperature: 最大安全温度
        save_dir: 模型保存目录
        log_dir: 日志目录
        eval_freq: 评估频率
        seed: 随机种子
    """

    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"{algorithm}_{network_case}_{timestamp}")
    log_path = os.path.join(log_dir, f"{algorithm}_{network_case}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Network Case: {network_case}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Parallel Environments: {n_envs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Max Steps per Episode: {max_steps_per_episode}")
    print(f"FDI Attack Probability: {fdi_attack_prob}")
    print(f"Max Temperature Threshold: {max_temperature}°C")
    print(f"Save Path: {save_path}")
    print(f"Log Path: {log_path}")
    print(f"{'='*80}\n")

    # 环境配置
    env_kwargs = {
        'network_case': network_case,
        'max_steps': max_steps_per_episode,
        'fdi_attack_prob': fdi_attack_prob,
        'max_temperature': max_temperature,
    }

    # 创建训练环境
    if n_envs > 1:
        env = SubprocVecEnv([make_env(env_kwargs, rank=i, seed=seed) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(env_kwargs, seed=seed)])

    # 创建评估环境
    eval_env = DummyVecEnv([make_env(env_kwargs, seed=seed + 1000)])

    # 检查环境（仅对单个环境）
    print("Checking environment...")
    test_env = make_substation_env(**env_kwargs)
    try:
        check_env(test_env, warn=True)
        print("✓ Environment check passed!\n")
    except Exception as e:
        print(f"✗ Environment check failed: {e}\n")
    finally:
        test_env.close()

    # 配置日志
    logger = configure(log_path, ["stdout", "tensorboard"])

    # 创建回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=eval_freq // n_envs,  # 除以环境数量
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )

    tensorboard_callback = TensorboardCallback(verbose=1)

    callbacks = CallbackList([eval_callback, tensorboard_callback])

    # 创建模型
    print(f"Creating {algorithm.upper()} model...")

    if algorithm.lower() == 'ppo':
        model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            verbose=1,
            tensorboard_log=log_path,
            seed=seed,
        )
    elif algorithm.lower() == 'a2c':
        model = A2C(
            policy='MlpPolicy',
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            verbose=1,
            tensorboard_log=log_path,
            seed=seed,
        )
    elif algorithm.lower() == 'dqn':
        model = DQN(
            policy='MlpPolicy',
            env=env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            verbose=1,
            tensorboard_log=log_path,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    model.set_logger(logger)

    print(f"Starting training for {total_timesteps:,} timesteps...\n")

    # 训练
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    # 保存最终模型
    final_model_path = os.path.join(save_path, f"{algorithm}_final")
    model.save(final_model_path)
    print(f"\n✓ Training completed! Final model saved to: {final_model_path}")

    # 清理
    env.close()
    eval_env.close()

    return model, save_path


def evaluate(
    model_path,
    network_case='case14',
    n_eval_episodes=10,
    max_steps_per_episode=200,
    fdi_attack_prob=0.1,
    max_temperature=90.0,
    render=False,
    seed=999,
):
    """
    评估训练好的模型

    Args:
        model_path: 模型路径（不含扩展名）
        network_case: 电网案例
        n_eval_episodes: 评估 episode 数量
        max_steps_per_episode: 每个 episode 的最大步数
        fdi_attack_prob: FDI 攻击概率
        max_temperature: 最大安全温度
        render: 是否渲染
        seed: 随机种子
    """

    print(f"\n{'='*80}")
    print(f"Evaluation")
    print(f"{'='*80}")
    print(f"Model Path: {model_path}")
    print(f"Network Case: {network_case}")
    print(f"Eval Episodes: {n_eval_episodes}")
    print(f"{'='*80}\n")

    # 创建环境
    env_kwargs = {
        'network_case': network_case,
        'max_steps': max_steps_per_episode,
        'fdi_attack_prob': fdi_attack_prob,
        'max_temperature': max_temperature,
        'render_mode': 'human' if render else None,
    }
    env = make_substation_env(**env_kwargs)
    env.seed(seed)

    # 加载模型
    # 自动检测算法类型
    if 'ppo' in model_path.lower():
        model = PPO.load(model_path)
    elif 'a2c' in model_path.lower():
        model = A2C.load(model_path)
    elif 'dqn' in model_path.lower():
        model = DQN.load(model_path)
    else:
        # 默认尝试 PPO
        print("Warning: Cannot detect algorithm from path, trying PPO...")
        model = PPO.load(model_path)

    # 评估
    episode_rewards = []
    episode_lengths = []
    all_stats = []

    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        steps = 0

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        all_stats.append(info['stats'])

        print(f"Episode {episode + 1}/{n_eval_episodes}: "
              f"Reward = {episode_reward:.2f}, Length = {steps}")

    # 统计结果
    print(f"\n{'='*80}")
    print(f"Evaluation Results")
    print(f"{'='*80}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")

    # 聚合统计信息
    aggregated_stats = {key: [] for key in all_stats[0].keys()}
    for stats in all_stats:
        for key, value in stats.items():
            aggregated_stats[key].append(value)

    print(f"\nAggregated Statistics:")
    for key, values in aggregated_stats.items():
        print(f"  {key}: {np.mean(values):.2f} ± {np.std(values):.2f}")

    # 计算准确率
    total_correct = sum(s['correct_disconnections'] for s in all_stats)
    total_false = sum(s['false_disconnections'] for s in all_stats)
    total_missed = sum(s['missed_disconnections'] for s in all_stats)
    total_decisions = total_correct + total_false + total_missed

    if total_decisions > 0:
        accuracy = total_correct / total_decisions
        print(f"\nDisconnect Decision Accuracy: {accuracy:.2%}")

    # FDI 检测率
    total_fdi_attacks = sum(s['total_fdi_attacks'] for s in all_stats)
    total_fdi_detected = sum(s['fdi_attacks_detected'] for s in all_stats)

    if total_fdi_attacks > 0:
        fdi_detection_rate = total_fdi_detected / total_fdi_attacks
        print(f"FDI Detection Rate: {fdi_detection_rate:.2%}")

    print(f"{'='*80}\n")

    env.close()

    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'stats': aggregated_stats,
    }


def main():
    parser = argparse.ArgumentParser(description='Train/Evaluate Substation Transformer Control with SB3')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Mode: train or eval')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'a2c', 'dqn'],
                        help='RL algorithm')
    parser.add_argument('--network', type=str, default='case14', choices=['case14', 'case30'],
                        help='Network case')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--n_envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='Max steps per episode')
    parser.add_argument('--fdi_prob', type=float, default=0.1,
                        help='FDI attack probability')
    parser.add_argument('--max_temp', type=float, default=90.0,
                        help='Maximum safe temperature')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Model path for evaluation (without extension)')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render during evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    if args.mode == 'train':
        train(
            algorithm=args.algorithm,
            network_case=args.network,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.lr,
            max_steps_per_episode=args.max_steps,
            fdi_attack_prob=args.fdi_prob,
            max_temperature=args.max_temp,
            seed=args.seed,
        )
    elif args.mode == 'eval':
        if args.model_path is None:
            print("Error: --model_path is required for evaluation mode")
            return

        evaluate(
            model_path=args.model_path,
            network_case=args.network,
            n_eval_episodes=args.eval_episodes,
            max_steps_per_episode=args.max_steps,
            fdi_attack_prob=args.fdi_prob,
            max_temperature=args.max_temp,
            render=args.render,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()
