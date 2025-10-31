"""
快速启动脚本

使用预定义的配置快速开始训练或评估

使用方法：
    python quick_start.py --scenario basic --mode train
    python quick_start.py --scenario high_fdi --mode train
    python quick_start.py --scenario debug --mode eval --model_path ./models_sb3/xxx/ppo_final
"""

import argparse
import sys
from config_example import get_config, print_config
from train_sb3 import train, evaluate


def main():
    parser = argparse.ArgumentParser(description='Quick Start for Substation RL Training')
    parser.add_argument('--scenario', type=str, required=True,
                        choices=[
                            'basic', 'high_fdi', 'conservative', 'aggressive',
                            'large', 'debug', 'scheduled_fdi', 'a2c', 'dqn'
                        ],
                        help='Scenario to run')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Mode: train or eval')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Model path for evaluation (without extension)')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render during evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # 获取配置
    config = get_config(args.scenario)

    # 打印配置
    print(f"\n{'='*80}")
    print(f"Quick Start - Scenario: {args.scenario.upper()}")
    print(f"Mode: {args.mode.upper()}")
    print(f"{'='*80}")
    print_config(config)

    if args.mode == 'train':
        # 训练模式
        print(f"\n🚀 Starting training with '{args.scenario}' configuration...\n")

        train(
            algorithm=config['training']['algorithm'],
            network_case=config['env']['network_case'],
            total_timesteps=config['training']['total_timesteps'],
            n_envs=config['training']['n_envs'],
            learning_rate=config['training']['learning_rate'],
            n_steps=config['training'].get('n_steps', 2048),
            batch_size=config['training']['batch_size'],
            max_steps_per_episode=config['env']['max_steps'],
            fdi_attack_prob=config['env']['fdi_attack_prob'],
            max_temperature=config['env']['max_temperature'],
            seed=args.seed,
        )

        print("\n✅ Training completed!")
        print("\n💡 To view training progress, run:")
        print("   tensorboard --logdir ./logs_sb3")
        print("\n💡 To evaluate the model, run:")
        print(f"   python quick_start.py --scenario {args.scenario} --mode eval "
              f"--model_path <model_path>")

    elif args.mode == 'eval':
        # 评估模式
        if args.model_path is None:
            print("\n❌ Error: --model_path is required for evaluation mode")
            print("\nExample:")
            print("   python quick_start.py --scenario basic --mode eval "
                  "--model_path ./models_sb3/ppo_case14_20250101_120000/ppo_final")
            sys.exit(1)

        print(f"\n🔍 Starting evaluation with '{args.scenario}' configuration...\n")

        evaluate(
            model_path=args.model_path,
            network_case=config['env']['network_case'],
            n_eval_episodes=args.eval_episodes,
            max_steps_per_episode=config['env']['max_steps'],
            fdi_attack_prob=config['env']['fdi_attack_prob'],
            max_temperature=config['env']['max_temperature'],
            render=args.render,
            seed=args.seed,
        )

        print("\n✅ Evaluation completed!")


if __name__ == '__main__':
    main()
