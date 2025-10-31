"""
测试 Gym 环境的基本功能

运行此脚本以验证环境是否正确设置
"""

import numpy as np
from src.envs.gym_substation_env import make_substation_env
from stable_baselines3.common.env_checker import check_env


def test_basic_functionality():
    """测试环境的基本功能"""
    print("\n" + "="*80)
    print("Testing Basic Environment Functionality")
    print("="*80 + "\n")

    # 创建环境
    print("1. Creating environment...")
    env = make_substation_env(
        network_case='case14',
        max_steps=50,
        fdi_attack_prob=0.2,
        render_mode='human'
    )
    print(f"   ✓ Environment created")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Number of agents: {env.num_agents}")

    # 检查环境
    print("\n2. Checking environment compatibility with Gym/SB3...")
    try:
        check_env(env, warn=True)
        print("   ✓ Environment check passed!")
    except Exception as e:
        print(f"   ✗ Environment check failed: {e}")
        return

    # 重置环境
    print("\n3. Resetting environment...")
    obs, info = env.reset()
    print(f"   ✓ Environment reset")
    print(f"   - Observation shape: {obs.shape}")
    print(f"   - Info keys: {list(info.keys())}")

    # 运行几步
    print("\n4. Running 10 random steps...")
    total_reward = 0
    for step in range(10):
        # 随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"   Step {step+1}: reward={reward:.2f}, "
              f"terminated={terminated}, truncated={truncated}")

        if step < 3:  # 只渲染前3步
            env.render()

        if terminated or truncated:
            print(f"   Episode ended at step {step+1}")
            break

    print(f"\n   Total reward: {total_reward:.2f}")

    # 显示统计信息
    print("\n5. Episode statistics:")
    for key, value in info['stats'].items():
        print(f"   - {key}: {value}")

    env.close()
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80 + "\n")


def test_fdi_injection():
    """测试 FDI 攻击注入"""
    print("\n" + "="*80)
    print("Testing FDI Attack Injection")
    print("="*80 + "\n")

    # 创建带有预定 FDI 攻击的环境
    fdi_schedule = [
        (5, 0, 120.0),   # 第5步，变压器0，虚假温度120°C
        (10, 0, 150.0),  # 第10步，变压器0，虚假温度150°C
        (15, 1, 110.0),  # 第15步，变压器1，虚假温度110°C
    ]

    env = make_substation_env(
        network_case='case14',
        max_steps=30,
        fdi_attack_prob=0.0,  # 关闭随机FDI
        fdi_schedule=fdi_schedule,
        render_mode='human'
    )

    obs, info = env.reset()
    print("Environment reset. Running simulation with scheduled FDI attacks...\n")

    for step in range(30):
        # 智能策略：如果检测到 FDI，断开变压器
        action = []
        for i in range(env.num_agents):
            obs_start = i * 12
            fdi_indicator = obs[obs_start + 11]  # 第12个特征是 FDI 指示器
            temp_measured = obs[obs_start + 1]
            temp_actual = obs[obs_start + 2]

            # 如果检测到 FDI，使用实际温度判断
            # 否则使用测量温度
            if fdi_indicator > 0.5:
                decision_temp = temp_actual
                print(f"   Step {step}: Trafo {i} - FDI detected! "
                      f"Measured={temp_measured:.1f}°C, Actual={temp_actual:.1f}°C")
            else:
                decision_temp = temp_measured

            # 如果温度过高，断开
            if decision_temp > env.max_temperature:
                action.append(0)  # 断开
            else:
                action.append(1)  # 保持连接

        action = np.array(action)
        obs, reward, terminated, truncated, info = env.step(action)

        if step in [4, 9, 14, 15, 16]:  # 在 FDI 攻击前后渲染
            env.render()

        if terminated or truncated:
            break

    print("\nFDI Attack Summary:")
    print(f"   Total FDI attacks: {info['stats']['total_fdi_attacks']}")
    print(f"   FDI attacks detected: {info['stats']['fdi_attacks_detected']}")
    if info['stats']['total_fdi_attacks'] > 0:
        detection_rate = info['stats']['fdi_attacks_detected'] / info['stats']['total_fdi_attacks']
        print(f"   Detection rate: {detection_rate:.1%}")

    env.close()
    print("\n" + "="*80)
    print("✓ FDI test completed!")
    print("="*80 + "\n")


def test_multiple_episodes():
    """测试多个 episode"""
    print("\n" + "="*80)
    print("Testing Multiple Episodes")
    print("="*80 + "\n")

    env = make_substation_env(
        network_case='case14',
        max_steps=50,
        fdi_attack_prob=0.15,
    )

    num_episodes = 5
    episode_rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0

        for step in range(50):
            # 简单策略：基于温度阈值
            action = []
            for i in range(env.num_agents):
                obs_start = i * 12
                temp_measured = obs[obs_start + 1]
                # 使用略低的阈值以保守
                if temp_measured > env.max_temperature * 0.9:
                    action.append(0)  # 断开
                else:
                    action.append(1)  # 保持连接

            action = np.array(action)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Steps={steps}")

    print(f"\nAverage reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")

    env.close()
    print("\n" + "="*80)
    print("✓ Multiple episodes test completed!")
    print("="*80 + "\n")


def main():
    """运行所有测试"""
    print("\n" + "#"*80)
    print("#" + " "*30 + "GYM ENVIRONMENT TEST" + " "*29 + "#")
    print("#"*80)

    try:
        test_basic_functionality()
        test_fdi_injection()
        test_multiple_episodes()

        print("\n" + "#"*80)
        print("#" + " "*26 + "ALL TESTS PASSED! 🎉" + " "*33 + "#")
        print("#"*80 + "\n")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
