"""
Gym-compatible Multi-Agent Substation Environment for Transformer Control
支持 Stable-Baselines3 的标准强化学习环境

目标：
1. 学习何时断开变压器以避免过热过载
2. 能够检测并绕过温度测量的虚假数据注入攻击（FDI Attack）

作者：重构自原始实现
日期：2025-10-31
"""

import numpy as np
import pandas as pd
import pandapower as pp
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import random


class GymSubstationEnv(gym.Env):
    """
    符合 Gymnasium 标准的变电站环境

    每个变压器是一个智能体，可以决定是否断开连接
    环境会模拟电力系统的运行，包括：
    - 电压稳定性
    - 线路和变压器加载
    - 温度升高（基于负载）
    - FDI 攻击（虚假温度读数）

    状态空间：每个智能体观察局部信息
    动作空间：每个智能体选择断开(0)或保持连接(1)
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(
        self,
        net: pp.pandapowerNet,
        trafo_indices: List[int],
        max_steps: int = 200,
        # 温度模型参数
        T_ambient: float = 25.0,
        T_rated: float = 65.0,
        n: float = 1.6,
        max_temperature: float = 90.0,  # 安全阈值
        # FDI 攻击参数
        fdi_attack_prob: float = 0.1,  # 每步发生 FDI 攻击的概率
        fdi_temp_range: Tuple[float, float] = (80.0, 120.0),  # FDI 虚假温度范围
        fdi_schedule: Optional[List[Tuple[int, int, float]]] = None,  # [(step, trafo_idx, fake_temp), ...]
        # 奖励函数参数
        reward_correct_disconnect: float = 10.0,
        reward_stay_connected: float = 2.0,
        penalty_missed_disconnect: float = 15.0,
        penalty_false_disconnect: float = 8.0,
        penalty_voltage_violation: float = 5.0,
        penalty_line_overload: float = 3.0,
        reward_fdi_detection: float = 5.0,  # 正确识别 FDI 并做出决策的奖励
        # 系统约束
        voltage_min: float = 0.95,
        voltage_max: float = 1.05,
        line_loading_limit: float = 100.0,
        # 其他
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # 核心组件
        self.net = net
        self.trafo_indices = trafo_indices
        self.num_agents = len(trafo_indices)
        self.max_steps = max_steps
        self.current_step = 0

        # 温度模型参数
        self.T_ambient = T_ambient
        self.T_rated = T_rated
        self.n = n
        self.max_temperature = max_temperature

        # FDI 攻击参数
        self.fdi_attack_prob = fdi_attack_prob
        self.fdi_temp_range = fdi_temp_range
        self.fdi_schedule = fdi_schedule if fdi_schedule else []
        self.current_fdi_attacks = {}  # {trafo_idx: fake_temp}

        # 奖励参数
        self.reward_correct_disconnect = reward_correct_disconnect
        self.reward_stay_connected = reward_stay_connected
        self.penalty_missed_disconnect = penalty_missed_disconnect
        self.penalty_false_disconnect = penalty_false_disconnect
        self.penalty_voltage_violation = penalty_voltage_violation
        self.penalty_line_overload = penalty_line_overload
        self.reward_fdi_detection = reward_fdi_detection

        # 系统约束
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max
        self.line_loading_limit = line_loading_limit

        # 历史数据（用于趋势分析）
        self.temperature_history = {idx: np.zeros(5) for idx in trafo_indices}

        # 渲染模式
        self.render_mode = render_mode

        # 设置随机种子
        if seed is not None:
            self.seed(seed)

        # 定义观察空间和动作空间
        self._define_spaces()

        # 初始化网络拓扑缓存
        self._cache_network_topology()

        # 统计信息
        self.episode_stats = {
            'total_disconnections': 0,
            'correct_disconnections': 0,
            'false_disconnections': 0,
            'missed_disconnections': 0,
            'fdi_attacks_detected': 0,
            'total_fdi_attacks': 0,
        }

    def _define_spaces(self):
        """定义观察空间和动作空间"""
        # 每个智能体的观察维度：
        # [0] transformer_loading_percent (0-200)
        # [1] temperature_measured (0-300)
        # [2] actual_temperature (0-300)
        # [3] temperature_trend (过去5步平均, 0-300)
        # [4] temperature_jump (当前-趋势, -100-100)
        # [5] voltage_pu (0.8-1.2)
        # [6] local_line_loading_mean (0-200)
        # [7] local_line_loading_max (0-200)
        # [8] local_load_p_mw (0-100)
        # [9] is_connected (0 or 1)
        # [10] normalized_time (0-1)
        # [11] fdi_indicator (|measured - actual| > threshold, 0 or 1)

        obs_dim_per_agent = 12

        # 对于多智能体，我们可以有两种选择：
        # 1. 联合观察空间（所有智能体的观察拼接在一起）
        # 2. 单个智能体的观察空间（如果使用像 VecEnv 的包装器）

        # 这里我们使用联合观察空间，适合单一策略控制所有智能体
        total_obs_dim = obs_dim_per_agent * self.num_agents

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )

        # 动作空间：每个智能体选择 0 (断开) 或 1 (保持连接)
        # 使用 MultiBinary 或 MultiDiscrete
        # 这里使用 MultiDiscrete，每个智能体有 2 个离散动作
        self.action_space = spaces.MultiDiscrete([2] * self.num_agents)

    def _cache_network_topology(self):
        """缓存网络拓扑信息以提高性能"""
        self.trafo_lv_buses = {}
        self.trafo_connected_lines = {}
        self.trafo_local_loads = {}

        for trafo_idx in self.trafo_indices:
            lv_bus = self.net.trafo.at[trafo_idx, "lv_bus"]
            self.trafo_lv_buses[trafo_idx] = lv_bus

            # 连接到该变压器低压侧的线路
            connected_lines = self.net.line[
                (self.net.line.from_bus == lv_bus) | (self.net.line.to_bus == lv_bus)
            ].index.tolist()
            self.trafo_connected_lines[trafo_idx] = connected_lines

            # 连接到该变压器低压侧的负载
            local_loads = self.net.load[self.net.load.bus == lv_bus].index.tolist()
            self.trafo_local_loads[trafo_idx] = local_loads

    def seed(self, seed: Optional[int] = None):
        """设置随机种子"""
        self.np_random = np.random.RandomState(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境到初始状态

        Returns:
            observation: 初始观察
            info: 额外信息字典
        """
        super().reset(seed=seed)

        if seed is not None:
            self.seed(seed)

        self.current_step = 0
        self.current_fdi_attacks = {}

        # 重置所有变压器为连接状态
        for trafo_idx in self.trafo_indices:
            self.net.trafo.at[trafo_idx, 'in_service'] = True

        # 重置温度历史
        self.temperature_history = {idx: np.zeros(5) for idx in self.trafo_indices}

        # 重置统计信息
        self.episode_stats = {
            'total_disconnections': 0,
            'correct_disconnections': 0,
            'false_disconnections': 0,
            'missed_disconnections': 0,
            'fdi_attacks_detected': 0,
            'total_fdi_attacks': 0,
        }

        try:
            pp.runpp(self.net, algorithm='nr', init='flat')
        except pp.LoadflowNotConverged:
            print("Warning: Initial power flow did not converge, using flat start")
            self._fill_default_results()

        self._update_temperatures()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        for i, trafo_idx in enumerate(self.trafo_indices):
            action_value = int(action[i])
            if action_value == 0:
                self.net.trafo.at[trafo_idx, 'in_service'] = False
                self.episode_stats['total_disconnections'] += 1
            else:
                self.net.trafo.at[trafo_idx, 'in_service'] = True

        converged = self._run_power_flow()

        self._update_temperatures()

        self._inject_fdi_attacks()

        reward = self._calculate_reward(action)

        self.current_step += 1

        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()

        if not converged:
            reward -= 100.0
            terminated = True
            info['convergence_failure'] = True

        return observation, reward, terminated, truncated, info

    def _run_power_flow(self) -> bool:
        try:
            pp.runpp(self.net, algorithm='nr', init='results')
            return True
        except pp.LoadflowNotConverged:
            print(f"Warning: Power flow did not converge at step {self.current_step}")
            self._fill_default_results()
            return False

    def _fill_default_results(self):
        """当电力流不收敛时，填充默认结果"""
        # 这是一个简化的处理，实际应用中可能需要更复杂的逻辑
        if 'vm_pu' not in self.net.res_bus.columns or self.net.res_bus['vm_pu'].isna().any():
            self.net.res_bus['vm_pu'] = 1.0

        if 'loading_percent' not in self.net.res_trafo.columns or self.net.res_trafo['loading_percent'].isna().any():
            self.net.res_trafo['loading_percent'] = 0.0

        if 'loading_percent' not in self.net.res_line.columns or self.net.res_line['loading_percent'].isna().any():
            self.net.res_line['loading_percent'] = 0.0

    def _update_temperatures(self):
        """
        根据变压器负载计算实际温度
        使用 IEEE 标准温度模型：
        T_actual = T_ambient + T_rated * (loading / 100) ^ n
        """
        for trafo_idx in self.trafo_indices:
            if self.net.trafo.at[trafo_idx, 'in_service']:
                loading = self.net.res_trafo.at[trafo_idx, 'loading_percent']
                loading = np.clip(loading, 0, 200)  # 限制在合理范围内

                # 计算实际温度
                actual_temp = self.T_ambient + self.T_rated * (loading / 100.0) ** self.n
            else:
                # 断开的变压器温度等于环境温度
                actual_temp = self.T_ambient

            # 存储实际温度
            self.net.trafo.at[trafo_idx, 'actual_temperature'] = actual_temp

            # 初始化测量温度为实际温度（稍后可能被 FDI 攻击修改）
            self.net.trafo.at[trafo_idx, 'temperature_measured'] = actual_temp

    def _inject_fdi_attacks(self):
        """注入虚假数据注入攻击（FDI）"""
        self.current_fdi_attacks = {}

        # 检查预定的 FDI 攻击
        for step, trafo_idx, fake_temp in self.fdi_schedule:
            if step == self.current_step and trafo_idx in self.trafo_indices:
                self.net.trafo.at[trafo_idx, 'temperature_measured'] = fake_temp
                self.current_fdi_attacks[trafo_idx] = fake_temp
                self.episode_stats['total_fdi_attacks'] += 1

        # 随机 FDI 攻击
        if self.np_random.random() < self.fdi_attack_prob:
            # 随机选择一个变压器进行攻击
            trafo_idx = self.np_random.choice(self.trafo_indices)
            if trafo_idx not in self.current_fdi_attacks:
                # 生成虚假温度
                fake_temp = self.np_random.uniform(*self.fdi_temp_range)
                self.net.trafo.at[trafo_idx, 'temperature_measured'] = fake_temp
                self.current_fdi_attacks[trafo_idx] = fake_temp
                self.episode_stats['total_fdi_attacks'] += 1

    def _get_observation(self) -> np.ndarray:
        """获取当前观察"""
        observations = []

        for trafo_idx in self.trafo_indices:
            obs = self._get_local_observation(trafo_idx)
            observations.append(obs)

        # 拼接所有智能体的观察
        return np.concatenate(observations).astype(np.float32)

    def _get_local_observation(self, trafo_idx: int) -> np.ndarray:
        """获取单个变压器的局部观察"""
        try:
            # [0] transformer_loading_percent
            loading = float(self.net.res_trafo.at[trafo_idx, 'loading_percent'])
            loading = np.nan_to_num(loading, nan=0.0)

            # [1] temperature_measured (可能包含 FDI)
            temp_measured = float(self.net.trafo.at[trafo_idx, 'temperature_measured'])

            # [2] actual_temperature
            temp_actual = float(self.net.trafo.at[trafo_idx, 'actual_temperature'])

            # 更新温度历史
            self.temperature_history[trafo_idx] = np.roll(self.temperature_history[trafo_idx], -1)
            self.temperature_history[trafo_idx][-1] = temp_measured

            # [3] temperature_trend
            temp_trend = float(np.mean(self.temperature_history[trafo_idx]))

            # [4] temperature_jump
            temp_jump = temp_measured - temp_trend

            # [5] voltage_pu
            lv_bus = self.trafo_lv_buses[trafo_idx]
            voltage = float(self.net.res_bus.at[lv_bus, 'vm_pu'])
            voltage = np.nan_to_num(voltage, nan=1.0)

            # [6] local_line_loading_mean, [7] local_line_loading_max
            connected_lines = self.trafo_connected_lines[trafo_idx]
            if connected_lines:
                line_loadings = self.net.res_line.loc[connected_lines, 'loading_percent'].values
                line_loadings = np.nan_to_num(line_loadings, nan=0.0)
                line_mean = float(np.mean(line_loadings))
                line_max = float(np.max(line_loadings))
            else:
                line_mean = 0.0
                line_max = 0.0

            # [8] local_load_p_mw
            local_loads = self.trafo_local_loads[trafo_idx]
            if local_loads:
                total_load = float(self.net.load.loc[local_loads, 'p_mw'].sum())
            else:
                total_load = 0.0

            # [9] is_connected
            is_connected = float(self.net.trafo.at[trafo_idx, 'in_service'])

            # [10] normalized_time
            normalized_time = self.current_step / self.max_steps

            # [11] fdi_indicator
            fdi_delta = abs(temp_measured - temp_actual)
            fdi_indicator = float(fdi_delta > 5.0)

            obs = np.array([
                loading,
                temp_measured,
                temp_actual,
                temp_trend,
                temp_jump,
                voltage,
                line_mean,
                line_max,
                total_load,
                is_connected,
                normalized_time,
                fdi_indicator
            ], dtype=np.float32)

            return obs

        except Exception as e:
            print(f"Error getting observation for trafo {trafo_idx}: {e}")
            return np.zeros(12, dtype=np.float32)

    def _calculate_reward(self, action: np.ndarray) -> float:
        """
        计算奖励

        奖励设计目标：
        1. 鼓励在过热时断开变压器
        2. 鼓励在安全时保持连接
        3. 惩罚误断（在安全时断开）
        4. 惩罚漏断（在过热时不断开）
        5. 惩罚电压违规
        6. 惩罚线路过载
        7. 奖励正确识别并处理 FDI 攻击
        """
        total_reward = 0.0

        for i, trafo_idx in enumerate(self.trafo_indices):
            # 获取状态
            temp_actual = self.net.trafo.at[trafo_idx, 'actual_temperature']
            temp_measured = self.net.trafo.at[trafo_idx, 'temperature_measured']
            is_connected = self.net.trafo.at[trafo_idx, 'in_service']
            action_value = int(action[i])

            # 判断是否过热
            is_overheat = temp_actual > self.max_temperature

            # 判断是否有 FDI 攻击
            fdi_delta = abs(temp_measured - temp_actual)
            has_fdi = fdi_delta > 5.0

            # 基于动作和状态的奖励
            if is_overheat:
                if action_value == 0:  # 断开
                    # 正确断开过热变压器
                    total_reward += self.reward_correct_disconnect
                    self.episode_stats['correct_disconnections'] += 1
                else:  # 保持连接
                    # 漏断：应该断开但没断开
                    total_reward -= self.penalty_missed_disconnect
                    self.episode_stats['missed_disconnections'] += 1
            else:  # 不过热
                if action_value == 0:  # 断开
                    # 误断：不应该断开但断开了
                    total_reward -= self.penalty_false_disconnect
                    self.episode_stats['false_disconnections'] += 1

                    # 如果有 FDI 攻击导致的误断，减少惩罚（因为被误导）
                    if has_fdi:
                        total_reward += self.penalty_false_disconnect * 0.3
                else:  # 保持连接
                    # 正确保持连接
                    total_reward += self.reward_stay_connected

            # FDI 检测奖励
            if has_fdi:
                # 如果智能体能够根据实际情况做出正确决策（即使有 FDI）
                # 这意味着它检测到了 FDI
                if (is_overheat and action_value == 0) or (not is_overheat and action_value == 1):
                    total_reward += self.reward_fdi_detection
                    self.episode_stats['fdi_attacks_detected'] += 1

            # 电压约束
            lv_bus = self.trafo_lv_buses[trafo_idx]
            voltage = self.net.res_bus.at[lv_bus, 'vm_pu']
            if voltage < self.voltage_min:
                total_reward -= self.penalty_voltage_violation * (self.voltage_min - voltage)
            elif voltage > self.voltage_max:
                total_reward -= self.penalty_voltage_violation * (voltage - self.voltage_max)

            # 线路过载惩罚
            if is_connected:
                connected_lines = self.trafo_connected_lines[trafo_idx]
                if connected_lines:
                    line_loadings = self.net.res_line.loc[connected_lines, 'loading_percent'].values
                    line_loadings = np.nan_to_num(line_loadings, nan=0.0)
                    for loading in line_loadings:
                        if loading > self.line_loading_limit:
                            total_reward -= self.penalty_line_overload * (loading - self.line_loading_limit) / 100.0

        return total_reward

    def _check_termination(self) -> bool:
        """检查是否应该提前终止"""
        # 可以添加终止条件，例如：
        # - 所有变压器都断开
        # - 电压崩溃
        # - 线路严重过载

        # 检查是否所有变压器都断开
        all_disconnected = all(
            not self.net.trafo.at[idx, 'in_service']
            for idx in self.trafo_indices
        )

        if all_disconnected:
            return True

        # 检查是否有严重的电压违规
        voltages = self.net.res_bus['vm_pu'].values
        if np.any(voltages < 0.85) or np.any(voltages > 1.15):
            return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        info = {
            'step': self.current_step,
            'transformers_connected': sum(
                1 for idx in self.trafo_indices
                if self.net.trafo.at[idx, 'in_service']
            ),
            'current_fdi_attacks': len(self.current_fdi_attacks),
            'stats': self.episode_stats.copy(),
        }

        # 添加每个变压器的详细信息
        for trafo_idx in self.trafo_indices:
            info[f'trafo_{trafo_idx}_loading'] = self.net.res_trafo.at[trafo_idx, 'loading_percent']
            info[f'trafo_{trafo_idx}_temp_actual'] = self.net.trafo.at[trafo_idx, 'actual_temperature']
            info[f'trafo_{trafo_idx}_temp_measured'] = self.net.trafo.at[trafo_idx, 'temperature_measured']
            info[f'trafo_{trafo_idx}_connected'] = self.net.trafo.at[trafo_idx, 'in_service']

        return info

    def render(self):
        """渲染环境（可选）"""
        if self.render_mode == 'human':
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"{'='*60}")

            for trafo_idx in self.trafo_indices:
                loading = self.net.res_trafo.at[trafo_idx, 'loading_percent']
                temp_actual = self.net.trafo.at[trafo_idx, 'actual_temperature']
                temp_measured = self.net.trafo.at[trafo_idx, 'temperature_measured']
                is_connected = self.net.trafo.at[trafo_idx, 'in_service']

                status = "CONNECTED" if is_connected else "DISCONNECTED"
                fdi_marker = " [FDI!]" if trafo_idx in self.current_fdi_attacks else ""

                print(f"Trafo {trafo_idx}: {status}")
                print(f"  Loading: {loading:.2f}%")
                print(f"  Temp (Actual): {temp_actual:.2f}°C")
                print(f"  Temp (Measured): {temp_measured:.2f}°C{fdi_marker}")
                print()

            print(f"Episode Stats:")
            for key, value in self.episode_stats.items():
                print(f"  {key}: {value}")
            print(f"{'='*60}\n")

    def close(self):
        """清理资源"""
        pass


# ============================================================================
# 用于 Stable-Baselines3 的包装器
# ============================================================================

def make_substation_env(
    network_case: str = 'case14',
    max_steps: int = 200,
    fdi_attack_prob: float = 0.1,
    **kwargs
) -> GymSubstationEnv:
    """
    创建变电站环境的工厂函数

    Args:
        network_case: 电网案例名称 ('case14', 'case30', etc.)
        max_steps: 最大步数
        fdi_attack_prob: FDI 攻击概率
        **kwargs: 传递给环境的其他参数

    Returns:
        GymSubstationEnv 实例
    """
    import pandapower.networks as pn

    # 创建电网
    if network_case == 'case14':
        net = pn.case14()
    elif network_case == 'case30':
        net = pn.case30()
    else:
        raise ValueError(f"Unknown network case: {network_case}")

    # 获取所有变压器索引
    trafo_indices = net.trafo.index.tolist()

    # 创建环境
    env = GymSubstationEnv(
        net=net,
        trafo_indices=trafo_indices,
        max_steps=max_steps,
        fdi_attack_prob=fdi_attack_prob,
        **kwargs
    )

    return env
