"""
Gym-compatible Multi-Agent Substation Environment for Transformer Control
 Stable-Baselines3 

1. 
2. FDI Attack

2025-10-31
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
     Gymnasium 

    
    
    - 
    - 
    - 
    - FDI 

    
    (0)(1)
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(
        self,
        net: pp.pandapowerNet,
        trafo_indices: List[int],
        max_steps: int = 200,
        # 
        T_ambient: float = 25.0,
        T_rated: float = 65.0,
        n: float = 1.6,
        max_temperature: float = 90.0,  # 
        # FDI 
        fdi_attack_prob: float = 0.1,  #  FDI 
        fdi_temp_range: Tuple[float, float] = (80.0, 120.0),  # FDI 
        fdi_schedule: Optional[List[Tuple[int, int, float]]] = None,  # [(step, trafo_idx, fake_temp), ...]
        # 
        reward_correct_disconnect: float = 10.0,
        reward_stay_connected: float = 2.0,
        penalty_missed_disconnect: float = 15.0,
        penalty_false_disconnect: float = 8.0,
        penalty_voltage_violation: float = 5.0,
        penalty_line_overload: float = 3.0,
        reward_fdi_detection: float = 5.0,  #  FDI 
        # 
        voltage_min: float = 0.95,
        voltage_max: float = 1.05,
        line_loading_limit: float = 100.0,
        # 
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # 
        self.net = net
        self.trafo_indices = trafo_indices
        self.num_agents = len(trafo_indices)
        self.max_steps = max_steps
        self.current_step = 0

        # 
        self.T_ambient = T_ambient
        self.T_rated = T_rated
        self.n = n
        self.max_temperature = max_temperature

        # FDI 
        self.fdi_attack_prob = fdi_attack_prob
        self.fdi_temp_range = fdi_temp_range
        self.fdi_schedule = fdi_schedule if fdi_schedule else []
        self.current_fdi_attacks = {}  # {trafo_idx: fake_temp}

        # 
        self.reward_correct_disconnect = reward_correct_disconnect
        self.reward_stay_connected = reward_stay_connected
        self.penalty_missed_disconnect = penalty_missed_disconnect
        self.penalty_false_disconnect = penalty_false_disconnect
        self.penalty_voltage_violation = penalty_voltage_violation
        self.penalty_line_overload = penalty_line_overload
        self.reward_fdi_detection = reward_fdi_detection

        # 
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max
        self.line_loading_limit = line_loading_limit

        # 
        self.temperature_history = {idx: np.zeros(5) for idx in trafo_indices}

        # 
        self.render_mode = render_mode

        # 
        if seed is not None:
            self.seed(seed)

        # 
        self._define_spaces()

        # 
        self._cache_network_topology()

        # 
        self.episode_stats = {
            'total_disconnections': 0,
            'correct_disconnections': 0,
            'false_disconnections': 0,
            'missed_disconnections': 0,
            'fdi_attacks_detected': 0,
            'total_fdi_attacks': 0,
        }

    def _define_spaces(self):
        """"""
        # 
        # [0] transformer_loading_percent (0-200)
        # [1] temperature_measured (0-300)
        # [2] actual_temperature (0-300)
        # [3] temperature_trend (5, 0-300)
        # [4] temperature_jump (-, -100-100)
        # [5] voltage_pu (0.8-1.2)
        # [6] local_line_loading_mean (0-200)
        # [7] local_line_loading_max (0-200)
        # [8] local_load_p_mw (0-100)
        # [9] is_connected (0 or 1)
        # [10] normalized_time (0-1)
        # [11] fdi_indicator (|measured - actual| > threshold, 0 or 1)

        obs_dim_per_agent = 12

        # 
        # 1. 
        # 2.  VecEnv 

        # 
        total_obs_dim = obs_dim_per_agent * self.num_agents

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )

        #  0 ()  1 ()
        #  MultiBinary  MultiDiscrete
        #  MultiDiscrete 2 
        self.action_space = spaces.MultiDiscrete([2] * self.num_agents)

    def _cache_network_topology(self):
        """"""
        self.trafo_lv_buses = {}
        self.trafo_connected_lines = {}
        self.trafo_local_loads = {}

        for trafo_idx in self.trafo_indices:
            lv_bus = self.net.trafo.at[trafo_idx, "lv_bus"]
            self.trafo_lv_buses[trafo_idx] = lv_bus

            # 
            connected_lines = self.net.line[
                (self.net.line.from_bus == lv_bus) | (self.net.line.to_bus == lv_bus)
            ].index.tolist()
            self.trafo_connected_lines[trafo_idx] = connected_lines

            # 
            local_loads = self.net.load[self.net.load.bus == lv_bus].index.tolist()
            self.trafo_local_loads[trafo_idx] = local_loads

    def seed(self, seed: Optional[int] = None):
        """"""
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
        

        Returns:
            observation: 
            info: 
        """
        super().reset(seed=seed)

        if seed is not None:
            self.seed(seed)

        self.current_step = 0
        self.current_fdi_attacks = {}

        # 
        for trafo_idx in self.trafo_indices:
            self.net.trafo.at[trafo_idx, 'in_service'] = True

        # 
        self.temperature_history = {idx: np.zeros(5) for idx in self.trafo_indices}

        # 
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
        """"""
        # 
        if 'vm_pu' not in self.net.res_bus.columns or self.net.res_bus['vm_pu'].isna().any():
            self.net.res_bus['vm_pu'] = 1.0

        if 'loading_percent' not in self.net.res_trafo.columns or self.net.res_trafo['loading_percent'].isna().any():
            self.net.res_trafo['loading_percent'] = 0.0

        if 'loading_percent' not in self.net.res_line.columns or self.net.res_line['loading_percent'].isna().any():
            self.net.res_line['loading_percent'] = 0.0

    def _update_temperatures(self):
        """
        
         IEEE 
        T_actual = T_ambient + T_rated * (loading / 100) ^ n
        """
        for trafo_idx in self.trafo_indices:
            if self.net.trafo.at[trafo_idx, 'in_service']:
                loading = self.net.res_trafo.at[trafo_idx, 'loading_percent']
                loading = np.clip(loading, 0, 200)  # 

                # 
                actual_temp = self.T_ambient + self.T_rated * (loading / 100.0) ** self.n
            else:
                # 
                actual_temp = self.T_ambient

            # 
            self.net.trafo.at[trafo_idx, 'actual_temperature'] = actual_temp

            #  FDI 
            self.net.trafo.at[trafo_idx, 'temperature_measured'] = actual_temp

    def _inject_fdi_attacks(self):
        """FDI"""
        self.current_fdi_attacks = {}

        #  FDI 
        for step, trafo_idx, fake_temp in self.fdi_schedule:
            if step == self.current_step and trafo_idx in self.trafo_indices:
                self.net.trafo.at[trafo_idx, 'temperature_measured'] = fake_temp
                self.current_fdi_attacks[trafo_idx] = fake_temp
                self.episode_stats['total_fdi_attacks'] += 1

        #  FDI 
        if self.np_random.random() < self.fdi_attack_prob:
            # 
            trafo_idx = self.np_random.choice(self.trafo_indices)
            if trafo_idx not in self.current_fdi_attacks:
                # 
                fake_temp = self.np_random.uniform(*self.fdi_temp_range)
                self.net.trafo.at[trafo_idx, 'temperature_measured'] = fake_temp
                self.current_fdi_attacks[trafo_idx] = fake_temp
                self.episode_stats['total_fdi_attacks'] += 1

    def _get_observation(self) -> np.ndarray:
        """"""
        observations = []

        for trafo_idx in self.trafo_indices:
            obs = self._get_local_observation(trafo_idx)
            observations.append(obs)

        # 
        return np.concatenate(observations).astype(np.float32)

    def _get_local_observation(self, trafo_idx: int) -> np.ndarray:
        """"""
        try:
            # [0] transformer_loading_percent
            loading = float(self.net.res_trafo.at[trafo_idx, 'loading_percent'])
            loading = np.nan_to_num(loading, nan=0.0)

            # [1] temperature_measured ( FDI)
            temp_measured = float(self.net.trafo.at[trafo_idx, 'temperature_measured'])

            # [2] actual_temperature
            temp_actual = float(self.net.trafo.at[trafo_idx, 'actual_temperature'])

            # 
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
        

        
        1. 
        2. 
        3. 
        4. 
        5. 
        6. 
        7.  FDI 
        """
        total_reward = 0.0

        for i, trafo_idx in enumerate(self.trafo_indices):
            # 
            temp_actual = self.net.trafo.at[trafo_idx, 'actual_temperature']
            temp_measured = self.net.trafo.at[trafo_idx, 'temperature_measured']
            is_connected = self.net.trafo.at[trafo_idx, 'in_service']
            action_value = int(action[i])

            # 
            is_overheat = temp_actual > self.max_temperature

            #  FDI 
            fdi_delta = abs(temp_measured - temp_actual)
            has_fdi = fdi_delta > 5.0

            # 
            if is_overheat:
                if action_value == 0:  # 
                    # 
                    total_reward += self.reward_correct_disconnect
                    self.episode_stats['correct_disconnections'] += 1
                else:  # 
                    # 
                    total_reward -= self.penalty_missed_disconnect
                    self.episode_stats['missed_disconnections'] += 1
            else:  # 
                if action_value == 0:  # 
                    # 
                    total_reward -= self.penalty_false_disconnect
                    self.episode_stats['false_disconnections'] += 1

                    #  FDI 
                    if has_fdi:
                        total_reward += self.penalty_false_disconnect * 0.3
                else:  # 
                    # 
                    total_reward += self.reward_stay_connected

            # FDI 
            if has_fdi:
                #  FDI
                #  FDI
                if (is_overheat and action_value == 0) or (not is_overheat and action_value == 1):
                    total_reward += self.reward_fdi_detection
                    self.episode_stats['fdi_attacks_detected'] += 1

            # 
            lv_bus = self.trafo_lv_buses[trafo_idx]
            voltage = self.net.res_bus.at[lv_bus, 'vm_pu']
            if voltage < self.voltage_min:
                total_reward -= self.penalty_voltage_violation * (self.voltage_min - voltage)
            elif voltage > self.voltage_max:
                total_reward -= self.penalty_voltage_violation * (voltage - self.voltage_max)

            # 
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
        """"""
        # 
        # - 
        # - 
        # - 

        # 
        all_disconnected = all(
            not self.net.trafo.at[idx, 'in_service']
            for idx in self.trafo_indices
        )

        if all_disconnected:
            return True

        # 
        voltages = self.net.res_bus['vm_pu'].values
        if np.any(voltages < 0.85) or np.any(voltages > 1.15):
            return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """"""
        info = {
            'step': self.current_step,
            'transformers_connected': sum(
                1 for idx in self.trafo_indices
                if self.net.trafo.at[idx, 'in_service']
            ),
            'current_fdi_attacks': len(self.current_fdi_attacks),
            'stats': self.episode_stats.copy(),
        }

        # 
        for trafo_idx in self.trafo_indices:
            info[f'trafo_{trafo_idx}_loading'] = self.net.res_trafo.at[trafo_idx, 'loading_percent']
            info[f'trafo_{trafo_idx}_temp_actual'] = self.net.trafo.at[trafo_idx, 'actual_temperature']
            info[f'trafo_{trafo_idx}_temp_measured'] = self.net.trafo.at[trafo_idx, 'temperature_measured']
            info[f'trafo_{trafo_idx}_connected'] = self.net.trafo.at[trafo_idx, 'in_service']

        return info

    def render(self):
        """"""
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
        """"""
        pass

# ============================================================================
#  Stable-Baselines3 
# ============================================================================

def make_substation_env(
    network_case: str = 'case14',
    max_steps: int = 200,
    fdi_attack_prob: float = 0.1,
    **kwargs
) -> GymSubstationEnv:
    """
    

    Args:
        network_case:  ('case14', 'case30', etc.)
        max_steps: 
        fdi_attack_prob: FDI 
        **kwargs: 

    Returns:
        GymSubstationEnv 
    """
    import pandapower.networks as pn

    # 
    if network_case == 'case14':
        net = pn.case14()
    elif network_case == 'case30':
        net = pn.case30()
    else:
        raise ValueError(f"Unknown network case: {network_case}")

    # 
    trafo_indices = net.trafo.index.tolist()

    # 
    env = GymSubstationEnv(
        net=net,
        trafo_indices=trafo_indices,
        max_steps=max_steps,
        fdi_attack_prob=fdi_attack_prob,
        **kwargs
    )

    return env
