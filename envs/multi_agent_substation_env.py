import numpy as np
import pandapower as pp
from utils.network import create_network


class multi_agent_substation_env:
    def __init__(self, net, trafo_indices, delta_p=0.05, initial_p=0.0,
                 voltage_tolerance=0.05, voltage_penalty_factor=10.0,
                 line_loading_limit=100.0, power_flow_penalty_factor=5.0,
                 load_reward_factor=20.0, transformer_reward_factor=20.0,
                 disconnection_penalty_factor=50.0, total_steps=200):
        self.net = net
        self.trafo_indices = trafo_indices
        self.delta_p = delta_p
        self.initial_p = initial_p
        self.voltage_tolerance = voltage_tolerance
        self.voltage_penalty_factor = voltage_penalty_factor
        self.line_loading_limit = line_loading_limit
        self.power_flow_penalty_factor = power_flow_penalty_factor
        self.load_reward_factor = load_reward_factor
        self.transformer_reward_factor = transformer_reward_factor
        self.disconnection_penalty_factor = disconnection_penalty_factor
        self.temperature_history = {index: np.zeros(5) for index in self.trafo_indices}
        self.total_steps = total_steps
        self.step_count = 0
        self.T_ambient = 25.0
        self.T_rated = 65.0
        self.n = 1.6


    def reset(self):
        self.step_count = 0
        net = create_network()
        self.net = net

        self.p = {index: self.initial_p for index in self.trafo_indices}

        try:
            pp.runpp(self.net)
        except pp.LoadflowNotConverged:
            print("Initial power flow did not converge")

        return {index: self.get_local_state(index) for index in self.trafo_indices}

    def get_local_state(self, trafo_index):
        try:
            bus_voltages = self.net.res_bus.vm_pu.values
            power_flows = self.net.res_line.loading_percent.values
            real_transformer_loading = np.nan_to_num(self.net.res_trafo.loading_percent.values, nan=0.0)
            actual_temperature = self.T_ambient + self.T_rated * (real_transformer_loading[trafo_index]) ** self.n
            self.temperature_history[trafo_index] = np.roll(self.temperature_history[trafo_index], shift=-1)
            self.temperature_history[trafo_index][-1] = actual_temperature
            temperature_trend = np.mean(self.temperature_history[trafo_index])

            local_state = np.concatenate([
                bus_voltages,  # num_buses
                power_flows,  # num_lines
                [real_transformer_loading[trafo_index]],
                [actual_temperature],
                [temperature_trend],
                [self.p[trafo_index]]
            ])

            assert len(
                local_state) == self.get_state_size(), f"State size mismatch: expected {self.get_state_size()}, got {len(local_state)}"
            return local_state

        except Exception as e:
            print(f"Error in getting state for Transformer {trafo_index}: {e}")
            return np.zeros(self.get_state_size())

    def get_local_reward(self, state):
        num_buses = len(self.net.res_bus.vm_pu)
        num_lines = len(self.net.res_line.loading_percent)
        bus_voltages = state[:num_buses]
        power_flows = state[num_buses:num_buses + num_lines]
        real_loading = state[-4]  # Transformer loading
        actual_temperature = state[-3]  # Actual temperature
        temperature_trend = state[-2]  # Temperature trend


        voltage_deviation = np.sum(np.maximum(np.abs(bus_voltages - 1.0) - self.voltage_tolerance, 0))
        voltage_penalty = -voltage_deviation * self.voltage_penalty_factor

        overloaded_lines = np.sum(power_flows > self.line_loading_limit)
        line_loading_penalty = -overloaded_lines * self.power_flow_penalty_factor

        temperature_residual = np.abs(actual_temperature - temperature_trend)
        temp_penalty = -50 if temperature_residual > 5 else 10

        if real_loading > 1:
            load_penalty = - (real_loading - 1) * self.transformer_reward_factor
        else:
            load_penalty = (1 - real_loading) * self.transformer_reward_factor

        overtemp_penalty = 50 if actual_temperature >100 else -10

        reward = voltage_penalty + line_loading_penalty + temp_penalty + load_penalty + overtemp_penalty

        return reward

    def get_state_size(self):
        num_buses = len(self.net.res_bus.vm_pu)
        num_lines = len(self.net.res_line.loading_percent)

        transformer_features = 4

        return num_buses + num_lines + transformer_features

    def update_disconnection_probabilities(self, actions):
        for idx, act in enumerate(actions):
            trafo_index = self.trafo_indices[idx]
            if act == 0:
                self.p[trafo_index] = max(self.p[trafo_index] - self.delta_p, 0.0)
            elif act == 1:
                self.p[trafo_index] = min(self.p[trafo_index] + self.delta_p, 1.0)

        # Apply the updated probabilities to decide transformer connection status
        for idx in self.trafo_indices:
            self.net.trafo.at[idx, "in_service"] = np.random.rand() > self.p[idx]





