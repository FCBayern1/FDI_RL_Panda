import numpy as np
import pandapower as pp

class multi_agent_substation_env:
    def __init__(self, net, trafo_indices, delta_p=0.05, initial_p=0.0,
                 voltage_tolerance=0.05, voltage_penalty_factor=10.0,
                 line_loading_limit=20.0, power_flow_penalty_factor=5.0,
                 load_reward_factor=20.0, transformer_reward_factor=20.0,
                 disconnection_penalty_factor=50.0, total_steps=200, max_temperature = 90):
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
        self.max_temperature = max_temperature
        self.p = {index: initial_p for index in trafo_indices}

    def get_local_state(self, trafo_index):
        # if "temperature_measured" not in self.net.trafo.columns:
        #     self.net.trafo["temperature_measured"] = np.nan
        # if "actual_temperature" not in self.net.trafo.columns:
        #     self.net.trafo["actual_temperature"] = np.nan
        # print(f"[Env] net id = {id(self.net)}, temperature_measured = {self.net.trafo['temperature_measured'].tolist()}")
        try:
            bus_voltages = self.net.res_bus.vm_pu.values
            power_flows = self.net.res_line.loading_percent.values
            real_transformer_loading = np.nan_to_num(self.net.res_trafo.loading_percent.values, nan=0.0)
            actual_temperature = self.T_ambient + self.T_rated * (real_transformer_loading[trafo_index]/100) ** self.n
            if "temperature_measured" in self.net.trafo.columns:
                temperature_reading = self.net.trafo.at[trafo_index, "temperature_measured"]
            else:
                print("no temperature reading value!!!!!!!!!!!!!!")
            self.temperature_history[trafo_index] = np.roll(self.temperature_history[trafo_index], shift=-1)
            self.temperature_history[trafo_index][-1] = temperature_reading
            temperature_trend = np.mean(self.temperature_history[trafo_index])

            local_state = np.concatenate([
                bus_voltages,  # length is num_buses
                power_flows,  # length is num_lines
                [real_transformer_loading[trafo_index]],
                [actual_temperature],
                [temperature_trend],
                [self.p[trafo_index]],
                [temperature_reading]
            ])

            assert len(local_state) == self.get_state_size(), f"State size mismatch: expected {self.get_state_size()}, got {len(local_state)}"
            return local_state

        except Exception as e:
            print(f"Error in getting state for Transformer {trafo_index}: {e}")
            return np.zeros(self.get_state_size())

    def get_local_reward(self, state):
        num_buses = len(self.net.res_bus.vm_pu)
        num_lines = len(self.net.res_line.loading_percent)
        bus_voltages = state[:num_buses]
        power_flows = state[num_buses:num_buses + num_lines]
        real_loading = state[-5]
        actual_temperature = state[-4]
        temperature_trend = state[-3]
        p_disconnect = state[-2]
        temperature_reading = state[-1]

        voltage_deviation = np.sum(np.maximum(np.abs(bus_voltages - 1.0) - self.voltage_tolerance, 0))
        voltage_penalty = -voltage_deviation * self.voltage_penalty_factor

        overloaded_lines = np.sum(power_flows > self.line_loading_limit)
        line_penalty = -overloaded_lines * self.power_flow_penalty_factor

        temp_diff = temperature_reading - temperature_trend
        if temp_diff > 3:
            temp_penalty = -10
        elif temp_diff < -3:
            temp_penalty = -5
        else:
            temp_penalty = 5


        reward = (
                voltage_penalty +
                line_penalty +
                temp_penalty
        )

        return reward

    def get_state_size(self):
        if not hasattr(self.net, "res_bus") or self.net.res_bus.empty:
            try:
                pp.runpp(self.net)
            except pp.LoadflowNotConverged:
                print("[Env] Power flow failed, cannot get state size")
                return 0
        num_buses = len(self.net.res_bus.vm_pu)
        num_lines = len(self.net.res_line.loading_percent)

        transformer_features = 5

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





