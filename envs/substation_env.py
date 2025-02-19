import numpy as np
import pandapower as pp

class SubstationEnv:
    def __init__(self, net, trafo_indices, delta_p=0.05, initial_p=0.0,
                 voltage_tolerance=0.05, voltage_penalty_factor=10.0,
                 line_loading_limit=100.0, power_flow_penalty_factor=5.0,
                 load_reward_factor=20.0, transformer_reward_factor=20.0,
                 disconnection_penalty_factor=50.0, total_steps=200, T_ambient=25.0, ΔT_rated=65.0, n=1.6):
        self.net = net
        self.trafo_indices = trafo_indices
        self.delta_p = delta_p  # Amount to adjust the disconnection probability
        self.initial_p = initial_p  # Initial disconnection probability
        self.step_count = 0
        self.total_steps = total_steps

        # Initialise disconnection probabilities for each transformer
        self.p = {index: self.initial_p for index in self.trafo_indices}

        # Reward function parameters
        self.voltage_tolerance = voltage_tolerance
        self.voltage_penalty_factor = voltage_penalty_factor
        self.line_loading_limit = line_loading_limit
        self.power_flow_penalty_factor = power_flow_penalty_factor
        self.load_reward_factor = load_reward_factor
        self.transformer_reward_factor = transformer_reward_factor
        self.disconnection_penalty_factor = disconnection_penalty_factor

        self.total_initial_load = self.net.load['p_mw'].sum()
        self.T_ambient = T_ambient
        self.ΔT_rated = ΔT_rated
        self.n = n

        # Observation space and action space are defined in the RLController

    def reset(self):
        self.step_count = 0

        # Reset transformers to in-service
        for index in self.trafo_indices:
            self.net.trafo.at[index, 'in_service'] = True

        # Reset disconnection probabilities
        self.p = {index: self.initial_p for index in self.trafo_indices}

        # Reset loads to initial values
        self.net.load['p_mw'] = self.net.load['p_mw'].apply(lambda x: x)

        # Run initial power flow calculation
        try:
            pp.runpp(self.net)
        except pp.LoadflowNotConverged:
            print("Initial power flow did not converge.")
            pass  # Handle as needed

        return self._get_state()

    def _get_state(self):
        try:
            bus_voltages = self.net.res_bus.vm_pu.values  # Shape: (num_buses,)
            individual_loads = self.net.load.p_mw.values  # Actual loads
            power_flow = self.net.res_line.loading_percent.values  # Line loadings
            transformer_loading = self.net.res_trafo.loading_percent.values  # Transformer loadings

            # Combine all observations into a single array
            state = np.concatenate([
                bus_voltages,
                individual_loads,
                power_flow,
                transformer_loading  # Add transformer loading to the state
            ])

            return state
        except Exception as e:
            print(f"Error in getting state: {e}")
            return None

    def get_state_size(self):
        num_buses = len(self.net.bus)
        num_loads = len(self.net.load)
        num_lines = len(self.net.line)
        num_transformers = len(self.net.trafo)  # Adding transformer loading

        state_size = num_buses + num_loads + num_lines + num_transformers  # Including transformer loadings
        return state_size

    def _calculate_reward(self, state):
        num_buses = len(self.net.bus)
        num_loads = len(self.net.load)
        num_lines = len(self.net.line)
        num_transformers = len(self.net.trafo)

        bus_voltages = state[:num_buses]
        individual_loads = state[num_buses:num_buses+num_loads]
        power_flow = state[num_buses+num_loads:num_buses+num_loads+num_lines]
        transformer_loading = state[num_buses+num_loads+num_lines:]

        # Initialise an array to store individual transformer rewards
        transformer_rewards = np.zeros(len(self.trafo_indices))

        # Voltage stability penalty (optional per transformer)
        voltage_deviation = np.sum(np.maximum(np.abs(bus_voltages - 1.0) - self.voltage_tolerance, 0))
        voltage_penalty = -voltage_deviation * self.voltage_penalty_factor

        # Calculate individual transformer rewards
        for i, loading in enumerate(transformer_loading):
            if loading > 100:  # Overloaded transformer
                transformer_rewards[i] -= (loading - 100) * self.transformer_reward_factor
            else:
                transformer_rewards[i] += (100 - loading) * self.transformer_reward_factor

        # Calculate line loading penalties (applies to transformers connected to overloaded lines)
        for i, loading in enumerate(power_flow):
            if loading > self.line_loading_limit:
                transformer_rewards[i % len(self.trafo_indices)] -= (loading - self.line_loading_limit) * self.power_flow_penalty_factor



        # Reward for maintaining transformer service
        for i, index in enumerate(self.trafo_indices):
            in_service = True
            switches = self.net.switch[(self.net.switch['element'] == index) & (self.net.switch['et'] == 't')]
            for _, switch in switches.iterrows():
                if not switch['closed']:
                    in_service = False
                    break

            if in_service:
                transformer_rewards[i] += self.transformer_reward_factor
            else:
                transformer_rewards[i] -= self.disconnection_penalty_factor


        return transformer_rewards  # Return individual transformer rewards

    def get_reward(self):
        state = self._get_state()
        rewards = self._calculate_reward(state)
        return rewards  # Returning an array of rewards, one for each transformer

    def update_disconnection_probabilities(self, actions):
        # Actions: list of actions for each transformer
        # 0: decrease disconnection probability
        # 1: increase disconnection probability

        # Update disconnection probabilities based on actions
        for idx, act in enumerate(actions):
            trafo_index = self.trafo_indices[idx]
            if act == 0:
                self.p[trafo_index] = max(self.p[trafo_index] - self.delta_p, 0.0)
            elif act == 1:
                self.p[trafo_index] = min(self.p[trafo_index] + self.delta_p, 1.0)
