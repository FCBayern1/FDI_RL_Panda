import torch
import numpy as np
from pandapower.control.basic_controller import Controller

import pandapower as pp
from models.dqn_agent import DQNAgent

class DQNTransformerController(Controller):
    def __init__(self, net, env, trafo_index, max_temperature, T_ambient=25.0, T_rated=65.0, n=1.6, fdi_list=None,
                 total_steps=200, in_service=True, order=0, level=0, model_path=None, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level, **kwargs)
        self.net = net
        self.env = env
        self.trafo_index = trafo_index
        self.max_temperature = max_temperature
        self.T_ambient = T_ambient
        self.T_rated = T_rated
        self.n = n
        self.fdi_list = fdi_list if fdi_list is not None else []
        self.total_steps = total_steps
        self.model_path = model_path
        self.current_time_step = None
        self.trafo_disconnected = False
        self.controller_converged = False
        self.agent = None
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

        if model_path:
            self.agent = DQNAgent(env.get_state_size(), 2)
            self.agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.agent.eval()

    def calculate_temperature(self, loading_percent):
        # Calculate the transformer temperature based on loading percent
        return self.T_ambient + self.T_rated * (loading_percent) ** self.n

    def control_step(self, net):
        if self.controller_converged:
            return
        time_step = self.current_time_step
        if time_step is None:
            return
        try:
            actual_loading_percent = np.nan_to_num(net.res_trafo.at[self.trafo_index, 'loading_percent'], 0.0)
        except KeyError:
            print(f"Time step {time_step}: KeyError - No data available for transformer at index {self.trafo_index}")
            self.controller_converged = True
            return

        actual_temperature = self.calculate_temperature(actual_loading_percent)
        self.net.trafo.at[self.trafo_index, 'temperature_measured'] = actual_temperature
        print(f"\n Time step {time_step}: The actual temperature of transformer {self.trafo_index} is {actual_temperature:.2f}°C, actual loading percent is {actual_loading_percent:.2f}")

        current_temperature_reading = actual_temperature
        fdi_attack = False
        for f_step, faulty_temperature in self.fdi_list:
            if f_step == time_step:
                self.net.trafo.at[self.trafo_index, 'temperature'] = faulty_temperature
                current_temperature_reading = faulty_temperature
                print(f"🌹🌹Time step {time_step}: FDI Injected, setting trafo {self.trafo_index} temperature to {faulty_temperature}°C")
                fdi_attack = True
                break

        if self.net.trafo.at[self.trafo_index, 'temperature_measured'] is None:
            self.net.trafo.at[self.trafo_index, 'temperature_measured'] = actual_temperature

        print(f"Time step {time_step}: Transformer {self.trafo_index} current reading: {current_temperature_reading:.2f}°C")

        state = self.env.get_local_state(self.trafo_index)
        action = self.select_action(state)
        self.env.update_disconnection_probabilities([action])
        net.trafo.at[self.trafo_index, "in_service"] = np.random.rand() > self.env.p[self.trafo_index]
        status_str = "Disconnected" if not net.trafo.at[self.trafo_index, "in_service"] else "In Service"
        print(
            f"Time step {time_step}, RL agent set the trafo {self.trafo_index} {status_str}, with disconnection p {self.env.p[self.trafo_index]}")
        predicted_disconnect = not net.trafo.at[self.trafo_index, "in_service"] # trafo disconnection (if disconnect then true, if in_service then false)
        actual_overtemp = actual_temperature > self.max_temperature # if actual temperature is overtemp then true, if normal then false
        # if fdi_attack:
        #     if actual_overtemp and predicted_disconnect:
        #         self.tp += 1
        #     elif not actual_overtemp and predicted_disconnect:
        #         self.fp += 1
        # else:
        #     if not actual_overtemp and not predicted_disconnect:
        #         self.tn += 1
        #     elif actual_overtemp and predicted_disconnect:
        #         self.fn += 1
        if actual_overtemp and predicted_disconnect:
            self.tp += 1
        elif not actual_overtemp and predicted_disconnect:
            self.fn += 1
        elif actual_overtemp and not predicted_disconnect:
            self.fp += 1
        elif not actual_overtemp and not predicted_disconnect:
            self.tn += 1

        self.env.step_count = self.current_time_step
        self.controller_converged = True

    def select_action(self, state):
        if self.agent is None:
            print(f"Warning: No trained DQN agent available for transformer {self.trafo_index}")
            return 0
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.agent(state_tensor)
        return int(torch.argmax(q_values).item())

    def time_step(self, net, time):
        self.current_time_step = time
        self.controller_converged = False  # Reset convergence at the beginning of each time step

    def is_converged(self, net):
        return self.controller_converged
