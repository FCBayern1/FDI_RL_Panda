import torch
import numpy as np
from pandapower.control.basic_controller import Controller
from src.networks.DDPG import Actor

class DDPGTransformerController(Controller):
    def __init__(self, env, trafo_index, max_temperature,
                 T_ambient=25.0, T_rated=65.0, n=1.6,
                 fdi_list=None, total_steps=200,
                 in_service=True, order=0, level=0,
                 model_path=None, **kwargs):
        super().__init__(env.net, in_service=in_service, order=order, level=level, **kwargs)
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
        self.controller_converged = False
        self.actor = None

        self.tp = self.fp = self.fn = self.tn = 0

        if model_path:
            self.actor = Actor(env.get_state_size())
            self.actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.actor.eval()

    def calculate_temperature(self, loading_percent):
        return self.T_ambient + self.T_rated * (loading_percent/100) ** self.n

    def control_step(self, net):
        if self.controller_converged:
            return

        time_step = self.current_time_step
        if time_step is None:
            return

        try:
            loading_percent = np.nan_to_num(net.res_trafo.at[self.trafo_index, 'loading_percent'], 0.0)
        except KeyError:
            print(f"Time step {time_step}: KeyError - No data for transformer {self.trafo_index}")
            self.controller_converged = True
            return

        actual_temp = self.calculate_temperature(loading_percent)

        if loading_percent > 150:
            print(f"overloading happens!!! in trafo {self.trafo_index}")

        net.trafo.at[self.trafo_index, 'actual_temperature'] = actual_temp
        net.trafo.at[self.trafo_index, 'temperature_measured'] = actual_temp

        print(f"Time step {time_step}: Actual temperature of transformer {self.trafo_index} = {actual_temp:.2f}°C")

        # 注入 FDI
        temp_reading = actual_temp
        for f_step, fake_temp in self.fdi_list:
            if f_step == time_step:
                net.trafo.at[self.trafo_index, 'temperature_measured'] = fake_temp
                temp_reading = fake_temp
                print(f"🌹 FDI Injected: step {time_step}, transformer {self.trafo_index}, fake temp = {fake_temp:.2f}°C")
                break

        state = self.normalize_state(self.env.get_local_state(self.trafo_index))
        # print(state)
        action = self.select_action(state)
        self.env.p[self.trafo_index] = action
        net.trafo.at[self.trafo_index, "in_service"] = np.random.rand() > action # ！！！
        # if loading_percent > 1:
        #     net.trafo.at[self.trafo_index, "in_service"] = False
        # if loading_percent > 1.0:
        #     overload_factor = loading_percent - 1.0
        #     net.trafo.at[self.trafo_index, "sn_mva"] *= (1 - 0.5 * overload_factor)
        #     print(f"[Info] Transformer {self.trafo_index} overload: reducing capacity by {0.5 * overload_factor:.2f}")

        is_disconnected = not net.trafo.at[self.trafo_index, "in_service"]

        print(f"Step {time_step}: RL agent sets trafo {self.trafo_index} {'Disconnected' if is_disconnected else 'In Service'} (p={action:.6f})")

        # Confusion Matrix 记录
        should_disconnect = actual_temp > self.max_temperature
        if should_disconnect and is_disconnected:
            self.tp += 1
        elif should_disconnect and not is_disconnected:
            self.fn += 1
        elif not should_disconnect and is_disconnected:
            self.fp += 1
        else:
            self.tn += 1

        self.env.step_count = time_step
        self.controller_converged = True

    def select_action(self, state):
        if self.actor is None:
            print(f"Warning: no trained DDPG actor available for transformer {self.trafo_index}")
            return 0.0
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.actor(state_tensor).squeeze().item()
        return float(np.clip(action, 0.0, 1.0))

    def time_step(self, net, time):
        self.current_time_step = time
        self.controller_converged = False

    def is_converged(self, net):
        return self.controller_converged

    def print_confusion_matrix(self):
        print("\n[Confusion Matrix Summary]")
        print(f"TP (该断断了):     {self.tp}")
        print(f"FN (该断没断):     {self.fn}")
        print(f"TN (不该断没断):   {self.tn}")
        print(f"FP (不该断断了):   {self.fp}")
        total = self.tp + self.fp + self.fn + self.tn
        accuracy = (self.tp + self.tn) / total if total > 0 else 0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 Score:  {f1:.3f}")

    def normalize_state(self, state):
        if state is None or len(state) == 0:
            return np.zeros(self.env.get_state_size())
        state_min, state_max = np.min(state), np.max(state)
        return np.nan_to_num((state - state_min) / (state_max - state_min) if state_max - state_min > 0 else state)


