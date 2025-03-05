import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pandapower.control.basic_controller import Controller
from models.dqn_agent import DQNAgent
import pandapower as pp

class multi_agent_controller(Controller):

    def __init__(self, env, net, trafo_indices, **kwargs):
        super().__init__(net, **kwargs)
        self.env = env
        self.net = net
        self.trafo_indices = trafo_indices
        self.agents = {}
        self.applied = False

        for trafo_index in self.trafo_indices:
            self.agents[trafo_index] = {
                "policy_net": DQNAgent(self.env.get_state_size(), 2),  # Each agent has 2 actions (disconnect/stay)
                "target_net": DQNAgent(self.env.get_state_size(), 2),
                "memory": [],
                "memory_capacity": 10000,
                "batch_size": 64,
                "gamma": 0.99,
                "epsilon": 0.1,
                "step_counter": 0
            }
            self.agents[trafo_index]["target_net"].load_state_dict(self.agents[trafo_index]["policy_net"].state_dict())
            self.agents[trafo_index]["target_net"].eval()
            self.agents[trafo_index]["optimizer"] = optim.Adam(self.agents[trafo_index]["policy_net"].parameters(),
                                                           lr=0.001)
        self.update_target_every = 10

    def control_step(self, net):
        state_dict = {idx: self.normalize_state(self.env.get_local_state(idx)) for idx in self.trafo_indices}
        action_dict = {}
        for trafo_index in self.trafo_indices:
            agent = self.agents[trafo_index]

            state = state_dict[trafo_index]

            action = self.select_action(agent, state)

            action_dict[trafo_index] = action

            if action == 1:
                self.env.p[trafo_index] = min(self.env.p[trafo_index] + self.env.delta_p, 1.0)
            else:
                self.env.p[trafo_index] = max(self.env.p[trafo_index] - self.env.delta_p, 0.0)

            net.trafo.at[trafo_index, "in_service"] = np.random.rand() > self.env.p[trafo_index]
            status_str = "Disconnected" if not net.trafo.at[trafo_index, "in_service"] else "In Service"
            print(f"Time step {self.env.step_count}, RL agent set the trafo {trafo_index} {status_str}, with disconnection p {self.env.p[trafo_index]}")


        try:
            pp.runpp(net)
        except pp.LoadflowNotConverged:
            print("Power flow not converge, skipping this step.")

        next_state_dict = {idx: self.normalize_state(self.env.get_local_state(idx)) for idx in self.trafo_indices}

        for trafo_index in self.trafo_indices:
            state = state_dict[trafo_index]

            action = action_dict[trafo_index]

            agent = self.agents[trafo_index]

            next_state = next_state_dict[trafo_index]

            reward = self.env.get_local_reward(next_state)

            done = self.env.step_count >= self.env.total_steps

            agent['memory'].append((state, action, reward, next_state, done))

            if len(agent["memory"]) > agent["memory_capacity"]:
                agent["memory"].pop(0)

            if len(agent["memory"]) >= agent["batch_size"]:
                self.learn(agent)

            agent['step_counter'] += 1

        self.applied = True
        self.env.step_count += 1

    def normalize_state(self, state):
        if state is None or len(state) == 0:
            return np.zeros(self.env.get_state_size(0))
        state_min, state_max = np.min(state), np.max(state)
        return np.nan_to_num((state - state_min) / (state_max - state_min) if state_max - state_min > 0 else state)

    def select_action(self, agent, state):
        if np.random.rand() < agent["epsilon"]:
            return np.random.randint(0, 2)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent["policy_net"](state_tensor)
        return int(torch.argmax(q_values).item())

    def learn(self, agent):
        batch = np.random.choice(len(agent["memory"]), agent["batch_size"], replace=False)
        experiences = [agent["memory"][i] for i in batch]

        states = torch.FloatTensor([exp[0] for exp in experiences])
        actions = torch.LongTensor([exp[1] for exp in experiences]).unsqueeze(1)
        rewards = torch.FloatTensor([exp[2] for exp in experiences]).unsqueeze(1)
        next_states = torch.FloatTensor([exp[3] for exp in experiences])
        dones = torch.BoolTensor([exp[4] for exp in experiences]).unsqueeze(1)

        with torch.no_grad():
            best_action_indices = agent["policy_net"](next_states).argmax(1, keepdim=True)
            next_q_values = agent["target_net"](next_states).gather(1, best_action_indices)
            target_q_values = rewards + agent["gamma"] * next_q_values * (~dones)

        q_values = agent["policy_net"](states).gather(1, actions)
        loss = nn.MSELoss()(q_values, target_q_values)

        agent["optimizer"].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent["policy_net"].parameters(), max_norm=1.0)
        agent["optimizer"].step()

    def is_converged(self, net):
        return self.applied

    def time_step(self, net, time):
        self.applied = False

