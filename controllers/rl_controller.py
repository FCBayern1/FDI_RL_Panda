import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pandapower.control.basic_controller import Controller
from models.dqn_agent import DQNAgent
import pandapower as pp

class RLController(Controller):
    def __init__(self, env, net, trafo_indices, **kwargs):
        super().__init__(net, **kwargs)
        self.env = env
        self.net = net
        self.trafo_indices = trafo_indices
        self.applied = False

        self.p = self.env.p

        # Define state and action sizes
        self.state_size = self.env.get_state_size()
        self.num_transformers = len(self.trafo_indices)
        self.action_size = 2  # For each transformer: 0 (no disconnection) or 1 (disconnection)

        # Initialise DQN
        self.policy_net = DQNAgent(self.state_size, self.action_size * self.num_transformers)
        self.target_net = DQNAgent(self.state_size, self.action_size * self.num_transformers)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = []
        self.memory_capacity = 10000
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.1
        self.update_target_every = 10
        self.train_every = 5
        self.step_counter = 0
        self.episode_rewards = []
        self.q_value_history = []


    def control_step(self, net):
        state = self.env._get_state()
        state = self.normalize_state(state)

        actions = self.select_actions(state)

        for idx, action in enumerate(actions):
            trafo_index = self.trafo_indices[idx]
            if action == 1:
                self.env.p[trafo_index] = min(self.env.p[trafo_index] + self.env.delta_p, 1.0)
            else:
                self.env.p[trafo_index] = max(self.env.p[trafo_index] - self.env.delta_p, 0.0)

            if np.random.rand() < self.env.p[trafo_index]:
                net.trafo.at[trafo_index, "in_service"] = False
            else:
                net.trafo.at[trafo_index, "in_service"] = True

        if not net.trafo["in_service"].any():
            net.trafo.at[self.trafo_indices[0], "in_service"] = True
            self.env.p[self.trafo_indices[0]] = 0.0
            print(f"No transformer in service, Reconnecting transformer {self.trafo_indices[0]}.")

        self.applied = True
        self.env.step_count += 1

        rewards = np.clip(self.env.get_reward(), -10, 10)
        self.episode_rewards.append(sum(rewards))

        try:
            pp.runpp(net)
        except pp.LoadflowNotConverged:
            print("Power flow did not converge, skipping this step.")

        next_state = self.env._get_state()
        next_state = self.normalize_state(next_state)

        done = self.env.step_count >= self.env.total_steps

        for idx, reward in enumerate(rewards):
            self.memory.append((state, actions[idx], reward, next_state, done))

        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)

        if len(self.memory) >= self.batch_size and self.step_counter % self.train_every == 0:
            self.learn()

        if done and self.step_counter % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.step_counter += 1

        q_values = self.policy_net(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()[0]
        q_values = np.clip(q_values, -10, 10)
        self.q_value_history.append(np.mean(q_values))

        transformer_statuses = [net.trafo.at[i, "in_service"] for i in self.trafo_indices]
        print(f"Time Step {self.env.step_count}: Rewards: {rewards}")
        print(f"Transformer Disconnection Probabilities: {[self.env.p[i] for i in self.trafo_indices]}")
        print(f"Transformer Statuses: {transformer_statuses}")
        print(f"Q-values: {q_values}\n")

    def learn(self):
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        experiences = [self.memory[i] for i in batch]

        states = torch.FloatTensor([exp[0] for exp in experiences])
        actions = torch.LongTensor([exp[1] for exp in experiences]).unsqueeze(1)
        rewards = torch.FloatTensor([exp[2] for exp in experiences]).unsqueeze(1)
        next_states = torch.FloatTensor([exp[3] for exp in experiences])
        dones = torch.BoolTensor([exp[4] for exp in experiences]).unsqueeze(1)

        with torch.no_grad():
            best_action_indices = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, best_action_indices)
            target_q_values = rewards + self.gamma * next_q_values * (~dones)
            target_q_values = torch.clamp(target_q_values, -10, 10)

        q_values = self.policy_net(states).gather(1, actions)
        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def normalize_state(self, state):
        if state is None or len(state) == 0:
            return np.zeros(self.state_size)
        state_min, state_max = np.min(state), np.max(state)
        return np.nan_to_num((state - state_min) / (state_max - state_min) if state_max - state_min != 0 else state)

    def select_actions(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2, size=self.num_transformers)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return [int(torch.argmax(q_values[0][i:i + 2]).item()) for i in range(0, len(q_values[0]), 2)]

    def is_converged(self, net):
        return self.applied

    def time_step(self, net, time):
        self.applied = False

    def log_episode_metrics(self, episode):
        avg_reward = np.mean(self.episode_rewards)
        avg_q_value = np.mean(self.q_value_history) if self.q_value_history else 0
        print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Q-value = {avg_q_value:.2f}, Total Steps = {self.env.step_count}")
        self.episode_rewards = []
        self.q_value_history = []


