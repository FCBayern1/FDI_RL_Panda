import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pandapower.control.basic_controller import Controller
from models.dqn_agent import DQNAgent
import pandapower as pp

class RLController(Controller):
    def __init__(self, env, net, trafo_indices, **kwargs):
        super().__init__(net, **kwargs)
        self.last_loss = None
        self.reward_log = []
        self.q_value_log = []
        self.loss_log = []
        self.time_step_counter = 0
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
        self.step_counter = 0

    def control_step(self, net):
        # 1. Fetch the current state from the environment
        state = self.env._get_state()
        state = self.normalize_state(state)

        # 2. Decide actions using the policy
        actions = self.select_actions(state)

        # 3. Update disconnection probabilities and apply to each transformer
        for idx, action in enumerate(actions):
            trafo_index = self.trafo_indices[idx]
            if action == 1:  # Action 1 increases disconnection probability
                self.env.p[trafo_index] = min(self.env.p[trafo_index] + self.env.delta_p, 1.0)
            else:  # Action 0 decreases disconnection probability
                self.env.p[trafo_index] = max(self.env.p[trafo_index] - self.env.delta_p, 0.0)

            # Disconnect based on the current disconnection probability
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

        # Calculate individual rewards for each transformer
        rewards = self.env.get_reward()

        try:
            pp.runpp(net)
        except pp.LoadflowNotConverged:
            print("Power flow did not converge, skipping this step.")
            state = self.reward_log[-1] if self.reward_log else np.zeros(self.state_size)

        # Get the next state
        next_state = self.env._get_state()
        next_state = self.normalize_state(next_state)

        # Check for terminal condition
        done = self.env.step_count >= self.env.total_steps

        # Store experience for each transformer
        for idx, reward in enumerate(rewards):
            self.memory.append((state, actions[idx], reward, next_state, done))
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)

        # Learn from experience
        if len(self.memory) >= self.batch_size:
            self.learn()

        # Update target network periodically
        if self.step_counter % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.step_counter += 1

        # Record the Q-value
        # with torch.no_grad():
        #     state_tensor = torch.FloatTensor(state).unsqueeze(0)
        #     q_values = self.policy_net(state_tensor)
        #     avg_q_value = q_values.mean().item()
        # self.q_value_log.append(avg_q_value)
        #
        # # Record the reward
        # self.reward_log.append(sum(rewards)/len(rewards))
        #
        # # Record loss values
        # self.loss_log.append(self.last_loss if self.last_loss is not None else None)
        #
        # self.save_logs()
        # self.time_step_counter += 1

        # Print the status and reward for each transformer
        statuses = [net.trafo.at[index, "in_service"] for index in self.trafo_indices]
        probabilities = [self.env.p[index] for index in self.trafo_indices]
        print(f"Time Step {self.env.step_count}: Rewards: {rewards}")
        print(f"Transformer statuses: {statuses}")
        print(f"Disconnection probabilities: {probabilities} \n")

    def normalize_state(self, state):
        if state is None or len(state) == 0:
            print("⚠️ Warning: Received empty state in normalize_state()")
            return np.zeros(self.state_size)  # 返回一个默认的零状态

        state_min = np.min(state)
        state_max = np.max(state)
        if state_max - state_min != 0:
            normalized_state = (state - state_min) / (state_max - state_min)
        else:
            normalized_state = state
        return np.nan_to_num(normalized_state)

    def save_logs(self, filename="training_logs.csv"):
        data = {
            "timestep": list(range(len(self.reward_log))),
            "reward": self.reward_log,
            "q_value": self.q_value_log,
            "loss": self.loss_log
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"📊 Logs saved to {filename}")

    def select_actions(self, state):
        # Choose actions for each transformer using epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            # Exploration: randomly select actions for each transformer
            actions = np.random.randint(0, 2, size=self.num_transformers)
        else:
            # Exploitation: select actions based on Q-values from policy network
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            # Choose actions based on Q-values for each transformer
            actions = [int(torch.argmax(q_values[0][i:i + 2]).item()) for i in range(0, len(q_values[0]), 2)]
        return actions

    def learn(self):
        # Sample a batch of experiences
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        experiences = [self.memory[i] for i in batch]

        states = torch.FloatTensor([exp[0] for exp in experiences])
        actions = torch.LongTensor([exp[1] for exp in experiences]).unsqueeze(1)
        rewards = torch.FloatTensor([exp[2] for exp in experiences]).unsqueeze(1)
        next_states = torch.FloatTensor([exp[3] for exp in experiences])
        dones = torch.BoolTensor([exp[4] for exp in experiences]).unsqueeze(1)

        # Compute current Q-values
        q_values = self.policy_net(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (~dones)

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimise the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.last_loss = loss.item()

    def is_converged(self, net):
        return self.applied

    def time_step(self, net, time):
        self.applied = False
