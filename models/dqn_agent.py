import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class MultiAgentTrainer:
    def __init__(self, state_size, action_size, trafo_indices, lr=0.0005, batch_size=64, gamma=0.99, epsilon=0.1):
        self.agents = {}
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon

        for idx in trafo_indices:
            policy_net = DQNAgent(state_size, action_size)
            target_net = DQNAgent(state_size, action_size)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
            optimizer = optim.Adam(policy_net.parameters(), lr=lr)

            self.agents[idx] = {
                "policy_net": policy_net,
                "target_net": target_net,
                "optimizer": optimizer,
                "memory": [],
                "batch_size": 64,
                "gamma": 0.99,
                "epsilon": 0.1,
                "step_counter": 0,
                "memory_capacity": 10000,
                "loss_history": []
            }

    # def select_action(self, idx, state):
    #     if np.random.rand() < self.epsilon:
    #         return np.random.randint(0, 2)
    #     state_tensor = torch.FloatTensor(state).unsqueeze(0)
    #     with torch.no_grad():
    #         q_values = self.agents[idx]["policy_net"](state_tensor)
    #     return int(torch.argmax(q_values).item())

    def store(self, idx, transition):
        agent = self.agents[idx]
        agent["memory"].append(transition)
        if len(agent["memory"]) > agent["memory_capacity"]:
            agent["memory"].pop(0)

    def train(self, idx):
        # print("training!!!")
        agent = self.agents[idx]
        memory = agent["memory"]
        # print("length of memory", len(memory))
        if len(memory) < self.batch_size:
            return

        batch = random.sample(memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)

        # Compute current Q values
        q_values = agent["policy_net"](states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            best_next_actions = agent["policy_net"](next_states).argmax(1, keepdim=True)
            target_q_values = agent["target_net"](next_states).gather(1, best_next_actions)
            expected_q_values = rewards + self.gamma * target_q_values * (~dones)

        # Compute and optimize loss
        loss = nn.MSELoss()(q_values, expected_q_values)
        agent["optimizer"].zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent["policy_net"].parameters(), max_norm=1.0)
        agent["optimizer"].step()

        agent["loss_history"].append(loss.item())

    def store_experience(self, idx, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.store(idx, transition)

    def train_if_ready(self, idx):
        self.train(idx)
        self.agents[idx]["step_counter"] += 1

    def update_target(self, idx):
        agent = self.agents[idx]
        agent["target_net"].load_state_dict(agent["policy_net"].state_dict())

    def update_all_targets(self):
        for idx in self.agents:
            self.update_target(idx)

    def get_policy_net(self, idx):
        return self.agents[idx]["policy_net"]

    def save_all_models(self, prefix_path="./dqn_models"):
        for idx, agent in self.agents.items():
            torch.save(agent["policy_net"].state_dict(), f"{prefix_path}/trafo_{idx}_dqn.pth")

    def load_all_models(self, prefix_path="./dqn_models"):
        for idx, agent in self.agents.items():
            model_path = f"{prefix_path}/trafo_{idx}_dqn.pth"
            agent["policy_net"].load_state_dict(torch.load(model_path))
            agent["target_net"].load_state_dict(agent["policy_net"].state_dict())
            agent["target_net"].eval()

    def get_loss_history(self, idx):
        return self.agents[idx]["loss_history"]

    def plot_loss_curves(self):
        for idx, agent in self.agents.items():
            loss_list = agent["loss_history"]
            if not loss_list:
                continue

            plt.figure(figsize=(6, 4))
            plt.plot(loss_list, label=f"Transformer {idx}", color='blue')
            plt.xlabel("Training Steps")
            plt.ylabel("Loss")
            plt.title(f"DQN Loss - Transformer {idx}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"./loss_plot_trafo_{idx}.png")
            plt.close()

    def plot_loss_curves_1(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for idx, agent in self.agents.items():
            loss_list = agent["loss_history"]
            if loss_list:
                plt.plot(loss_list, label=f"Transformer {idx}")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("DQN Loss per Transformer")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


