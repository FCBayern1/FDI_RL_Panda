# rl_controller.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandapower as pp

from models.dqn_agent import DQNAgent
from utils.network import create_network


class new_RLController:
    def __init__(self, env, net, trafo_indices, memory_capacity=10000, batch_size=32, gamma=0.99, epsilon=0.1,
                 update_target_every=10):
        self.env = env
        self.net = net
        self.trafo_indices = trafo_indices
        self.memory = []
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_target_every = update_target_every
        self.step_counter = 0

        # 初始化 DQN 网络
        self.policy_net = DQNAgent(self.env.get_state_size(), len(self.trafo_indices) * 2)
        self.target_net = DQNAgent(self.env.get_state_size(), len(self.trafo_indices) * 2)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        # 记录数据
        self.q_value_log = []
        self.loss_log = []

    def step(self):
        state = self.env._get_state()
        actions = self.select_actions(state)

        # 更新变压器状态概率
        self.env.update_disconnection_probabilities(actions)

        # 运行 pandapower 潮流计算
        try:
            pp.runpp(self.env.net)
        except pp.LoadflowNotConverged:
            print("⚠️ Power flow calculation did not converge.")

        # 获取新状态和奖励
        next_state = self.env._get_state()
        rewards = self.env.get_reward()

        # 存储经验
        self.memory.append((state, actions, rewards, next_state))
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)

        # 训练 DQN
        if len(self.memory) >= self.batch_size:
            self.learn()

        return next_state, rewards

    def select_actions(self, state):
        if np.random.rand() < self.epsilon:
            actions = np.random.randint(0, 2, size=len(self.trafo_indices))
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            actions = [int(torch.argmax(q_values[0][i:i + 2]).item()) for i in range(0, len(q_values[0]), 2)]
        return actions

    def learn(self):
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        experiences = [self.memory[i] for i in batch]

        states = torch.FloatTensor([exp[0] for exp in experiences])
        actions = torch.LongTensor([exp[1] for exp in experiences]).unsqueeze(1)
        rewards = torch.FloatTensor([exp[2] for exp in experiences]).unsqueeze(1)
        next_states = torch.FloatTensor([exp[3] for exp in experiences])

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)
        self.q_value_log.append(q_values.mean().item())
        self.loss_log.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step_counter % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.step_counter += 1


# test3.py
from rl_controller import RLController
from envs.substation_env import SubstationEnv
import pandapower as pp

time_steps = 200

net = create_network(time_steps)
trafo_indices = list(net.trafo.index)

env = SubstationEnv(
    net,
    trafo_indices=trafo_indices,
    delta_p=0.05,
    initial_p=0.0,
    voltage_tolerance=0.05,
    voltage_penalty_factor=10.0,
    line_loading_limit=100.0,
    power_flow_penalty_factor=5.0,
    load_reward_factor=20.0,
    transformer_reward_factor=2.0,
    disconnection_penalty_factor=50.0,
    total_steps=time_steps
)

rl_controller = RLController(env, net, trafo_indices)

for t in range(time_steps):
    print(f"🔄 Time Step {t} 🔄")
    next_state, rewards = rl_controller.step()
    print(f"State: {next_state}")
    print(f"Reward: {rewards}")

    if t % 10 == 0:
        pp.to_json(rl_controller.env.net, f"powergrid_state_{t}.json")
