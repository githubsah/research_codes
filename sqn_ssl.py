# sqn_ssl.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common_components import ReplayBuffer, SSLModel

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SQNAgent:
    def __init__(self, state_dim, action_dim, item_dim, gamma=0.99):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_q_network = QNetwork(state_dim, action_dim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

        self.ssl_model = SSLModel(state_dim, item_dim)
        self.ssl_optimizer = optim.Adam(self.ssl_model.parameters(), lr=0.001)

        self.buffer = ReplayBuffer(10000)
        self.gamma = gamma

    def select_action(self, state):
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def update(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return
        
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Self-supervised reward augmentation
        predicted_next_items = self.ssl_model(states)
        
        # Adjust the target shape to match predicted_next_items
        ssl_targets = self.ssl_model(next_states)  # Project next_states into item space

        ssl_loss = F.mse_loss(predicted_next_items, ssl_targets)

        ssl_reward = -ssl_loss.detach()

        # Final reward
        final_reward = rewards + 0.1 * ssl_reward

        # Q-learning update
        q_values = self.q_network(states)
        next_q_values = self.target_q_network(next_states)

        target_q = final_reward + self.gamma * (1 - dones) * next_q_values.max(1)[0]
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(current_q, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # SSL model update
        self.ssl_optimizer.zero_grad()
        ssl_loss.backward()
        self.ssl_optimizer.step()

    def soft_update_target(self, tau=0.01):
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
