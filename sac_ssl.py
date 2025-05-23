# sac_ssl.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common_components import ReplayBuffer, SSLModel

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.q = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

class SACAgent:
    def __init__(self, state_dim, action_dim, item_dim, gamma=0.99):
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.target_critic1 = Critic(state_dim, action_dim)
        self.target_critic2 = Critic(state_dim, action_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=0.001)
        
        self.ssl_model = SSLModel(state_dim, item_dim)
        self.ssl_optimizer = optim.Adam(self.ssl_model.parameters(), lr=0.001)

        self.buffer = ReplayBuffer(10000)
        self.gamma = gamma

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=0.001)

        # Initialize target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def select_action(self, state):
        with torch.no_grad():
            mean, std = self.actor(state)
            action = mean
        return action

    def update(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return
        
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # SSL reward augmentation
        predicted_next_items = self.ssl_model(states)
        ssl_targets = next_states
        ssl_loss = F.mse_loss(predicted_next_items, ssl_targets)

        ssl_reward = -ssl_loss.detach()
        final_reward = rewards + 0.1 * ssl_reward

        # Update critics
        with torch.no_grad():
            next_action, next_std = self.actor(next_states)
            next_q1 = self.target_critic1(next_states, next_action)
            next_q2 = self.target_critic2(next_states, next_action)
            next_q = torch.min(next_q1, next_q2)
            target_q = final_reward.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * next_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        mean, std = self.actor(states)
        new_action = mean
        q1 = self.critic1(states, new_action)
        q2 = self.critic2(states, new_action)
        actor_loss = (self.log_alpha.exp() * -q1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # SSL model update
        self.ssl_optimizer.zero_grad()
        ssl_loss.backward()
        self.ssl_optimizer.step()

        # Update alpha
        alpha_loss = -(self.log_alpha.exp() * (q1 + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def soft_update_target(self, tau=0.005):
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
