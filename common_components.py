# common_components.py

import torch
import torch.nn as nn
import random

# Replay buffer for experience storage
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Simple Self-Supervised Model for future interaction prediction
class SSLModel(nn.Module):
    def __init__(self, state_dim, item_dim):
        super(SSLModel, self).__init__()
        self.fc = nn.Linear(state_dim, item_dim)  # predicts next item vector

    def forward(self, state):
        return self.fc(state)
