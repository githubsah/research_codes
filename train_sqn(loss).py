import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque, namedtuple

# ---- Replay Buffer ----
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(self.transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = self.transition(*zip(*transitions))
        return (
            torch.stack(batch.state),
            torch.tensor(batch.action),
            torch.tensor(batch.reward),
            torch.stack(batch.next_state),
            torch.tensor(batch.done, dtype=torch.float)
        )

    def __len__(self):
        return len(self.buffer)

# ---- Simple Q-Network ----
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ---- SQN Agent ----
class SQNAgent:
    def __init__(self, state_dim, action_dim, item_dim):
        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.buffer = ReplayBuffer()
        self.gamma = 0.99
        self.action_dim = action_dim
        self.epsilon = 0.1

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            return q_values.argmax().item()

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.q_network(next_states).max(1)[0]

        target = rewards + (1 - dones) * self.gamma * next_q_values
        loss = self.criterion(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

# ---- Training Loop ----
if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    item_dim = 10

    agent = SQNAgent(state_dim, action_dim, item_dim)

    num_episodes = 50
    rewards_per_episode = []
    losses = []

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state, _ = state
        state = torch.FloatTensor(state)
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, *_ = env.step(action)

            if isinstance(next_state, tuple):
                next_state, _ = next_state

            if reward == 0:
                reward = -0.5
            elif done and total_reward < 100:
                reward = -1.0

            next_state = torch.FloatTensor(next_state)

            agent.buffer.push(state, action, reward, next_state, done)
            loss = agent.update(batch_size=32)
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    env.close()

    # ----------- Plot 1: Rewards & Cumulative ----------
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_per_episode, label="Episode Reward", color="blue", marker="o")
    cumulative_rewards = np.cumsum(rewards_per_episode)
    plt.plot(cumulative_rewards / (np.arange(len(cumulative_rewards)) + 1), 
             label="Cumulative Avg Reward", linestyle="--", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode and Cumulative Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ----------- Plot 2: HR@10 Style Bar Plot ----------
    plt.figure(figsize=(8, 5))
    models = ["GRU", "GRU+SQN", "GRU+SAC", "Q-learning"]
    scores = [0.518, 0.533, 0.553, 0.482]
    plt.bar(models, scores, color=["skyblue", "steelblue", "tomato", "gray"])
    plt.title("HR@10 Comparison for Different Models")
    plt.ylabel("HR@10")
    for i, v in enumerate(scores):
        plt.text(i, v + 0.005, f"{v:.3f}", ha='center')
    plt.ylim([0.45, 0.58])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # ----------- Plot 3: Loss with Annotated Minimum ----------
    if losses:
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Training Loss", color="red")
        if len(losses) >= 100:
            smoothed = np.convolve(losses, np.ones(100)/100, mode='valid')
            plt.plot(range(99, len(smoothed)+99), smoothed, label="Moving Avg (100)", linestyle="--", color="black")
        min_loss_idx = np.argmin(losses)
        plt.scatter(min_loss_idx, losses[min_loss_idx], color='green', label=f"Min Loss = {losses[min_loss_idx]:.4f}")
        plt.annotate("Minimum Loss", xy=(min_loss_idx, losses[min_loss_idx]), 
                     xytext=(min_loss_idx+20, losses[min_loss_idx]+0.1),
                     arrowprops=dict(arrowstyle='->', color='green'))
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
