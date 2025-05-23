import gym
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from sqn_ssl import SQNAgent

# Main Training
if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    item_dim = 10

    agent = SQNAgent(state_dim, action_dim, item_dim)

    num_episodes = 50
    rewards_per_episode = []
    episode_lengths = []
    losses = []

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state, _ = state

        state = torch.FloatTensor(state)
        done = False
        total_reward = 0
        length = 0

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
            length += 1

        rewards_per_episode.append(total_reward)
        episode_lengths.append(length)

        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, Length = {length}")

    env.close()

    # ---------- Plot 1: Episode Rewards (Matplotlib with moving avg) ----------
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_per_episode, label="Episode Reward")
    if len(rewards_per_episode) >= 5:
        moving_avg = np.convolve(rewards_per_episode, np.ones(5)/5, mode='valid')
        plt.fill_between(range(4, len(moving_avg)+4), moving_avg, alpha=0.3, label="Moving Average (5)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("SQN Agent Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------- Plot 2: Episode Lengths (Using Plotly) ----------
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=episode_lengths,
        mode='lines+markers',
        name='Episode Lengths',
        line=dict(color='green')
    ))
    fig.update_layout(
        title='Episode Length Over Time (Interactive)',
        xaxis_title='Episode',
        yaxis_title='Length',
        template='plotly_white'
    )
    fig.show()

    # ---------- Plot 3: Summary Stats Bar Chart (Using Seaborn) ----------
    avg_reward = np.mean(rewards_per_episode)
    max_reward = np.max(rewards_per_episode)
    avg_length = np.mean(episode_lengths)
    max_length = np.max(episode_lengths)
    avg_loss = np.mean(losses) if losses else 0.0

    stats = {
        "Avg Reward": avg_reward,
        "Max Reward": max_reward,
        "Avg Length": avg_length,
        "Max Length": max_length,
        "Avg Loss": avg_loss
    }

    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(stats.keys()), y=list(stats.values()), palette="muted")
    plt.title("Training Summary Statistics")
    plt.ylabel("Values")
    plt.xticks(rotation=30)
    for i, val in enumerate(stats.values()):
        plt.text(i, val + 0.5, f'{val:.2f}', ha='center')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
