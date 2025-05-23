import gym
import torch
from sqn_ssl import SQNAgent

# Main Training
if __name__ == "__main__":
    env = gym.make("CartPole-v1")  # Create the environment

    state_dim = env.observation_space.shape[0]  # CartPole has 4 states
    action_dim = env.action_space.n  # 2 possible actions (left or right)
    item_dim = 10  # arbitrary, you can keep 10 as earlier

    agent = SQNAgent(state_dim, action_dim, item_dim)

    num_episodes = 50

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # gymnasium returns (obs, info)
            state, _ = state

        state = torch.FloatTensor(state)
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, *_ = env.step(action)

            if isinstance(next_state, tuple):
                next_state, _ = next_state

            next_state = torch.FloatTensor(next_state)

            agent.buffer.push(state, action, reward, next_state, done)
            agent.update(batch_size=32)

            state = next_state
            total_reward += reward

        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")

    env.close()
