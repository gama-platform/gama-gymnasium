import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path
import asyncio
import gama_gymnasium

# Hyperparamètres
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500
TARGET_UPDATE = 10

# Réseau de neurones pour approximer Q(s, a)
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Fonction pour choisir une action (epsilon-greedy)
def select_action(state, policy_net, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    with torch.no_grad():
        state = torch.tensor([state], dtype=torch.float32)
        q_values = policy_net(state)
        return q_values.argmax().item()

# Expérience (s, a, r, s', done)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Entraînement
async def train_dqn(env_name="gama_gymnasium_env/GamaEnv-v0", exp_path=str(Path(__file__).parents[0] / "cartpole_env.gaml"), exp_name="gym_env", episodes=100):
    env = gym.make(env_name, 
                    gaml_experiment_path=exp_path, 
                    gaml_experiment_name=exp_name, 
                    gama_ip_address="localhost",
                    gama_port=1001)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    rewards_history = []
    epsilon = EPS_START
    steps_done = 0

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(500):
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            action = select_action(state, policy_net, epsilon, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            steps_done += 1

            if len(replay_buffer) >= BATCH_SIZE:
                transitions = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*transitions)

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

                q_values = policy_net(states).gather(1, actions)
                next_q_values = target_net(next_states).max(1)[0].detach().unsqueeze(1)
                expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

                loss = nn.MSELoss()(q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # Mise à jour du réseau cible
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        rewards_history.append(total_reward)
        print(f"Episode {episode} - Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")

    env.close()
    plot_rewards(rewards_history)

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training Progress on CartPole")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("dqn_cartpole_rewards.png")
    plt.show()

# Lancer l'entraînement
if __name__ == "__main__":
    asyncio.run(train_dqn())
