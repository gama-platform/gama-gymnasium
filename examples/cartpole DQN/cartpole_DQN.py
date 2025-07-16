"""
Deep Q-Network (DQN) Training Example with GAMA-Gymnasium

This script demonstrates how to train a Deep Q-Network agent on a CartPole environment
implemented in GAMA and accessed through the Gymnasium interface. The implementation
includes experience replay, target networks, and epsilon-greedy exploration.

Key components:
- FullyConnectedModel: Neural network for Q-value estimation
- QNetwork: Wrapper for the neural network with optimization
- ReplayMemory: Experience replay buffer for stable learning
- DQN_Agent: Main agent implementing the DQN algorithm

The script trains multiple agents with different random seeds and evaluates their
performance, providing insights into learning stability and final performance.

Requirements:
- GAMA server running on localhost:1001 (for training) and localhost:1000 (for testing)
- PyTorch, Gymnasium, and other dependencies
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import asyncio
import os
from os.path import exists

import collections
from collections import namedtuple, deque
import tqdm
import matplotlib.pyplot as plt
import random
import gymnasium as gym
import gama_gymnasium

# Commented out seed functionality for deterministic behavior
# SEED = 1234

# def set_seed(seed: int = 0):
#     """Set random seeds for reproducibility across different libraries."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#     if torch.backends.mps.is_available():
#         torch.backends.mps.manual_seed(seed)
#     if torch.backends.cudnn.is_available():
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

class FullyConnectedModel(nn.Module):
    """
    Neural Network for Q-value approximation in DQN.
    
    This network takes the CartPole state (4 dimensions: position, velocity, 
    pole angle, pole angular velocity) and outputs Q-values for each action
    (2 actions: left, right).
    
    Architecture:
    - Input layer: 4 neurons (state dimensions)
    - Hidden layers: 3 layers of 32 neurons each with ReLU activation
    - Output layer: 2 neurons (Q-values for each action)
    """
    def __init__(self, input_size, output_size):
        super(FullyConnectedModel, self).__init__()

        # Define hidden layers with ReLU activation
        # Using 3 hidden layers of 32 neurons each for adequate representation capacity
        self.linear1 = nn.Linear(input_size, 32)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(32, 32)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(32, 32)
        self.activation3 = nn.ReLU()

        # Output layer without activation function (raw Q-values)
        self.output_layer = nn.Linear(32, output_size)

        # Weight initialization using Xavier uniform for better convergence
        # This helps prevent vanishing/exploding gradients in deep networks
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, inputs):
        """
        Forward pass through the network.
        
        Args:
            inputs: State tensor of shape (batch_size, 4)
            
        Returns:
            Q-values tensor of shape (batch_size, 2)
        """
        # Forward pass through the layers with ReLU activations
        x = self.activation1(self.linear1(inputs))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        # No activation on output layer - raw Q-values
        x = self.output_layer(x)
        return x
    
class QNetwork:
    """
    Wrapper class for the Q-Network with training utilities.
    
    This class encapsulates the neural network model along with its optimizer
    and provides methods for loading pre-trained models. It serves as an
    interface between the DQN agent and the underlying neural network.
    """
    def __init__(self, env, lr, logdir=None):
        # Initialize Q-network with CartPole-specific architecture (4 inputs, 2 outputs)
        self.net = FullyConnectedModel(4, 2)
        self.env = env
        self.lr = lr 
        self.logdir = logdir
        # Adam optimizer for stable and efficient training
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def load_model(self, model_file):
        """Load complete model state from a file."""
        return self.net.load_state_dict(torch.load(model_file))

    def load_model_weights(self, weight_file):
        """Load only model weights from a file."""
        return self.net.load_state_dict(torch.load(weight_file))
    

class ReplayMemory:
    """
    Experience Replay Buffer for DQN training.
    
    Experience replay is a key component of DQN that stores transitions
    (state, action, next_state, reward) and allows the agent to learn
    from past experiences by sampling random mini-batches. This breaks
    the correlation between consecutive samples and stabilizes training.
    """
    def __init__(self, env, memory_size=50000, burn_in=1000):
        """
        Initialize the replay memory buffer.
        
        Args:
            env: The environment instance
            memory_size: Maximum number of transitions to store
            burn_in: Number of random transitions to collect before training
        """
        self.memory_size = memory_size
        self.burn_in = burn_in
        # Use deque for efficient append/pop operations with max length
        self.memory = collections.deque([], maxlen=memory_size)
        self.env = env

    def sample_batch(self, batch_size=32):
        """
        Sample a random batch of transitions for training.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of randomly sampled Transition objects
        """
        return random.sample(self.memory, batch_size)

    def append(self, transition):
        """
        Add a new transition to the replay memory.
        
        Args:
            transition: A Transition namedtuple containing (state, action, next_state, reward)
        """
        self.memory.append(transition)


# Named tuple for storing experience transitions
# This provides a clean interface for storing and accessing transition components
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN_Agent:
    """
    Deep Q-Network Agent implementing the DQN algorithm.
    
    This agent combines several key techniques:
    1. Deep Q-Networks for function approximation
    2. Experience replay for stable learning
    3. Target networks for stability
    4. Epsilon-greedy exploration strategy
    
    The agent interacts with a GAMA-based CartPole environment and learns
    to balance the pole through reinforcement learning.
    """

    def __init__(self, environment_name, exp_path, exp_name, lr=5e-4, render=False):
        """
        Initialize the DQN Agent with environment and hyperparameters.
        
        Args:
            environment_name: Name of the Gymnasium environment
            exp_path: Path to the GAMA experiment file
            exp_name: Name of the experiment in GAMA
            lr: Learning rate for the neural network
            render: Whether to render the environment (not used in headless GAMA)
        """
        # Create the GAMA-Gymnasium environment
        self.env = gym.make(environment_name, 
                            gaml_experiment_path=exp_path, 
                            gaml_experiment_name=exp_name, 
                            gama_ip_address="localhost",
                            gama_port=1001)  # Different port for training
        
        self.lr = lr
        
        # Initialize policy network (main Q-network for action selection)
        self.policy_net = QNetwork(self.env, self.lr)
        
        # Initialize target network (stable Q-network for target computation)
        # Target network is updated less frequently to provide stable targets
        self.target_net = QNetwork(self.env, self.lr)
        self.target_net.net.load_state_dict(self.policy_net.net.state_dict())
        
        # Initialize experience replay memory
        self.rm = ReplayMemory(self.env)
        
        # DQN hyperparameters
        self.batch_size = 64        # Mini-batch size for training
        self.gamma = 0.99          # Discount factor for future rewards
        self.c = 0                 # Counter for target network updates
        
        # Performance monitoring
        self.step_times = []       # Track step execution times
        self.reset_times = []      # Track environment reset times

    def burn_in_memory(self):
        """
        Initialize replay memory with random experiences.
        
        Before training begins, we fill the replay buffer with random experiences
        to ensure we have enough diverse data for mini-batch sampling. This
        prevents training on highly correlated initial experiences.
        """
        cnt = 0
        terminated = False
        truncated = False
        
        # Reset environment and get initial state
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Collect random experiences until we reach burn_in threshold
        while cnt < self.rm.burn_in:
            # Reset environment if episode ended
            if terminated or truncated:
                state, _ = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            # Take random action (uniform exploration for diverse experiences)
            action = torch.tensor(random.sample([0, 1], 1)[0]).reshape(1, 1)
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward])
            
            # Handle terminal states (next_state = None for terminal transitions)
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                
            # Store transition in replay memory
            transition = Transition(state, action, next_state, reward)
            self.rm.memory.append(transition)
            state = next_state
            cnt += 1

    def epsilon_greedy_policy(self, q_values, epsilon=0.05):
        """
        Implement epsilon-greedy action selection.
        
        With probability epsilon, select a random action (exploration).
        Otherwise, select the action with highest Q-value (exploitation).
        This balances exploration and exploitation during training.
        
        Args:
            q_values: Q-values for all actions from the neural network
            epsilon: Probability of selecting random action
            
        Returns:
            Selected action as a tensor
        """
        if random.random() > epsilon:
            # Exploitation: choose action with highest Q-value
            with torch.no_grad():
                return self.greedy_policy(q_values)
        else:
            # Exploration: choose random action
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)

    def greedy_policy(self, q_values):
        """
        Implement greedy action selection for testing/evaluation.
        
        Always select the action with the highest Q-value.
        Used during evaluation when we want the agent's best performance.
        
        Args:
            q_values: Q-values for all actions
            
        Returns:
            Action with highest Q-value
        """
        return torch.argmax(q_values)
        
    def train(self):
        """
        Train the Q-network using the DQN algorithm for one episode.
        
        This method implements the core DQN training loop:
        1. Interact with environment using epsilon-greedy policy
        2. Store experiences in replay memory
        3. Sample mini-batches and update the Q-network
        4. Periodically update the target network
        
        The training continues until the episode terminates.
        """
        # Reset environment and measure reset time
        start_reset = time.perf_counter()
        state, _ = self.env.reset()
        end_reset = time.perf_counter()
        self.reset_times.append(end_reset - start_reset)
        
        # Convert state to tensor format
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        terminated = False
        truncated = False

        # Main training loop for one episode
        while not (terminated or truncated):
            # Get Q-values for current state (no gradient computation for action selection)
            with torch.no_grad():
                q_values = self.policy_net.net(state)

            # Select action using epsilon-greedy strategy
            action = self.epsilon_greedy_policy(q_values).reshape(1, 1)
            
            # Execute action in environment and measure step time
            start_step = time.perf_counter()
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            end_step = time.perf_counter()
            self.step_times.append(end_step - start_step)
            
            # Convert reward to tensor
            reward = torch.tensor([reward])
            
            # Handle terminal states (set next_state to None)
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Store experience transition in replay memory
            transition = Transition(state, action, next_state, reward)
            self.rm.memory.append(transition)

            # Move to next state
            state = next_state

            # Sample mini-batch from replay memory for Q-network update
            transitions = self.rm.sample_batch(self.batch_size)
            batch = Transition(*zip(*transitions))
            
            # Create mask for non-terminal states
            non_final_mask = torch.tensor(
                tuple(map(lambda s: s is not None, batch.next_state)), 
                dtype=torch.bool
            )
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            
            # Convert batch components to tensors
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s,a) for the taken actions
            values = self.policy_net.net(state_batch)
            state_action_values = values.gather(1, action_batch)
            
            # Compute max Q'(s',a') for next states using target network
            next_state_values = torch.zeros(self.batch_size)
            with torch.no_grad():
                # Only update non-terminal states (terminal states have value 0)
                next_state_values[non_final_mask] = self.target_net.net(non_final_next_states).max(1)[0]
                    
            # Compute target values: r + γ max Q'(s',a')
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            
            # Compute loss and perform gradient descent
            criterion = torch.nn.MSELoss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            
            # Backpropagation
            self.policy_net.optimizer.zero_grad()
            loss.backward()
            self.policy_net.optimizer.step()

            # Update target network periodically for stability
            self.c += 1
            if self.c % 50 == 0:
                # Copy policy network weights to target network
                sdict = self.policy_net.net.state_dict()
                self.target_net.net.load_state_dict(sdict)

    def test(self, model_file=None):
        """
        Evaluate the agent's performance over a single episode.
        
        This method runs the agent using its learned policy (greedy action selection)
        and returns the total reward achieved. Used for evaluating training progress.
        
        Args:
            model_file: Optional path to save the model after testing
            
        Returns:
            Total reward achieved in the test episode
        """
        max_t = 1000  # Maximum steps per episode
        state, _ = self.env.reset()
        rewards = []
        actions = [0, 0]  # Counter for left (0) and right (1) actions

        # Run episode with learned policy
        for t in range(max_t):
            # Convert state to tensor and get Q-values
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net.net(state)
            
            # Debug information (can be commented out for faster execution)
            print(f"Test step {t+1}/{max_t}, state: {state.numpy()}, q_values: {q_values.numpy()}")
            
            # Select action greedily (best action according to Q-values)
            action = self.greedy_policy(q_values)
            actions[action] += 1
            
            # Execute action and observe results
            state, reward, terminated, truncated, _ = self.env.step(action.item())
            rewards.append(reward)
            
            # Break if episode ends
            if terminated or truncated:
                break

        # Save model if requested
        if model_file:
            torch.save(self.policy_net.net.state_dict(), model_file)
        
        # Display action distribution for analysis
        print(f"Actions taken: {actions[0]} left, {actions[1]} right")
        return np.sum(rewards)


async def main():
    """
    Main training function that orchestrates the complete DQN training process.
    
    This function:
    1. Sets up training parameters and environment configuration
    2. Trains multiple agents with different random seeds for robustness
    3. Evaluates agent performance during training
    4. Saves trained models and displays results
    5. Demonstrates the trained agent in action
    
    The training process includes:
    - Experience replay buffer initialization (burn-in)
    - Episodic training with periodic evaluation
    - Performance monitoring and visualization
    - Model persistence for later use
    """
    model_score = {}  # Dictionary to store performance results for each agent

    # Training configuration parameters
    env_name = 'gama_gymnasium_env/GamaEnv-v0'  # GAMA-Gymnasium environment
    exp_path = str(Path(__file__).parents[0] / "cartpole_env.gaml")  # GAMA model file
    exp_name = "gym_env"  # Experiment name in GAMA model
    
    # Hyperparameters
    num_episodes_train = 200  # Total training episodes per agent
    num_episodes_test = 20    # Episodes for evaluation during training
    learning_rate = 5e-4      # Neural network learning rate

    # Multi-seed training for robust evaluation
    num_seeds = 2             # Number of different agents to train
    l = num_episodes_train // 10  # Evaluation frequency (every 10 episodes)
    res = np.zeros((num_seeds, l))  # Results matrix for all agents
    gamma = 0.99              # Discount factor (defined but not used here)

    print("Starting DQN training with GAMA-Gymnasium CartPole environment...")
    print(f"Training {num_seeds} agents for {num_episodes_train} episodes each...")

    # Train multiple agents with different random initializations
    for i in tqdm.tqdm(range(num_seeds), desc="Training agents"):
        reward_means = []

        print(f"\n--- Training Agent {i+1}/{num_seeds} ---")
        
        # Create and initialize DQN agent
        agent = DQN_Agent(env_name, exp_path, exp_name, lr=learning_rate)
        
        # Initialize replay memory with random experiences
        start_burn_in = time.perf_counter()
        agent.burn_in_memory()
        end_burn_in = time.perf_counter()
        print(f"Burn-in memory completed in {end_burn_in - start_burn_in:.2f} seconds.")

        # Main training loop
        for m in range(num_episodes_train):
            # Train for one episode
            agent.train()

            # Evaluate agent performance every 10 episodes
            if m % 10 == 0:
                print(f"Episode: {m}")

                # Run multiple test episodes for robust evaluation
                G = np.zeros(num_episodes_test)
                for k in range(num_episodes_test):
                    g = agent.test()  # Get total reward for one test episode
                    G[k] = g

                # Calculate performance statistics
                reward_mean = G.mean()
                reward_sd = G.std()
                print(f"The test reward for episode {m} is {reward_mean} with a standard deviation of {reward_sd}.")
                reward_means.append(reward_mean)

                # Monitor environment interaction performance
                reset_mean = np.mean(agent.reset_times)
                step_mean = np.mean(agent.step_times)
                print(f"Average reset time: {reset_mean:.4f} seconds, Average step time: {step_mean:.4f} seconds")
                
                # Clear timing data for next evaluation period
                agent.reset_times = []
                agent.step_times = []

        # Store results for this agent
        res[i] = np.array(reward_means)

        # Save the trained model
        models_dir = Path(__file__).parents[0] / "models_saved"
        if not models_dir.exists():
            os.makedirs(str(models_dir))

        save_path = str(models_dir / f"cartpole_dqn{i}.pth")
        torch.save(agent.policy_net.net.state_dict(), save_path)
        model_score[f"cartpole_dqn{i}"] = res[i]
        print(f"Model saved to: {save_path}")

    print("\n--- Training Complete ---")
    print("Model scores:", model_score)

    # Visualize training results
    print("\nGenerating performance visualization...")
    ks = np.arange(l) * 10  # Episode numbers for x-axis
    avs = np.mean(res, axis=0)  # Average performance across agents
    maxs = np.max(res, axis=0)  # Best performance
    mins = np.min(res, axis=0)  # Worst performance

    # Create performance plot
    plt.figure(figsize=(10, 6))
    plt.fill_between(ks, mins, maxs, alpha=0.1, label='Min-Max Range')
    plt.plot(ks, avs, '-o', markersize=1, label='Average Performance')
    plt.xlabel('Episode', fontsize=15)
    plt.ylabel('Avg. Return', fontsize=15)
    plt.title('DQN Training Performance on GAMA CartPole', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Demonstrate trained agent performance
    print("\n--- Demonstrating Trained Agent ---")
    await demonstrate_trained_agent(env_name, exp_path, exp_name)


async def demonstrate_trained_agent(env_name, exp_path, exp_name):
    """
    Demonstrate the performance of a trained agent.
    
    This function loads a saved model and runs it in the environment
    to showcase the learned behavior. Used for visual validation
    of training results.
    
    Args:
        env_name: Name of the Gymnasium environment
        exp_path: Path to GAMA experiment file
        exp_name: Name of GAMA experiment
    """
    print("Creating environment for demonstration...")
    
    # Create CartPole environment for demonstration (different port from training)
    env = gym.make(env_name,
                   gaml_experiment_path=exp_path,
                   gaml_experiment_name=exp_name,
                   gama_ip_address="localhost",
                   gama_port=1000)  # Different port for demo
    
    # Load the most recent trained agent
    try:
        model_file = str(Path(__file__).parents[0] / "models_saved" / "cartpole_dqn0.pth")
        model = load_nn_model(model_file)
        print(f"Loaded trained model from: {model_file}")
    except FileNotFoundError:
        print("No trained model found. Please run training first.")
        env.close()
        return
    
    # Reset environment and prepare for demonstration
    state, _ = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    print("Running demonstration episode...")
    time.sleep(5)  # Allow environment initialization

    # Run demonstration episode
    max_steps = 2000  # Maximum steps for demonstration
    while not done and step_count < max_steps:
        # Convert state to tensor and get Q-values
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
        
        # Select best action (greedy policy)
        action = torch.argmax(q_values).item()
        
        # Execute action
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Check for episode termination
        if terminated or truncated:
            done = True
            print(f"Episode ended after {step_count} steps with total reward: {total_reward}")

    # Clean up
    env.close()
    print("Demonstration complete!")


def load_nn_model(model_file):
    """
    Load a pre-trained neural network model from file.
    
    This utility function creates a new FullyConnectedModel instance
    and loads the saved state dictionary. Used for model evaluation
    and demonstration after training.
    
    Args:
        model_file: Path to the saved model file (.pth)
        
    Returns:
        Loaded neural network model ready for inference
    """
    model = FullyConnectedModel(4, 2)  # CartPole-specific architecture
    model.load_state_dict(torch.load(model_file))
    model.eval()  # Set to evaluation mode
    return model

async def test_train_cartpole():
    """
    Alternative testing function for evaluating a pre-trained model.
    
    This function demonstrates how to load and test a previously trained
    DQN model without going through the full training process. Useful
    for quick evaluation of saved models.
    
    Note: This function is currently not called in the main execution
    but can be used independently for model testing.
    """
    print("--- Testing Pre-trained CartPole Model ---")
    
    # Environment configuration
    env_name = 'gama_gymnasium_env/GamaEnv-v0'
    exp_path = str(Path(__file__).parents[0] / "cartpole_env.gaml")
    exp_name = "gym_env"

    # Load the pre-trained model
    model_file = str(Path(__file__).parents[0] / "models_saved" / "cartpole_dqn0.pth")
    try:
        model = load_nn_model(model_file)
        print(f"Successfully loaded model from: {model_file}")
    except FileNotFoundError:
        print(f"Model file not found: {model_file}")
        print("Please run training first to generate a saved model.")
        return

    # Create a dummy agent for testing (reuses existing agent structure)
    agent = DQN_Agent(env_name, exp_path, exp_name)
    agent.policy_net.net = model  # Replace with loaded model

    # Create test environment
    env = gym.make(env_name,
                   gaml_experiment_path=exp_path,
                   gaml_experiment_name=exp_name,
                   gama_ip_address="localhost",
                   gama_port=1000)  # Test environment port
    
    # Reset and prepare for testing
    state, _ = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    print("Starting extended test run...")
    time.sleep(12)  # Extended initialization time

    # Extended test run (up to 2000 steps)
    max_steps = 2000
    while not done and step_count < max_steps:
        # Get action from trained policy
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            q_values = agent.policy_net.net(state_tensor)
        action = agent.greedy_policy(q_values).detach().numpy()
        
        # Execute action and update state
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Check termination conditions
        if terminated or truncated:
            done = True
            print(f"Test episode completed after {step_count} steps")
            print(f"Total reward achieved: {total_reward}")

    # Clean up resources
    env.close()
    print("Test completed successfully!")


if __name__ == "__main__":
    """
    Main execution block.
    
    This script can be run in two modes:
    1. Full training mode (default): Trains new DQN agents from scratch
    2. Testing mode (commented): Tests a pre-trained model
    
    Before running, ensure GAMA server is running:
    - Training: localhost:1001
    - Testing/Demo: localhost:1000
    
    Commands to start GAMA server:
    ./gama-headless.sh -socket <port>  # Linux/MacOS
    gama-headless.bat -socket <port>   # Windows
    """
    print("=" * 60)
    print("DQN Training with GAMA-Gymnasium CartPole Environment")
    print("=" * 60)
    print("\nMake sure GAMA servers are running:")
    print("- Training server: localhost:1001")
    print("- Demo server: localhost:1000")
    print("\nStarting training process...\n")
    
    # Run main training process
    asyncio.run(main())
    
    # Alternative: Run only the testing function (uncomment to use)
    # asyncio.run(test_train_cartpole())