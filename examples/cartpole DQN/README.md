# CartPole DQN Example

This directory contains a complete implementation of Deep Q-Network (DQN) reinforcement learning applied to a CartPole environment using GAMA-Gymnasium integration.

## 📁 Files Overview

| File                      | Description                                                      |
| ------------------------- | ---------------------------------------------------------------- |
| `cartpole_env.gaml`     | GAMA simulation model implementing the CartPole environment      |
| `cartpole_DQN.py`       | Complete DQN training implementation with neural networks        |
| `cartpole_gymnasium.py` | Simple demonstration of GAMA-Gymnasium interaction               |
| `test_cartpole.py`      | Basic testing script for environment validation                  |
| `models_saved/`         | Directory containing trained DQN models (created after training) |

## 🎯 CartPole Problem

The CartPole problem is a classic control task in reinforcement learning:

- **Goal**: Balance a pole on a moving cart for as long as possible
- **State**: 4-dimensional vector (cart position, cart velocity, pole angle, pole angular velocity)
- **Actions**: 2 discrete actions (move cart left or right)
- **Reward**: +1 for each timestep the pole remains upright
- **Episode End**: Pole falls beyond ±15° or cart moves beyond ±2.4 units

## 🧠 Deep Q-Network (DQN) Implementation

### Algorithm Features

The `cartpole_DQN.py` implementation includes:

- **Deep Q-Networks**: Neural network function approximation for Q-values
- **Experience Replay**: Buffer storing past experiences for stable learning
- **Target Networks**: Separate network for stable target computation
- **Epsilon-Greedy**: Exploration strategy balancing exploration vs exploitation

### Network Architecture

```text
Input Layer:    4 neurons (state dimensions)
Hidden Layer 1: 32 neurons + ReLU
Hidden Layer 2: 32 neurons + ReLU  
Hidden Layer 3: 32 neurons + ReLU
Output Layer:   2 neurons (Q-values for each action)
```

### Key Components

#### 1. FullyConnectedModel

Neural network class implementing the Q-function approximator with Xavier weight initialization.

#### 2. ReplayMemory

Experience replay buffer storing transitions (state, action, next_state, reward) for mini-batch sampling.

#### 3. DQN_Agent

Main agent class implementing:

- Training loop with epsilon-greedy policy
- Q-network updates using MSE loss
- Target network periodic updates (every 50 steps)
- Performance evaluation

## 🚀 Getting Started

### Prerequisites

1. **GAMA Platform**: Install from [gama-platform.org](https://gama-platform.org/download)
2. **Python Dependencies**:

   ```bash
   pip install torch gymnasium gama-gymnasium numpy matplotlib tqdm
   ```

### Running the Examples

#### 1. Start GAMA Servers and GUI

Start a GAMA server instances:

```bash
# Training server (port 1001)
./gama-headless.sh -socket 1001
```

Lunch GAMA application with the default port 1000.

#### 2. Basic Environment Test

Test the GAMA-Gymnasium connection:

```bash
python cartpole_gymnasium.py
```

This script demonstrates:

- Environment creation and configuration
- Basic interaction loop (reset, step, observation)
- Performance monitoring (step timing)

#### 3. Full DQN Training

Train DQN agents from scratch:

```bash
python cartpole_DQN.py
```

This will:

- Train 2 agents for 200 episodes each
- Evaluate performance every 10 episodes
- Save trained models to `models_saved/`
- Display training progress visualization
- Demonstrate trained agent performance

#### 4. Test Pre-trained Models

Evaluate saved models without training:

```python
# Uncomment the last line in cartpole_DQN.py
asyncio.run(test_train_cartpole())
```

## 📊 Training Process

### Hyperparameters

| Parameter     | Value    | Description                         |
| ------------- | -------- | ----------------------------------- |
| Learning Rate | 5e-4     | Neural network optimization rate    |
| Batch Size    | 64       | Mini-batch size for training        |
| Gamma         | 0.99     | Discount factor for future rewards  |
| Epsilon       | 0.05     | Exploration probability             |
| Memory Size   | 50,000   | Experience replay buffer capacity   |
| Burn-in       | 1,000    | Initial random experiences          |
| Target Update | 50 steps | Frequency of target network updates |

### Training Schedule

1. **Burn-in Phase**: Collect 1,000 random experiences
2. **Training Episodes**: 200 episodes per agent
3. **Evaluation**: Every 10 episodes (20 test episodes each)
4. **Model Saving**: After each agent completes training

### Performance Metrics

- **Episode Reward**: Total reward achieved per episode
- **Success Rate**: Percentage of episodes reaching maximum reward
- **Step Timing**: Environment interaction performance
- **Learning Stability**: Variance across multiple training runs

## 📈 Expected Results

A well-trained DQN agent should achieve:

- **Average Reward**: 450-500+ points per episode
- **Episode Length**: Close to maximum (500 steps for standard CartPole)
- **Success Rate**: >95% of episodes reaching maximum reward
- **Learning Time**: Convergence within 100-150 episodes

## 🔧 Customization

### Modifying the Neural Network

Edit the `FullyConnectedModel` class to change:

- Number of hidden layers
- Layer sizes
- Activation functions
- Weight initialization methods

### Adjusting Hyperparameters

Modify variables in the `main()` function:

- `num_episodes_train`: Training duration
- `learning_rate`: Optimization speed
- `num_seeds`: Number of training runs
- Batch size, gamma, epsilon in `DQN_Agent.__init__()`

### Environment Modifications

Edit `cartpole_env.gaml` to customize:

- Pole length and mass
- Cart mass and friction
- Episode termination conditions
- Reward structure

## 🐛 Troubleshooting

### Common Issues

1. **GAMA Connection Errors**

   - Ensure GAMA servers are running on correct ports
   - Check firewall settings
   - Verify GAMA model file path
2. **Training Instability**

   - Reduce learning rate
   - Increase replay buffer size
   - Adjust target network update frequency
3. **Poor Performance**

   - Increase training episodes
   - Tune exploration (epsilon) parameter
   - Check reward function implementation

### Debug Mode

Enable detailed logging by uncommenting debug print statements in:

- `test()` method: Q-values and state information
- `train()` method: Action and reward tracking

## 📚 Further Reading

- [DQN Paper](https://arxiv.org/abs/1312.5602): Original Deep Q-Network publication
- [GAMA Documentation](https://gama-platform.org/wiki/Home): GAMA platform guide
- [Gymnasium Documentation](https://gymnasium.farama.org/): Environment interface standard
- [PyTorch Tutorials](https://pytorch.org/tutorials/): Deep learning framework

## 🤝 Contributing

To contribute improvements:

1. Test modifications with multiple random seeds
2. Maintain backward compatibility with GAMA models
3. Update documentation and comments
4. Add performance benchmarks for validation

## 🔗 Related Documentation

- **[🏠 Main Project](../../README.md)**: Overall GAMA-Gymnasium documentation and setup
- **[🎯 Basic Example](../basic_example/README.md)**: Start here if you're new to GAMA-Gymnasium
- **[🧪 Testing Guide](../../tests/README.md)**: Validate your setup and run comprehensive tests
- **[📦 Source Code](../../src/README.md)**: Technical package documentation
- **[🧊 Frozen Lake QLearning](../frozen%20lake%20QLearning/)**: Another RL example

---

This example demonstrates the power of combining agent-based simulation (GAMA) with modern deep reinforcement learning techniques, providing a foundation for more complex multi-agent RL scenarios.
