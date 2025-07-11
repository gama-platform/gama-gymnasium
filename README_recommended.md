# GAMA-Gymnasium Integration

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/gama-gymnasium.svg)](https://badge.fury.io/py/gama-gymnasium)

A seamless integration between [GAMA Platform](https://gama-platform.org/) simulations and [OpenAI Gymnasium](https://gymnasium.farama.org/) for reinforcement learning research.

## 🎯 Overview

**GAMA-Gymnasium** bridges the gap between agent-based modeling and reinforcement learning by enabling GAMA simulations to be used as Gymnasium environments. This allows researchers to:

- Train RL agents in realistic, complex environments modeled in GAMA
- Leverage GAMA's powerful spatial and social modeling capabilities
- Use standard RL libraries (Stable-Baselines3, Ray RLlib, etc.) with GAMA
- Combine multi-agent systems with reinforcement learning

## 🚀 Key Features

- **🔌 Plug-and-Play Integration**: Convert any GAMA simulation into a Gymnasium environment
- **📡 Real-time Communication**: Efficient socket-based communication between GAMA and Python
- **🎛️ Flexible Spaces**: Support for discrete, continuous, and mixed action/observation spaces
- **🔧 Type Safety**: Full type hints and modern Python practices
- **📊 Multiple Examples**: From basic grid worlds to complex scenarios
- **🏗️ Extensible Architecture**: Easy to customize for specific use cases

## 📦 Installation

### From PyPI (Recommended)
```bash
pip install gama-gymnasium
```

### From Source
```bash
git clone https://github.com/gama-platform/gama-gymnasium.git
cd gama-gymnasium
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/gama-platform/gama-gymnasium.git
cd gama-gymnasium
pip install -e ".[dev,docs,examples]"
pre-commit install
```

## 🏃‍♂️ Quick Start

### 1. Basic Usage

```python
import asyncio
from gama_gymnasium import GamaEnv

async def main():
    # Create environment from GAMA model
    env = GamaEnv(
        model_path="path/to/your/model.gaml",
        experiment_name="your_experiment"
    )
    
    # Standard Gymnasium interface
    observation, info = await env.reset()
    
    for _ in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = await env.step(action)
        
        if terminated or truncated:
            obs, info = await env.reset()
    
    await env.close()

# Run the example
asyncio.run(main())
```

### 2. With Stable-Baselines3

```python
from stable_baselines3 import DQN
from gama_gymnasium.wrappers import SyncWrapper

# Wrap for synchronous libraries
env = SyncWrapper(GamaEnv("model.gaml", "experiment"))

# Train your agent
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Test the trained agent
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
```

## 📁 Project Structure

```
gama-gymnasium/
├── src/gama_gymnasium/          # Main package
│   ├── __init__.py             # Public API
│   ├── core/                   # Core functionality
│   │   ├── gama_env.py        # Main environment class
│   │   ├── client.py          # GAMA client communication
│   │   └── message_handler.py # Message processing
│   ├── spaces/                 # Space conversion utilities
│   │   ├── converters.py      # Space type converters
│   │   └── validators.py      # Space validation
│   ├── wrappers/              # Gymnasium wrappers
│   │   ├── sync.py           # Async to sync wrapper
│   │   └── monitoring.py     # Logging and monitoring
│   └── utils/                 # Utility functions
│       ├── logging.py        # Logging configuration
│       └── exceptions.py     # Custom exceptions
├── examples/                   # Usage examples
│   ├── basic/                 # Simple examples
│   ├── rl_training/          # RL training examples
│   └── advanced/             # Complex scenarios
├── tests/                     # Test suite
├── docs/                      # Documentation
├── gama_models/              # GAMA model files
└── pyproject.toml           # Project configuration
```

## 📚 Examples

### Basic Grid World
A simple agent learning to navigate to a target in a grid environment.

- **GAMA Model**: `examples/basic/target_seeking.gaml`
- **Python Script**: `examples/basic/basic_training.py`
- **Tutorial**: [Basic Example Guide](examples/basic/README.md)

### CartPole with DQN
Classic CartPole problem implemented in GAMA with Deep Q-Learning.

- **GAMA Model**: `examples/rl_training/cartpole/cartpole.gaml`
- **Training Script**: `examples/rl_training/cartpole/train_dqn.py`
- **Results**: Pre-trained models available

### Multi-Agent Scenarios
Complex environments with multiple interacting agents.

- **Predator-Prey**: `examples/advanced/predator_prey/`
- **Traffic Simulation**: `examples/advanced/traffic/`
- **Market Simulation**: `examples/advanced/market/`

## 🔧 GAMA Model Requirements

To make your GAMA model compatible with Gymnasium:

### 1. Add the GymAgent

```gaml
global {
    // Your global variables and init
}

// Main agent that communicates with Python
agent GymAgent skills: [GymnasiumLink] {
    // Define observation and action spaces
    map<string, unknown> observation_space <- [
        "type": "Box",
        "low": [0.0, 0.0], 
        "high": [10.0, 10.0],
        "shape": [2]
    ];
    
    map<string, unknown> action_space <- [
        "type": "Discrete",
        "n": 4
    ];
    
    // Implement required methods
    map<string, unknown> get_observation {
        return ["position": [location.x, location.y]];
    }
    
    float get_reward {
        // Calculate and return reward
        return reward_value;
    }
    
    bool is_terminated {
        // Define termination condition
        return game_over;
    }
    
    bool is_truncated {
        // Define truncation condition  
        return time_limit_exceeded;
    }
    
    map<string, unknown> get_info {
        return ["step": cycle, "score": score];
    }
    
    void apply_action(map<string, unknown> action) {
        // Process the action from RL agent
        int chosen_action <- int(action["action"]);
        // Apply action to environment
    }
}
```

### 2. Define Your Environment Agents

```gaml
// Your simulation agents
agent target_seeking_agent {
    point target_location;
    
    reflex move {
        // Agent behavior
        location <- location + {rnd(-1.0, 1.0), rnd(-1.0, 1.0)};
    }
    
    // Visualization
    aspect default {
        draw circle(0.5) color: #blue;
    }
}
```

## 🧪 Testing

Run the complete test suite:

```bash
# Basic tests
pytest

# With coverage
pytest --cov=gama_gymnasium

# Integration tests (requires GAMA)
pytest tests/integration/

# Performance tests
pytest tests/performance/ -v
```

## 📖 Documentation

- **API Reference**: [Full API Documentation](https://gama-gymnasium.readthedocs.io/api/)
- **User Guide**: [Complete User Guide](https://gama-gymnasium.readthedocs.io/guide/)
- **Examples**: [Example Gallery](https://gama-gymnasium.readthedocs.io/examples/)
- **GAMA Integration**: [GAMA Model Development](https://gama-gymnasium.readthedocs.io/gama/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/gama-platform/gama-gymnasium.git
cd gama-gymnasium

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [GAMA Platform](https://gama-platform.org/) for the powerful simulation framework
- [OpenAI Gymnasium](https://gymnasium.farama.org/) for the RL environment standard
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for RL algorithms

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/gama-platform/gama-gymnasium/issues)
- **Discussions**: [GitHub Discussions](https://github.com/gama-platform/gama-gymnasium/discussions)
- **GAMA Community**: [GAMA Platform Community](https://gama-platform.org/community)

---

**Made with ❤️ by the GAMA Platform Team**
