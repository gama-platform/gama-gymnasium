# GAMA-Gymnasium

[![Python Package](https://img.shields.io/pypi/v/gama-gymnasium)](https://pypi.org/project/gama-gymnasium/)
[![License](https://img.shields.io/github/license/gama-platform/gama-gymnasium)](LICENSE)

**GAMA-Gymnasium** is a generic [Gymnasium](https://gymnasium.farama.org/) environment that enables the integration of simulations from the [GAMA](https://gama-platform.org/) modeling platform with reinforcement learning algorithms.

## 🎯 Purpose

This library allows researchers and developers to easily use GAMA models as reinforcement learning environments, leveraging the power of GAMA for agent-based modeling and the Python ecosystem for AI.

## ⚡ Quick Start

### Installation

```bash
pip install gama-gymnasium
```

### Prerequisites

- **GAMA Platform**: Install GAMA from [gama-platform.org](https://gama-platform.org/download)
- **Python 3.8+** with required dependencies

```bash
pip install gama-client gymnasium
```

### Basic Usage

```python
import gama_gymnasium
import gymnasium as gym

# Create the environment
env = gym.make('gama-gymnasium-v0', 
               gama_model_path='your_model.gaml',
               gama_port=6868)

# Use as a standard Gymnasium environment
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### GymAgent

The `GymAgent` is a GAMA agent that is required in the model to enable interaction between the simulation's learning agent and the Gymnasium environment. It has nine variables and one action. The `GymAgent` is a GAMA agent required in the model to allow the interaction between the learning agent of the simulation and Gymnasium environments. It have 9 variables and 1 action

Structure of the agent:

```gaml
species GymAgent{
	map<string, unknown> action_space;
	map<string, unknown> observation_space;

	unknown state;
	float reward;
	bool terminated;
	bool truncated;
	map<string, unknown> info;

	unknown next_action;

	map<string, unknown> data;

	action update_data {
		data <- [
			"State"::state,
			"Reward"::reward,
			"Terminated"::terminated,
			"Truncated"::truncated,
			"Info"::info
		];
	}
}
```

### GAMA Configuration

1. **Add the GAMA component** to your model:
   Make sure you have added the species `GymAgent` described above to your model:

   ```gaml
   species GymAgent;
   ```

   Set up the `action_space` and `observation_space`:

   ```gaml
   global {
       init{
           create GymAgent{
             action_space <- ["type"::"Discrete", "n"::4];
             observation_space <- ["type"::"Box", "low"::0, "high"::grid_size, "shape"::[2], "dtype"::"int"];
           }
       }
   }
   ```

   Update the gym agent's data after the action is completed:

   ```gaml
   ask GymAgent[0] {
       do update_data;
   }
   ```
2. **Launch GAMA in server mode**:

```bash
# Linux/MacOS
./gama-headless.sh -socket 6868

# Windows
gama-headless.bat -socket 6868
```

## 📁 Project Structure

```text
gama-gymnasium/
├── 📁 src/               # Main Python package source code
├── 📁 tests/             # Comprehensive test suite
├── 📁 examples/          # Complete examples and tutorials
├── pyproject.toml	      # Python package configuration
├── LICENSE               # Package license
└── � pytest.ini         # Testing configuration
```

## 📚 Documentation and Examples

### 🚀 Tutorials and Examples

| Example                          | Description                                           | Documentation                                          |
| -------------------------------- | ----------------------------------------------------- | ------------------------------------------------------ |
| **Basic Example**          | Introduction to GAMA-Gymnasium integration            | [📖 README](examples/basic_example/README.md)             |
| **CartPole DQN**           | Deep Q-Network implementation on CartPole environment | [📖 README](examples/cartpole%20DQN/README.md)            |

### 📖 Detailed Guides

- **[Basic Example Guide](examples/basic_example/README.md)**: Complete tutorial for creating your first environment
- **[Direct GAMA Test](examples/basic_example/README_basic_test.md)**: Low-level communication with GAMA
- **[Source Code Documentation](src/README.md)**: Technical documentation of the package structure
- **[Testing Guide](tests/README.md)**: Comprehensive testing framework and best practices

## 🛠 Advanced Installation

### From Source Code

```bash
git clone https://github.com/gama-platform/gama-gymnasium.git
cd gama-gymnasium
pip install -e src/ 
```

## 🧪 Testing and Validation

```bash
# Run tests
python tests/test_manager.py --quick
```

## 🤝 Contributing

Contributions are welcome! Check the [issues](https://github.com/gama-platform/gama-gymnasium/issues) to see how you can help.

## 🔗 Useful Links

- [GAMA Platform](https://gama-platform.org/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [GAMA-Client PyPI](https://pypi.org/project/gama-client/)

---

For more technical details and practical examples, check the documentation in the [`examples/`](examples/) and [`src/`](src/) folders, or explore our comprehensive [testing framework](tests/README.md).
