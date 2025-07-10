# Basic Example: GAMA-Gymnasium Integration

This example demonstrates the integration between GAMA (an agent-based simulation platform) and Gymnasium (a reinforcement learning environment interface). The integration allows reinforcement learning algorithms to interact with GAMA simulations through the standard Gymnasium API.

## Files in this Example

- `basic_env.gaml`: The GAMA simulation model file
- `basic_gymnasium.py`: Python script demonstrating the Gymnasium interface to the GAMA model
- `basic_test.py`: Additional test script that demonstrates direct communication with GAMA ([see dedicated README](README_basic_test.md))
- `README_basic_test.md`: Detailed documentation of the basic_test.py script

## How It Works

This example creates a simple grid-based environment where an agent needs to find a target cell. The environment is defined in GAMA, and the Python code interacts with it through Gymnasium.

### The GAMA Environment (`basic_env.gaml`)

[GAMA](https://gama-platform.org/) is an agent-based modeling and simulation platform that provides a complete modeling and simulation development environment. In this example, GAMA is used to create a simple grid environment that can be controlled by an external reinforcement learning algorithm.

#### Model Structure

The GAMA model (`basic_env.gaml`) defines:

1. **Grid Environment**: A 2D grid (configurable size, default 4×4) where cells are connected to their neighbors.
2. **Target Cell**: One cell is randomly selected as the target (colored red).
3. **Target Seeking Agent**: An agent that can move around the grid in four directions seeking the target cell.
4. **GymAgent**: A special agent with the `GymnasiumLink` skill that handles communication with Python.

#### GAML Code Breakdown

The GAML code is organized as follows:

**Global Section**:

```gaml
global {
    // Communication port for GAMA-Gymnasium connection
    int gama_server_port <- 0;
  
    // Size of the grid environment
    int grid_size <- 4;
  
    // Initialization code...
}
```

This section defines global variables and the initialization process. The `gama_server_port` is particularly important as it's used to establish the connection with the Python environment.

**Initialization**:

```gaml
init {
    // Create a GymAgent to handle the Gymnasium interface
    create GymAgent;
    // Define action space: 4 discrete actions (up, down, right, left)
    GymAgent[0].action_space <- ["type"::"Discrete", "n"::4];
    // Define observation space: 2D coordinates in the grid
    GymAgent[0].observation_space <- ["type"::"Box", "low"::0, "high"::grid_size, "shape"::[2], "dtype"::"int"];
  
    // Create the target seeking agent and set up the environment
    // ...
}
```

The initialization process creates the necessary agents and defines the action and observation spaces, which are crucial for the Gymnasium interface. The action and observation spaces are defined as JSON-compatible structures.

**Main Simulation Cycle**:

```gaml
reflex {
    // Ask the target seeking agent to perform the next action received from the gym
    ask target_seeking_agent {
        do step(int(GymAgent[0].next_action));
    }
    // Update the gym agent's data after the action is completed
    ask GymAgent {
        do update_data;
    }
}
```

This reflex (automatically executed at each simulation step) makes the target seeking agent perform the action received from Python and then updates the state information that will be sent back to Python.

**Agent Species**:

1. **GymAgent**: A special agent with the `GymnasiumLink` skill that handles the communication protocol:

   ```gaml
   species GymAgent skills:[GymnasiumLink];
   ```

2. **Target Seeking Agent**: The agent that moves in the grid seeking the target:

   ```gaml
   species target_seeking_agent {
       GymAgent gym_agent <- GymAgent[0];  // Reference to the gym agent
       my_grid my_cell;  // Current cell position

       // Agent initialization, movement logic, and visual representation
       // ...
   }
   ```

   The `step` action implements the agent's behavior when receiving an action from Python:

   ```gaml
   action step(int action_) {
       // Movement logic based on action
       // Update environment state, reward, termination status
       // ...
   }
   ```

**Grid Structure**:

```gaml
grid my_grid width: grid_size height: grid_size {
    bool target;  // Flag to mark if this cell is the target
  
    // References to neighboring cells for movement
    my_grid cell_up;
    my_grid cell_down;
    my_grid cell_right;
    my_grid cell_left;
  
    // Initialization of cell connections
    // ...
}
```

The grid cells have references to their neighbors, which are used for agent movement. One cell is marked as the target.

**Experiments**:

1. **test_env**: For testing the environment without external connections.
2. **gym_env**: For connecting to the external Gymnasium environment, with parameters for the communication port and grid size.

#### Key Components in GAMA

- **Action Space**: 4 discrete actions (0: up, 1: down, 2: right, 3: left).
- **Observation Space**: The agent's position as 2D coordinates [x, y].
- **Reward**: 1.0 when the agent reaches the target, 0.0 otherwise.
- **Termination**: The episode ends when the agent reaches the target.
- **Communication**: The `GymnasiumLink` skill handles the socket communication with Python.

### The Python Interface (`basic_gymnasium.py`)

The Python script demonstrates how to connect to and interact with the GAMA environment through the Gymnasium API. For a lower-level approach that shows direct communication with GAMA, see the [`basic_test.py`](README_basic_test.md) script.

## Code Explanation

```python
import asyncio
from pathlib import Path

import gymnasium as gym
from gama_gymnasium.gama_env import GamaEnv
```

These are the necessary imports:

- `asyncio`: For asynchronous programming (the communication with GAMA is async)
- `pathlib.Path`: For manipulating file paths in a platform-independent way
- `gymnasium`: The reinforcement learning environment interface
- `gama_gymnasium.gama_env.GamaEnv`: The custom environment that connects to GAMA

```python
async def main():
    """
    Main asynchronous function that creates and runs a GAMA environment through Gymnasium.
    Uses random actions to interact with the environment until the episode ends.
    """
```

The main function is defined as asynchronous because the communication with GAMA happens asynchronously.

```python
    # Set the size of the grid environment
    grid_size = 3
    # Get the absolute path to the GAMA model file
    exp_path = str(Path(__file__).parents[0] / "basic_env.gaml")
    # Name of the experiment defined in the GAML file
    exp_name = "gym_env"
    # Parameters to pass to the GAMA experiment (here we set the grid size)
    exp_parameters = [{"type": "int", "name": "grid_size", "value": grid_size}]
```

This section sets up the parameters for the GAMA simulation:

- `grid_size`: The size of the grid (3×3)
- `exp_path`: The path to the GAML file
- `exp_name`: The name of the experiment in the GAML file to run
- `exp_parameters`: Parameters to be passed to the GAMA experiment (here, just the grid size)

```python
    # Create the GAMA environment using the Gymnasium make function
    # This connects to the GAMA server and initializes the environment
    env = gym.make('gama_gymnasium_env/GamaEnv-v0',
                  gaml_experiment_path=exp_path,  # Path to the GAML file
                  gaml_experiment_name=exp_name,  # Name of the experiment to run
                  gaml_experiment_parameters=exp_parameters,  # Parameters to pass to GAMA
                  gama_ip_address="localhost",  # GAMA server address
                  gama_port=1001)  # Communication port
```

This creates the environment using Gymnasium's `make` function:

- The environment ID is `'gama_gymnasium_env/GamaEnv-v0'`
- The parameters specify:
  - The GAML file to use (`gaml_experiment_path`)
  - The experiment to run (`gaml_experiment_name`)
  - The parameters for the experiment (`gaml_experiment_parameters`)
  - The IP address and port for the GAMA server

```python
    # Initialize episode tracking
    done = False
    # Reset the environment to get the initial observation and info
    obs, info = env.reset()
```

Before starting an episode, we need to:

- Initialize a flag to track when the episode is done
- Reset the environment to get the initial observation and info

```python
    # Main loop: continue until the episode is done
    while not done:
        # Choose a random action from the action space
        action = env.action_space.sample()
        # Execute the action in the environment and get the results
        obs, reward, terminated, truncated, info = env.step(action)
        # Print the current state of the environment
        print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Terminated: {terminated}")
  
        # Check if the episode is over (either terminated or truncated)
        if terminated or truncated:
            done = True
            print("Episode finished.")
```

This is the main loop of the episode:

1. Sample a random action from the action space
2. Take a step in the environment with that action
3. Get the observation, reward, termination flags, and info
4. Print the current state
5. Check if the episode is over (if the agent reached the target or if some other termination condition was met)

```python
    # Close the environment and clean up resources
    env.close()
```

After the episode is done, we close the environment to clean up resources.

```python
if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
```

This ensures that the `main` function is only run when this script is executed directly (not imported as a module), and it runs the async function using `asyncio.run()`.

## Running the Example

### Prerequisites

1. **GAMA Platform**: Make sure you have [GAMA](https://gama-platform.org/download) installed (version 1.9.4 or higher recommended).
2. **Python Environment**: Set up a Python environment with the required packages.

### Setup Steps

1. Install the GAMA-Gymnasium integration package:

   ```bash
   pip install gama_gymnasium
   ```

2. Launch a [GAMA Headless server](https://gama-platform.org/wiki/HeadlessServer) with the correct communication port:

   Go to the `headless` directory in your Gama installation folder and run the script `gama-headless.sh` (or `gama-headless.bat`) with the argument `-socket` followed by the port number you want your Gama server to run on.
3. Start the example:

   ```bash
   python basic_gymnasium.py
   ```

This will automatically:

- Connect to the GAMA server
- Load the `basic_env.gaml` model
- Create a new experiment with the specified parameters
- Start the interaction between Python and GAMA

The script will connect to GAMA, run the simulation, and display the actions and observations as the agent randomly explores the environment.

## Expected Output

The script will output:

1. Information about the observation space
2. A sequence of actions, observations, rewards, and termination flags
3. A message when the episode finishes (when the agent reaches the target)

## Understanding the GAMA-Gymnasium Integration

### Communication Flow

1. **Initialization**:

   - Python establishes a socket connection with GAMA
   - GAMA creates a GymAgent with the `GymnasiumLink` skill
   - Action and observation spaces are defined and shared with Python
2. **Interaction Cycle**:

   - Python generates an action (in this example, randomly)
   - The action is sent to GAMA
   - GAMA executes the action in the simulation
   - GAMA sends back the observation, reward, and termination status
   - Python processes this information and generates the next action
3. **Termination**:

   - When the episode ends (agent reaches target), GAMA sends a termination signal
   - Python closes the environment, cleaning up resources

### Technical Implementation

The integration between GAMA and Python uses:

- Socket communication for data exchange
- JSON format for structuring messages
- The `GymnasiumLink` skill in GAMA to handle the protocol

## Next Steps

Once you understand this basic example, you can:

1. Implement your own agent with a learning algorithm instead of random actions
2. Create more complex GAMA environments with multiple agents
3. Try different reinforcement learning algorithms on the same environment
4. Extend the observation and action spaces for more complex tasks
5. Add visualization and metrics to track learning progress
6. Explore the lower-level GAMA API by studying the [basic_test.py](README_basic_test.md) script
