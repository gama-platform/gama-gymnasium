"""
Basic example showing how to use the GAMA-Gymnasium integration.
This script connects to a GAMA simulation and interacts with it through
the Gymnasium interface, using random actions to explore the environment.
"""

import asyncio
from pathlib import Path

import gymnasium as gym
from gama_gymnasium.gama_env import GamaEnv


async def main():
    """
    Main asynchronous function that creates and runs a GAMA environment through Gymnasium.
    Uses random actions to interact with the environment until the episode ends.
    """

    # Set the size of the grid environment
    grid_size = 3
    # Get the absolute path to the GAMA model file
    exp_path = str(Path(__file__).parents[0] / "basic_env.gaml")
    # Name of the experiment defined in the GAML file
    exp_name = "gym_env"
    # Parameters to pass to the GAMA experiment (here we set the grid size)
    exp_parameters = [{"type": "int", "name": "grid_size", "value": grid_size}]

    # Create the GAMA environment using the Gymnasium make function
    # This connects to the GAMA server and initializes the environment
    env = gym.make('gama_gymnasium_env/GamaEnv-v0',
                  gaml_experiment_path=exp_path,  # Path to the GAML file
                  gaml_experiment_name=exp_name,  # Name of the experiment to run
                  gaml_experiment_parameters=exp_parameters,  # Parameters to pass to GAMA
                  gama_ip_address="localhost",  # GAMA server address
                  gama_port=1000)  # Communication port
    
    # Display information about the observation space
    print("Observation space:", env.observation_space)
    print(f"Example observation: {env.observation_space.sample()}")

    # Initialize episode tracking
    done = False
    # Reset the environment to get the initial observation and info
    obs, info = env.reset()
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

    # Close the environment and clean up resources
    env.close()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())