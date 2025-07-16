"""
CartPole Gymnasium Example with GAMA Integration

This script demonstrates how to use a GAMA simulation as a Gymnasium environment
for reinforcement learning. It connects to a GAMA CartPole environment and runs
random actions to test the integration.

The example showcases:
- Setting up a GAMA-Gymnasium environment
- Basic interaction loop (reset, step, observation)
- Performance monitoring (step timing)
- Proper environment cleanup
"""

import asyncio
from pathlib import Path
import numpy as np
import time

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from gama_gymnasium.gama_env import GamaEnv


async def main():
    """
    Main function that demonstrates GAMA-Gymnasium integration.
    
    This function:
    1. Creates a GAMA environment using the Gymnasium interface
    2. Runs a series of random actions
    3. Monitors performance and displays results
    4. Properly closes the environment
    """
    # List to store execution times for performance analysis
    step_times = []

    # Define paths to the GAMA model and experiment
    # The .gaml file contains the CartPole simulation model
    exp_path = str(Path(__file__).parents[0] / "cartpole_env.gaml")
    exp_name = "gym_env"  # Name of the experiment defined in the GAML file

    # Create the Gymnasium environment with GAMA backend
    # This connects to a GAMA server running the CartPole simulation
    env = gym.make('gama_gymnasium_env/GamaEnv-v0',
                  gaml_experiment_path=exp_path,        # Path to GAMA model file
                  gaml_experiment_name=exp_name,        # Experiment name in GAMA
                  gama_ip_address="localhost",          # GAMA server IP
                  gama_port=1000)                       # GAMA server port, change for server mode (ex: 1001, 6868...)

    # Display environment specifications
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Initialize episode control variable
    done = False
    
    # Reset the environment to get initial observation
    # This starts a new episode in the GAMA simulation
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    
    # Allow time for the GAMA environment to properly initialize
    time.sleep(5)
    
    # Main interaction loop - run for a fixed number of steps
    # In a real RL scenario, this would be replaced by an agent's policy
    print("\n--- Starting interaction loop ---")
    for step in range(100):  # Run for 100 steps for demonstration
        # Sample a random action from the action space
        # In RL training, this would be the agent's action selection
        action = env.action_space.sample()
        
        # Measure step execution time for performance monitoring
        start_time = time.perf_counter()
        
        # Execute the action in the environment
        # Returns: observation, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record execution time
        end_time = time.perf_counter()
        step_times.append(end_time - start_time)
        
        # Display step information for debugging/monitoring
        print(f"Step {step+1}: Action: {action}, Obs: {obs}, Reward: {reward}, Terminated: {terminated}")
        
        # Check if episode has ended (either terminated or truncated)
        if terminated or truncated:
            done = True
            print("Episode finished - resetting environment.")
            
            # Reset environment for a new episode
            obs, info = env.reset()
            # Allow time for environment reset
            time.sleep(5)

    # Properly close the environment and clean up resources
    # This ensures the GAMA connection is properly terminated
    env.close()
    
    # Calculate and display performance statistics
    avg_step_time = np.mean(step_times)
    print(f"\n--- Performance Summary ---")
    print(f"Total steps executed: {len(step_times)}")
    print(f"Average step time: {avg_step_time:.5f} seconds")
    print(f"Steps per second: {1/avg_step_time:.2f}")


if __name__ == "__main__":
    """
    Entry point of the script.
    
    This script uses asyncio to handle the asynchronous communication
    with the GAMA server. Make sure GAMA is running in server mode
    before executing this script.
    
    To run GAMA in GUI mode:
    keep the gama port as default (1000)
    
    To run GAMA in server mode:
    ./gama-headless.sh -socket 1001  # Linux/MacOS
    gama-headless.bat -socket 1001   # Windows
    """
    print("Starting GAMA-Gymnasium CartPole example...")
    print("Make sure GAMA server is running on localhost:1000")
    asyncio.run(main())