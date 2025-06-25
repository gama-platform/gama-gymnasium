import asyncio
from pathlib import Path
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from gama_gymnasium.gama_env import GamaEnv

import random


async def main():

    grid_size = 3
    exp_path = str(Path(__file__).parents[0] / "basic_env.gaml")
    print(exp_path)
    exp_name = "gym_env"
    exp_parameters = [{"type": "int", "name": "grid_size", "value": grid_size}]

    env = gym.make('gama_gymnasium_env/GamaEnv-v0',
                  gaml_experiment_path=exp_path,
                  gaml_experiment_name=exp_name,
                  gaml_experiment_parameters=exp_parameters,
                  gama_ip_address="localhost",
                  gama_port=1001)
    
    print("Observation space:", env.observation_space)
    print(f"Example observation: {env.observation_space.sample()}")

    done = False
    obs, info = env.reset()
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Terminated: {terminated}")
        
        if terminated or truncated:
            done = True
            print("Episode finished.")

    env.close()

if __name__ == "__main__":
    asyncio.run(main())