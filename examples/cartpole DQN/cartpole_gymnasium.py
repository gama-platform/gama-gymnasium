import asyncio
from pathlib import Path
import numpy as np
import time

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from gama_gymnasium.gama_env import GamaEnv


async def main():
    step_times = []

    exp_path = str(Path(__file__).parents[0] / "cartpole_env.gaml")
    exp_name = "gym_env"

    env = gym.make('gama_gymnasium_env/GamaEnv-v0',
                  gaml_experiment_path=exp_path,
                  gaml_experiment_name=exp_name,
                  gama_ip_address="localhost",
                  gama_port=1000)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    done = False
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    time.sleep(5)  # Allow some time for the environment to initialize
    # while not done:
    for _ in range(100):  # Run for a fixed number of steps
        action = env.action_space.sample()
        start_time = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        end_time = time.perf_counter()
        step_times.append(end_time - start_time)
        print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Terminated: {terminated}")
        
        if terminated or truncated:
            done = True
            print("Episode finished.")
            obs, info = env.reset()
            time.sleep(5)  # Allow some time before the next episode starts

    env.close()
    avg_step_time = np.mean(step_times)
    print(f"Average step time: {avg_step_time:.5f} seconds")

if __name__ == "__main__":
    asyncio.run(main())