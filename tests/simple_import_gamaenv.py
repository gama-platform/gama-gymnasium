import asyncio
from pathlib import Path

import gymnasium
from gymnasium.spaces import Discrete
from gama_gymnasium.gama_env import GamaEnv


async def main():
    experiment_path = str(Path(__file__).parents[0] / "simplest_gama_model.gaml")
    experiment_name = "expe"
    env = gymnasium.make('gama_gymnasium_env/GamaEnv-v0',
                         observation_space=Discrete(2),
                         action_space=Discrete(2),
                         gaml_experiment_path=experiment_path,
                         gaml_experiment_name=experiment_name,
                         gama_ip_address="localhost",
                         gama_port=1001)
    env.reset()

asyncio.run(main())
