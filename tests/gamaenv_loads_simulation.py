import asyncio
from pathlib import Path

import gymnasium
from gymnasium.spaces import Discrete
from gama_gymnasium.gama_env import GamaEnv

"""
In this example, we just test that our current environment is able to load the gama-gymnasium env and that it can
load a gama experiment. There is no call to the training loop or any other function than the `reset` one.

To run this example you need to have a gama server started on port 1001, or on another port and change the `gama_port` 
parameter accordingly.
"""


async def main():
    experiment_path = str(Path(__file__).parents[0] / "empty simulation.gaml")
    experiment_name = "expe"
    env = gymnasium.make('gama_gymnasium_env/GamaEnv-v0',
                         observation_space=Discrete(1),  # Here it has no meaning, so we give a simple value
                         action_space=Discrete(1),  # Here it has no meaning, so we give a simple value
                         gaml_experiment_path=experiment_path,
                         gaml_experiment_name=experiment_name,
                         gama_ip_address="localhost",
                         gama_port=1001)
    env.reset()

asyncio.run(main())
