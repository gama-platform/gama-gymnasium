import asyncio
from pathlib import Path

import gymnasium
from gymnasium.spaces import Discrete, Box, Sequence
from gama_gymnasium.gama_env import GamaEnv

"""
In this example, we load a gama-gymnasium env and simulate a learning loop. No actual learning is done, this is just
to test that all the functionalities are available.

To run this example you need to have a gama server started on port 1001, or on another port and change the `gama_port` 
parameter accordingly.
"""


async def main():

    # parameters to access the file containing the gama simulation
    experiment_path = str(Path(__file__).parents[0] / "minimal example.gaml")
    experiment_name = "expe"

    # learning parameters
    nb_learning_iteration = 3
    nb_step_per_learning_iteration = 100

    env = gymnasium.make('gama_gymnasium_env/GamaEnv-v0',
                         observation_space=Sequence(Box(low=0, high=200)),
                         action_space=Sequence(Box(low=0, high=200)),
                         gaml_experiment_path=experiment_path,
                         gaml_experiment_name=experiment_name,
                         gama_ip_address="localhost",
                         gama_port=1001)

    for i in range(nb_learning_iteration):
        # Here we initialize the simulation
        env.reset()
        for j in range(nb_step_per_learning_iteration):
            env.step([1])
    env.close()

asyncio.run(main())
