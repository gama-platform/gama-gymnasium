import asyncio
import sys
from typing import Any, SupportsFloat, Dict, Tuple

import gama_client.base_client
import gymnasium as gym
import numpy as np
from gymnasium import Space
from gymnasium.core import ActType, ObsType
from numpy.typing import NDArray
import socket
from gama_client import *

from gama_client.sync_client import GamaSyncClient

from gama_gymnasium.message_util import *
from gama_client.message_types import *


async def async_command_answer_handler(message: Dict):
    print("Here is the answer to an async command: ", message)


async def gama_server_message_handler(message: Dict):
    print("I just received a message from Gama-server and it's not an answer to a command!")
    print("Here it is:", message)


# TODO: add info as return for reset and step ?
class GamaEnv(gym.Env):
    # USER LOCAL VARIABLES
    gaml_file_path: str  # Path to the gaml file containing the experiment/simulation to run
    experiment_name: str  # Name of the experiment to run

    # GAMA server variables
    gama_server_client: GamaSyncClient = None
    """
    This is the id used by gama-server to identify the simulation we are manipulating.
    It is provided by gama-server as a return when we are done loading the simulation.
    """
    simulation_id: str = None

    # Simulation execution variables
    simulation_socket = None
    simulation_as_file = None
    simulation_connection = None  # Resulting from socket create connection

    def __init__(self, gaml_experiment_path: str, gaml_experiment_name: str,
                 observation_space: Space[ObsType], action_space: Space[ActType],
                 gama_ip_address: str | None = None, gama_port: int = 6868, render_mode=None):

        self.state = None
        self.gaml_file_path = gaml_experiment_path
        self.experiment_name = gaml_experiment_name

        # Creating the object to interact with gama server
        self.gama_server_client = GamaSyncClient(gama_ip_address, gama_port, async_command_answer_handler,
                                                 gama_server_message_handler)
        # We try to connect to gama-server
        self.gama_server_client.sync_connect()

        # Finally we allocate the gymnasium environment variables
        self.observation_space = observation_space
        self.action_space = action_space
        self.render_mode = render_mode

    def reset(self, seed: int = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:

        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        # Check if the environment terminated
        if self.simulation_connection is not None:
            if self.simulation_connection.fileno() != -1:
                self.simulation_connection.shutdown(socket.SHUT_RDWR)
                self.simulation_connection.close()
                self.simulation_socket.shutdown(socket.SHUT_RDWR)
                self.simulation_socket.close()
        if self.simulation_as_file is not None:
            self.simulation_as_file.close()
            self.simulation_as_file = None

        # Starts gama and get initial state
        self.run_gama_simulation()
        self.wait_for_gama_to_connect()
        self.state, end = self.read_observations()

        return self.state, {}  # TODO: currently no additional information

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        reward = None
        end = True
        try:

            # sending actions
            str_action = action_to_string(np.array(action))
            self.simulation_as_file.write(str_action)
            self.simulation_as_file.flush()

            # we wait for the reward
            policy_reward = self.simulation_as_file.readline()
            reward = float(policy_reward)

            # We read observations from the simulation and set the state
            self.state, end = self.read_observations()

            # If it was the final step, we need to send a message back to the simulation once
            # everything is done to acknowledge that it can now close
            if end:
                self.simulation_as_file.write("END\n")
                self.simulation_as_file.flush()
                self.simulation_as_file.close()
                self.simulation_connection.shutdown(socket.SHUT_RDWR)
                self.simulation_connection.close()
                self.simulation_socket.shutdown(socket.SHUT_RDWR)
                self.simulation_socket.close()
        except ConnectionResetError:
            print("connection reset, end of simulation")
        except:
            print("EXCEPTION during runtime")
            print(sys.exc_info()[0])
            sys.exit(-1)

        return self.state, reward, end, False, {}  # TODO: here too we don't provide information yet

    def render(self):
        pass
        # TODO: check that we can't do something with snapshots maybe ?

    def close(self):
        # Closing the connection to gama-server
        if self.gama_server_client is not None:
            self.gama_server_client.sync_close_connection()

    def run_gama_simulation(self) -> bool:
        """
        This function asks gama-server to run the simulation described at initialization of the environment.
        We also set up a communication channel to this simulation and provide the port with which to communicate
        as a simulation parameter.
        """
        communication_port = self.listener_init()
        parameters = [
            {
                "name": "communication_port",
                "value": communication_port,
                "type": "int"
            },
        ]

        server_answer = self.gama_server_client.sync_load(self.gaml_file_path, self.experiment_name, False,
                                                          False, False, False, parameters=parameters)
        if server_answer["type"] == MessageTypes.CommandExecutedSuccessfully:
            self.simulation_id = server_answer["content"]
            return True
        return False

    # Initialize the socket to communicate with gama
    def listener_init(self) -> int:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket successfully created")

        s.bind(('', 0))  # localhost + port given by the os
        port = s.getsockname()[1]
        print("Socket bound to %s" % port)

        s.listen()
        print("Socket started listening")

        self.simulation_socket = s
        return port

    def wait_for_gama_to_connect(self):
        """Waits for the gama simulation to be completely initialised and for it to connect on the
        server opened by the environment to exchange actions and observations."""
        self.gama_simulation_connection, addr = self.simulation_socket.accept()
        print("gama connected:", self.gama_simulation_connection, addr)
        self.gama_simulation_as_file = self.gama_simulation_connection.makefile(mode='rw')
        print("self.gama_simulation_as_file", self.gama_simulation_as_file)

    def read_observations(self) -> Tuple[ObsType, bool]:
        """
        Reads the observations from the gama simulation.
        :return: A tuple containing the observation and a boolean indicating if the simulation has ended or not
        """
        received_observations: str = self.simulation_as_file.readline()
        # print("model received:", received_observations)

        over = self.is_simulation_over(received_observations)
        obs = string_to_nparray(received_observations)

        return obs, over

    def is_simulation_over(self, received_observations: str) -> bool:
        """
        Given the observations received, determines if the simulation is over or not.
        This method should be overwritten in case another communication protocol is used.
        :param received_observations: the whole message representing the observations sent by the simulation
        :return: True if the simulation has stopped
        """
        return observation_contains_end(received_observations)
