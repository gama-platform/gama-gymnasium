import sys
from typing import Any, SupportsFloat, Dict

import gama_client.base_client
import gymnasium as gym
import numpy as np
from gymnasium import Space
from gymnasium.core import ActType, ObsType
from numpy.typing import NDArray
import socket
from gama_client import *




async def async_command_answer_handler(message: Dict):
    print("Here is the answer to an async command: ", message)


async def gama_server_message_handler(message: Dict):
    print("I just received a message from Gama-server and it's not an answer to a command!")
    print("Here it is:", message)

# TODO: add info as return for reset and step ?
class GamaEnv(gym.Env):

    # USER LOCAL VARIABLES
    gaml_file_path: str             # Path to the gaml file containing the experiment/simulation to run
    experiment_name: str            # Name of the experiment to run

    # GAMA server variables
    gama_server_client: gama_client.base_client.GamaBaseClient = None

    # Simulation execution variables
    simulation_socket = None
    simulation_as_file = None
    simulation_connection = None # Resulting from socket create connection

    def __init__(self, gaml_experiment_path: str, gaml_experiment_name: str, observation_space: Space[ObsType],
                 action_space: Space[ActType], gama_ip_address: str | None, gama_port: int = 6868, render_mode=None):

        self.gaml_file_path = gaml_experiment_path
        self.experiment_name = gaml_experiment_name

        # If we don't have an ip address then we try to run gama-server ourselves
        if gama_ip_address is None:
            exec(f"gama -socket {gama_port}")
            gama_ip_address = "localhost"

        self.gama_server_client = gama_client.sync_client.GamaSyncClient(gama_ip_address, gama_port, async_command_answer_handler, gama_server_message_handler)

        # We try to connect to gama-server

        self.observation_space = observation_space
        self.action_space = action_space

        self.render_mode = render_mode

    def reset(self, seed: int = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        #print("RESET")

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        #print("self.gama_simulation_as_file", self.gama_simulation_as_file)
        #print("self.gama_simulation_connection",
        #      self.gama_simulation_connection)

        #Check if the environment terminated
        if self.simulation_connection is not None:
            #print("self.gama_simulation_connection.fileno()",
            #      self.gama_simulation_connection.fileno())
            if self.simulation_connection.fileno() != -1:
                self.simulation_connection.shutdown(socket.SHUT_RDWR)
                self.simulation_connection.close()
                self.simulation_socket.shutdown(socket.SHUT_RDWR)
                self.simulation_socket.close()
        if self.simulation_as_file is not None:
            self.simulation_as_file.close()
            self.simulation_as_file = None

        self.clean_subprocesses()

        # Starts gama and get initial state
        self.run_gama_simulation()
        self.wait_for_gama_to_connect()
        self.state, end = self.read_observations()

        #print('after reset self.state', self.state)
        #print('after reset end', end)
        #print("END RESET")

        return np.array(self.state, dtype=np.float32), {}

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        try:
            #print("STEP")

            # sending actions
            str_action = GamaEnv.action_to_string(np.array(action)) #TODO: can we always convert it to an array ?

            self.simulation_as_file.write(str_action)
            self.simulation_as_file.flush()
            #print("model sent policy, now waiting for reward")

            # we wait for the reward
            policy_reward = self.simulation_as_file.readline()
            reward = float(policy_reward)

            #print("model received reward:", policy_reward, " as a float: ", reward)
            self.state, end = self.read_observations()
            #print("observations received", self.state, end)
            # If it was the final step, we need to send a message back to the simulation once everything done to acknowledge that it can now close
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
            print("EXCEPTION pendant l'execution")
            print(sys.exc_info()[0])
            sys.exit(-1)
        #print("END STEP")
        return self.state, reward, end, False, {}

    def render(self):
        pass
        # TODO: check that we can't do something with snapshots maybe ?

    def close(self):
        # TODO: check that there's nothing to close on the gama side
        pass


    # Init the server + run gama
    def run_gama_simulation(self):
        """
        In this function we ask gama-server to run the simulation with the parameters we want
        :return:
        """

        port = self.listener_init()
        xml_path = GamaEnv.generate_gama_xml(self.headless_dir, port, self.gaml_file_path, self.experiment_name)
        self.thread_id = start_new_thread(GamaEnv.run_gama_headless, (self, xml_path, self.headless_dir, self.run_headless_script_path))

    def read_observations(self):

        received_observations: str = self.simulation_as_file.readline()
        #print("model received:", received_observations)

        over = "END" in received_observations
        obs = GamaEnv.string_to_nparray(received_observations.replace("END", ""))
        #obs[2]  = float(self.n_times_4_action - self.i_experience)  # We change the last observation to be the number of times that remain for changing the policy

        return obs, over

    # Converts a string to a numpy array of floats
    @classmethod
    def string_to_nparray(cls, array_as_string: str) -> NDArray[np.float64]:
        # first we remove brackets and parentheses
        clean = "".join([c if c not in "()[]{}" else '' for c in str(array_as_string)])
        # then we split into numbers
        nbs = [float(nb) for nb in filter(lambda s: s.strip() != "", clean.split(','))]
        return np.array(nbs)


    # Converts an action to a string to be sent to the simulation
    @classmethod
    def action_to_string(cls, actions: NDArray) -> str:
        return ",".join([str(action) for action in actions]) + "\n"