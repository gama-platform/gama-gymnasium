import json
import time
import asyncio
from typing import Any, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium import Space
from gymnasium.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Text, Tuple, Dict, Sequence, Graph, OneOf
from gymnasium.core import ActType, ObsType

from gama_client.sync_client import GamaSyncClient

from gama_gymnasium.message_util import *
from gama_client.message_types import *


async def async_command_answer_handler(message: dict):
    print("Here is the answer to an async command: ", message)


async def gama_server_message_handler(message: dict):
    print("I just received a message from Gama-server and it's not an answer to a command!")
    print("Here it is:", message)


class GamaEnv(gym.Env):
    """
    This class is a placeholder for the GamaEnv implementation. It should be replaced with the actual implementation
    that interacts with the Gama server.
    """

    def __init__(self, gaml_experiment_path: str, gaml_experiment_name: str, gaml_experiment_parameters: list[dict[str, Any]] | None = None,
                 gama_ip_address: str | None = None, gama_port: int = 6868, render_mode=None):
        

        self.gaml_file_path = gaml_experiment_path
        self.experiment_name = gaml_experiment_name
        self.experiment_parameters = gaml_experiment_parameters if gaml_experiment_parameters is not None else []
        
        # Creating the object to interact with gama server
        self.gama_server_client = GamaSyncClient(gama_ip_address, gama_port, async_command_answer_handler,
                                                 gama_server_message_handler)
        # We try to connect to gama-server
        self.gama_server_client.connect()

        # Finally we allocate the gymnasium environment variables
        self.render_mode = render_mode

        gama_response = self.gama_server_client.load(self.gaml_file_path, self.experiment_name, console=False, runtime=True, parameters=self.experiment_parameters)
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while loading", gama_response)
        self.experiment_id = gama_response["content"]

        if gama_port == 1000:
            time.sleep(5)  # Allow some time for the environment to initialize

        # while done:
        #     gama_response = self.gama_server_client.listen()
        #     if gama_response["type"] == MessageTypes.CommandExecutedSuccessfully.value:
        #         print("Gama response:", gama_response["content"])
        #         done = True

        gama_response = self.gama_server_client.expression(self.experiment_id, r"GymAgent[0].observation_space")
        # gama_response = self.gama_server_client.expression(self.experiment_id, r"observation_space")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while getting observation space", gama_response)
        gama_observation_map = json.loads(gama_response["content"])
        # print("Gama observation map:", gama_observation_map)

        gama_response = self.gama_server_client.expression(self.experiment_id, r"GymAgent[0].action_space")
        # gama_response = self.gama_server_client.expression(self.experiment_id, r"action_space")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while getting action space", gama_response)
        gama_action_map = json.loads(gama_response["content"])
        # print("Gama action map:", gama_action_map)

        self.observation_space = map_to_space(gama_observation_map)
        self.action_space = map_to_space(gama_action_map)

    def reset(self, seed: int = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:

        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        gama_response = self.gama_server_client.reload(self.experiment_id)
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while reloading", gama_response)
        
        # Set the seed for the experiment
        if seed is not None: 
            gama_response = self.gama_server_client.expression(self.experiment_id, fr"seed <- {seed};")
            if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
                raise Exception("error while setting seed", gama_response)
        else:
            gama_response = self.gama_server_client.expression(self.experiment_id, fr"seed <- {np.random.random()};")
            if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
                raise Exception("error while setting seed to 0", gama_response)
        
        gama_response = self.gama_server_client.expression(self.experiment_id, r"GymAgent[0].state")
        state = json.loads(gama_response["content"])
        state = self.observation_space.from_jsonable([state])[0]
        # print("State after reset:", state)
        # print("Is in observation space:", self.observation_space.contains(state))

        gama_response = self.gama_server_client.expression(self.experiment_id, r"GymAgent[0].info")
        info = json.loads(gama_response["content"])
        

        return state, info
    
    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # start_set_action = time.perf_counter()
        action = self.action_space.to_jsonable([action])[0]
        gama_response = self.gama_server_client.expression(self.experiment_id, fr"GymAgent[0].next_action <- {action};")
        # end_set_action = time.perf_counter()
        # print(f"Setting action took {end_set_action - start_set_action:.5f} seconds")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while setting action", gama_response)

        # start_step = time.perf_counter()
        gama_response = self.gama_server_client.step(self.experiment_id, sync=True)
        # end_step = time.perf_counter()
        # print(f"Step took {end_step - start_step:.5f} seconds")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while running step", gama_response)
        
        # start_get_data = time.perf_counter()
        gama_response = self.gama_server_client.expression(self.experiment_id, r"GymAgent[0].data")
        # end_get_data = time.perf_counter()
        # print(f"Getting state took {end_get_data - start_get_data:.5f} seconds")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while getting state", gama_response)
        data = json.loads(gama_response["content"])
        state = self.observation_space.from_jsonable([data["State"]])[0]
        reward = data["Reward"]
        terminated = data["Terminated"]
        truncated = data["Truncated"]
        info = data["Info"]

        # end = time.perf_counter()

        # print(f"Total time for step: {end - start_set_action:.5f} seconds")

        return state, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        # Placeholder for rendering logic
        print("Rendering the environment... (not implemented)")

        return
    
    def close(self):

        if self.gama_server_client is not None:
            self.gama_server_client.close_connection()

    def convert_action_to_gama(self, action: ActType) -> str:
        """
        Converts the action to a GAMA-compatible string format.
        This is a placeholder and should be replaced with actual conversion logic.
        """
        return str(action)
    
    def convert_observation_from_gama(self, observation: str) -> ObsType:
        """
        Converts the observation from GAMA format to a format compatible with Gymnasium.
        This is a placeholder and should be replaced with actual conversion logic.
        """
        return np.array(json.loads(observation), dtype=np.int32)

def map_to_space(map):
    if "type" in map:
        if map["type"] == "Discrete":
            return map_to_discrete(map)
        elif map["type"] == "Box":
            return map_to_box(map)
        elif map["type"] == "MultiBinary":
            return map_to_multi_binary(map)
        elif map["type"] == "MultiDiscrete":
            return map_to_multi_discrete(map)
        elif map["type"] == "Text":
            return map_to_text(map)
        # elif map["type"] == "Tuple":
        #     return map_to_tuple(map)
        # elif map["type"] == "Dict":
        #     return map_to_dict(map)
        # elif map["type"] == "Sequence":
        #     return map_to_sequence(map)
        # elif map["type"] == "Graph":
        #     return map_to_graph(map)
        # elif map["type"] == "OneOf":
        #     return map_to_one_of(map)
        else:
            print("Unknown type in the map, cannot map to space.")
            return None
    else:
        print("No type specified in the map, cannot map to space.")
        return None

def map_to_box(box):
    if "low" in box:
        if isinstance(box["low"], list):
            low = np.array(replace_infinity(box["low"]))
        else:
            low = box["low"]
    else:
        low = -np.inf
    
    if "high" in box:
        if isinstance(box["high"], list):
            high = np.array(replace_infinity(box["high"]))
        else:
            high = box["high"]
    else:
        high = np.inf

    if "shape" in box:
        shape = box["shape"]
    else:
        shape = None

    if "dtype" in box:
        if box["dtype"] == "int":
            dtype = np.int64
        elif box["dtype"] == "float":
            dtype = np.float64
        else:
            print("Unknown dtype in the box, defaulting to float32.")
            dtype = np.float32
    else:
        dtype = np.float32

    return Box(low=low, high=high, shape=shape, dtype=dtype)

def map_to_discrete(discrete):
    n = discrete["n"]
    if "start" in discrete:
        start = discrete["start"]
        return Discrete(n, start=start)
    else:
        return Discrete(n)

def map_to_multi_binary(mb):
    n = mb["n"]
    if len(n) == 1:
        return MultiBinary(n[0])
    else:
        return MultiBinary(n)

def map_to_multi_discrete(md):
    nvec = md["nvec"]
    if "start" in md:
        start = md["start"]
        return MultiDiscrete(nvec, start=start)
    else:
        return MultiDiscrete(nvec)

def map_to_text(text):
    if "min_length" in text:
        min = text["min_length"]
    else:
        min = 0

    if "max_length" in text:
        max = text["max_length"]
    else:
        max = 1000
        
    return Text(min_length=min, max_length=max)

def replace_infinity(data):
    if isinstance(data, list):
        return [replace_infinity(item) for item in data]
    elif data == "Infinity":
        return float('inf')
    elif data == "-Infinity":
        return float('-inf')
    else:
        return data