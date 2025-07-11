"""
GAMA-Gymnasium Environment Integration

This module provides a bridge between GAMA simulation platform and OpenAI Gymnasium.
It allows GAMA simulations to be used as reinforcement learning environments
by implementing the standard Gymnasium interface.

The GamaEnv class handles communication with a GAMA server, converts between
GAMA and Gymnasium data formats, and manages the simulation lifecycle.
"""

import json
import time
import asyncio
from typing import Any, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from gama_client.sync_client import GamaSyncClient
from gama_client.message_types import *
from gama_gymnasium.space_converters import map_to_space


async def async_command_answer_handler(message: dict):
    """
    Handler for asynchronous command responses from GAMA server.
    
    Args:
        message (dict): The response message from GAMA server
    """
    print("Here is the answer to an async command: ", message)


async def gama_server_message_handler(message: dict):
    """
    Handler for unsolicited messages from GAMA server.
    
    Args:
        message (dict): The message from GAMA server
    """
    print("I just received a message from Gama-server and it's not an answer to a command!")
    print("Here it is:", message)


class GamaEnv(gym.Env):
    """
    Gymnasium environment wrapper for GAMA simulations.
    
    This class implements the standard Gymnasium interface to interact with
    GAMA simulations. It handles the communication with the GAMA server,
    manages the simulation lifecycle, and converts data between GAMA and
    Gymnasium formats.
    
    Attributes:
        gaml_file_path (str): Path to the GAML model file
        experiment_name (str): Name of the experiment to run
        experiment_parameters (list): Parameters to pass to the GAMA experiment
        gama_server_client (GamaSyncClient): Client for communicating with GAMA server
        experiment_id (str): Unique identifier for the running experiment
        observation_space (Space): Gymnasium observation space
        action_space (Space): Gymnasium action space
        render_mode (str): Rendering mode for the environment
    """

    def __init__(self, gaml_experiment_path: str, gaml_experiment_name: str, 
                 gaml_experiment_parameters: list[dict[str, Any]] | None = None,
                 gama_ip_address: str | None = None, gama_port: int = 6868, 
                 render_mode=None):
        """
        Initialize the GAMA environment.
        
        Args:
            gaml_experiment_path (str): Path to the GAML model file
            gaml_experiment_name (str): Name of the experiment in the GAML file
            gaml_experiment_parameters (list, optional): Parameters for the experiment
            gama_ip_address (str, optional): IP address of GAMA server
            gama_port (int): Port number for GAMA server communication
            render_mode (str, optional): Rendering mode (not implemented)
        """
        # Store experiment configuration
        self.gaml_file_path = gaml_experiment_path
        self.experiment_name = gaml_experiment_name
        self.experiment_parameters = gaml_experiment_parameters if gaml_experiment_parameters is not None else []
        
        # Initialize GAMA server client and establish connection
        self.gama_server_client = GamaSyncClient(
            gama_ip_address, gama_port, 
            async_command_answer_handler, gama_server_message_handler
        )
        self.gama_server_client.connect()

        # Set rendering mode
        self.render_mode = render_mode

        # Load the GAML experiment on the GAMA server
        gama_response = self.gama_server_client.load(
            self.gaml_file_path, self.experiment_name, 
            console=False, runtime=True, parameters=self.experiment_parameters
        )
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("Error while loading GAMA experiment", gama_response)
        self.experiment_id = gama_response["content"]

        # Temporary workaround: wait for environment initialization
        # TODO: Replace with proper status checking when available
        if gama_port == 1000:
            time.sleep(8)  # Allow time for the environment to initialize

        # Retrieve observation space definition from GAMA
        gama_response = self.gama_server_client.expression(
            self.experiment_id, r"GymAgent[0].observation_space"
        )
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("Error while getting observation space", gama_response)
        gama_observation_map = gama_response["content"]

        # Retrieve action space definition from GAMA
        gama_response = self.gama_server_client.expression(
            self.experiment_id, r"GymAgent[0].action_space"
        )
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("Error while getting action space", gama_response)
        gama_action_map = gama_response["content"]

        # Convert GAMA space definitions to Gymnasium spaces
        self.observation_space = map_to_space(gama_observation_map)
        self.action_space = map_to_space(gama_action_map)

    def reset(self, seed: int = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        """
        Reset the environment to its initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional options (not used)
            
        Returns:
            tuple: Initial observation and info dictionary
        """
        # Initialize random number generator with seed
        super().reset(seed=seed, options=options)

        # Temporary workaround: stop and reload the experiment
        gama_response = self.gama_server_client.stop(self.experiment_id)
        gama_response = self.gama_server_client.load(self.gaml_file_path, self.experiment_name,
                                                     console=False, runtime=True, parameters=self.experiment_parameters)
        # TODO: Implement proper reset logic
        # Reload the experiment to reset its state
        # gama_response = self.gama_server_client.reload(self.experiment_id)
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
           raise Exception("Error while reloading experiment", gama_response)
        
        # Set the random seed in GAMA
        if seed is not None:
            gama_response = self.gama_server_client.expression(
                self.experiment_id, fr"seed <- {seed};"
            )
            if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
                raise Exception("Error while setting seed", gama_response)
        else:
            # Use random seed if none provided
            random_seed = np.random.random()
            gama_response = self.gama_server_client.expression(
                self.experiment_id, fr"seed <- {random_seed};"
            )
            if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
                raise Exception("Error while setting random seed", gama_response)
            
            # Log the generated seed for debugging
            gama_response = self.gama_server_client.expression(self.experiment_id, r"seed")
            if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
                raise Exception("Error while getting seed", gama_response)
            print("Random seed set to:", gama_response["content"])
        
        # Get initial state from GAMA
        gama_response = self.gama_server_client.expression(self.experiment_id, r"GymAgent[0].state")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("Error while getting initial state", gama_response)
        state = gama_response["content"]
        state = self.observation_space.from_jsonable([state])[0]

        # Get initial info from GAMA
        gama_response = self.gama_server_client.expression(self.experiment_id, r"GymAgent[0].info")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("Error while getting initial info", gama_response)
        info = gama_response["content"]

        return state, info
    
    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment with the given action.
        
        Args:
            action (ActType): Action to execute
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Convert action to GAMA-compatible format
        action = self.action_space.to_jsonable([action])[0]
        
        # Send action to GAMA
        gama_response = self.gama_server_client.expression(
            self.experiment_id, fr"GymAgent[0].next_action <- {action};"
        )
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("Error while setting action", gama_response)

        # Execute one simulation step in GAMA
        gama_response = self.gama_server_client.step(self.experiment_id, sync=True)
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("Error while running simulation step", gama_response)
        
        # Retrieve step results from GAMA
        gama_response = self.gama_server_client.expression(self.experiment_id, r"GymAgent[0].data")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("Error while getting step data", gama_response)
        
        # Parse the data returned by GAMA
        data = gama_response["content"]
        
        # Extract and convert components
        state = self.observation_space.from_jsonable([data["State"]])[0]
        reward = data["Reward"]
        terminated = data["Terminated"]
        truncated = data["Truncated"]
        info = data["Info"]

        return state, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """
        Render the environment (placeholder implementation).
        
        Args:
            mode (str): Rendering mode
        """
        # TODO: Implement actual rendering logic
        print("Rendering the environment... (not implemented)")
        return
    
    def close(self):
        """
        Close the environment and clean up resources.
        """
        if self.gama_server_client is not None:
            self.gama_server_client.close_connection()