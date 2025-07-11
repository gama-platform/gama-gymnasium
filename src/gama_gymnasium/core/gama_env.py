"""
GAMA-Gymnasium Environment Integration

This module provides the main GamaEnv class that implements the Gymnasium
interface for GAMA simulations.
"""

import time
from typing import Any, SupportsFloat, Optional, Dict

import numpy as np
import gymnasium as gym
from gymnasium.core import ActType, ObsType

from .client import GamaClient
from .message_handler import MessageHandler
from ..spaces.converters import map_to_space
from ..utils.exceptions import ExperimentError, ConfigurationError


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
        gama_client (GamaClient): Client for communicating with GAMA server
        experiment_id (str): Unique identifier for the running experiment
        observation_space (Space): Gymnasium observation space
        action_space (Space): Gymnasium action space
        render_mode (str): Rendering mode for the environment
        message_handler (MessageHandler): Handler for message processing
    """

    def __init__(
        self,
        gaml_experiment_path: str,
        gaml_experiment_name: str,
        gaml_experiment_parameters: Optional[list[dict[str, Any]]] = None,
        gama_ip_address: Optional[str] = None,
        gama_port: int = 6868,
        render_mode: Optional[str] = None,
        initialization_wait: float = 0.0
    ):
        """
        Initialize the GAMA environment.
        
        Args:
            gaml_experiment_path (str): Path to the GAML model file
            gaml_experiment_name (str): Name of the experiment in the GAML file
            gaml_experiment_parameters (list, optional): Parameters for the experiment
            gama_ip_address (str, optional): IP address of GAMA server
            gama_port (int): Port number for GAMA server communication
            render_mode (str, optional): Rendering mode (not implemented)
            initialization_wait (float): Time to wait for environment initialization
        """
        # Store experiment configuration
        self.gaml_file_path = gaml_experiment_path
        self.experiment_name = gaml_experiment_name
        self.experiment_parameters = gaml_experiment_parameters or []
        self.render_mode = render_mode
        self.initialization_wait = initialization_wait
        
        # Initialize message handler
        self.message_handler = MessageHandler()
        
        # Initialize and connect GAMA client
        self.gama_client = GamaClient(gama_ip_address, gama_port)
        self.gama_client.connect()
        
        # Load the experiment and initialize spaces
        self._load_experiment()
        self._initialize_spaces()
        
        # Wait for environment initialization if needed
        if self.initialization_wait > 0:
            time.sleep(self.initialization_wait)

    def _load_experiment(self) -> None:
        """Load the GAMA experiment."""
        try:
            self.experiment_id = self.gama_client.load_experiment(
                self.gaml_file_path,
                self.experiment_name,
                self.experiment_parameters
            )
        except Exception as e:
            raise ExperimentError(f"Failed to load experiment: {e}")

    def _initialize_spaces(self) -> None:
        """Initialize observation and action spaces from GAMA."""
        try:
            # Get observation space definition
            obs_space_def = self.gama_client.execute_expression(
                self.experiment_id, r"GymAgent[0].observation_space"
            )
            self.message_handler.validate_space_definition(obs_space_def)
            self.observation_space = map_to_space(obs_space_def)
            
            # Get action space definition  
            action_space_def = self.gama_client.execute_expression(
                self.experiment_id, r"GymAgent[0].action_space"
            )
            self.message_handler.validate_space_definition(action_space_def)
            self.action_space = map_to_space(action_space_def)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize spaces: {e}")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> tuple[ObsType, Dict[str, Any]]:
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
        
        try:
            # Reload the experiment to reset its state
            self.experiment_id = self.gama_client.reload_experiment(
                self.experiment_id,
                self.gaml_file_path,
                self.experiment_name,
                self.experiment_parameters
            )
            
            # Set the random seed in GAMA
            if seed is not None:
                self.gama_client.execute_expression(
                    self.experiment_id, f"seed <- {seed};"
                )
            else:
                # Use random seed if none provided
                random_seed = np.random.random()
                self.gama_client.execute_expression(
                    self.experiment_id, f"seed <- {random_seed};"
                )
                
                # Log the generated seed for debugging
                actual_seed = self.gama_client.execute_expression(
                    self.experiment_id, "seed"
                )
                print(f"Random seed set to: {actual_seed}")
            
            # Get initial state from GAMA
            state = self.gama_client.execute_expression(
                self.experiment_id, r"GymAgent[0].state"
            )
            state = self.observation_space.from_jsonable([state])[0]
            
            # Get initial info from GAMA
            info = self.gama_client.execute_expression(
                self.experiment_id, r"GymAgent[0].info"
            )
            
            return state, info
            
        except Exception as e:
            raise ExperimentError(f"Failed to reset environment: {e}")

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment with the given action.
        
        Args:
            action (ActType): Action to execute
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        try:
            # Convert action to GAMA-compatible format
            action_jsonable = self.action_space.to_jsonable([action])[0]
            action_formatted = self.message_handler.format_action_for_gama(action_jsonable)
            
            # Send action to GAMA
            self.gama_client.execute_expression(
                self.experiment_id, f"GymAgent[0].next_action <- {action_formatted};"
            )
            
            # Execute one simulation step
            self.gama_client.step_simulation(self.experiment_id, sync=True)
            
            # Retrieve step results from GAMA
            data = self.gama_client.execute_expression(
                self.experiment_id, r"GymAgent[0].data"
            )
            
            # Validate and parse the data
            self.message_handler.validate_step_data(data)
            
            # Extract and convert components
            state = self.observation_space.from_jsonable([data["State"]])[0]
            reward = data["Reward"]
            terminated = data["Terminated"]
            truncated = data["Truncated"]
            info = data["Info"]
            
            return state, reward, terminated, truncated, info
            
        except Exception as e:
            raise ExperimentError(f"Failed to execute step: {e}")

    def render(self, mode: str = 'human') -> None:
        """
        Render the environment (placeholder implementation).
        
        Args:
            mode (str): Rendering mode
        """
        # TODO: Implement actual rendering logic
        if self.render_mode is not None:
            print(f"Rendering environment in {mode} mode... (not implemented)")

    def close(self) -> None:
        """Close the environment and clean up resources."""
        if hasattr(self, 'gama_client') and self.gama_client.is_connected:
            self.gama_client.disconnect()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
