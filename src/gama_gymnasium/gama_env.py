"""
Main GamaEnv class implementing the Gymnasium interface for GAMA simulations.
"""
import time
from typing import Any, SupportsFloat
import traceback
import json
import numpy as np
import gymnasium as gym
from gymnasium.core import ActType, ObsType

from .gama_client_wrapper import GamaClientWrapper
from .space_converter import SpaceConverter
from .exceptions import GamaEnvironmentError


class GamaEnv(gym.Env):
    """
    Gymnasium environment that connects to GAMA simulation server.
    
    This class provides the standard Gymnasium interface (reset, step, render, close)
    while communicating with a GAMA simulation in the background.
    """

    def __init__(self, gaml_experiment_path: str, gaml_experiment_name: str, 
                 gaml_experiment_parameters: list[dict[str, Any]] | None = None,
                 gama_ip_address: str | None = None, gama_port: int = 6868, 
                 render_mode=None):
        """
        Initialize the GAMA-Gymnasium environment.
        
        Args:
            gaml_experiment_path: Path to the .gaml file containing the experiment
            gaml_experiment_name: Name of the experiment to run
            gaml_experiment_parameters: Optional parameters for the experiment
            gama_ip_address: IP address of GAMA server (default: localhost)
            gama_port: Port of GAMA server (default: 6868)
            render_mode: Rendering mode (currently not implemented)
        """
        # Store configuration
        self.gaml_file_path = gaml_experiment_path
        self.experiment_name = gaml_experiment_name
        self.experiment_parameters = gaml_experiment_parameters or []
        self.render_mode = render_mode
        
        # Initialize GAMA client wrapper
        self.gama_client = GamaClientWrapper(gama_ip_address, gama_port)
        
        # Initialize space converter
        self.space_converter = SpaceConverter()
        
        # Connect and setup experiment
        self._setup_experiment()
        self._setup_spaces()

    def _setup_experiment(self):
        """Setup the GAMA experiment and get experiment ID."""
        self.experiment_id = self.gama_client.load_experiment(
            self.gaml_file_path, 
            self.experiment_name, 
            self.experiment_parameters
        )

    def _setup_spaces(self):
        """Setup observation and action spaces from GAMA."""
        # Get spaces from GAMA
        obs_space_data = self.gama_client.get_observation_space(self.experiment_id)
        action_space_data = self.gama_client.get_action_space(self.experiment_id)
        
        # Convert to Gymnasium spaces
        self.observation_space = self.space_converter.map_to_space(obs_space_data)
        self.action_space = self.space_converter.map_to_space(action_space_data)

    def _fast_act_encode(self, action) -> Any:
        """
        Optimization 1: Encode action without Gymnasium's to_jsonable() overhead.
        For Discrete → plain int; for Box → plain float or list of float.
        Falls back to the generic path for unknown space types.
        """
        space = self.action_space
        if isinstance(space, gym.spaces.Discrete):
            return int(action)
        elif isinstance(space, gym.spaces.Box):
            if np.ndim(action) == 0:
                return float(action)
            return [float(v) for v in np.ravel(action)]
        # Generic fallback
        return space.to_jsonable([action])[0]

    def _fast_obs_convert(self, state) -> Any:
        """
        Optimization 2: Convert GAMA observation without SpaceConverter overhead.
        For Box → direct np.array with the right dtype.
        Falls back to the generic SpaceConverter path for unknown types.
        """
        space = self.observation_space
        if isinstance(space, gym.spaces.Box):
            return np.array(state, dtype=space.dtype)
        # Generic fallback
        return self.space_converter.convert_gama_to_gym_observation(space, state)

    def reset(self, seed: int = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and return initial observation."""
        super().reset(seed=seed, options=options)
        
        # Reset GAMA experiment
        self.gama_client.reset_experiment(self.experiment_id, seed)
        
        # Get initial state
        state = self.gama_client.get_state(self.experiment_id)
        info = self.gama_client.get_info(self.experiment_id)
        
        # print("GAMA experiment reset. Initial state:", state)
        
        # Convert state using the optimized fast path
        try:
            obs = self._fast_obs_convert(state)
        except Exception as e:
            print(f"Conversion error: {e}")
            traceback.print_exc()
            raise GamaEnvironmentError(f"Failed to convert observation: {e}")
        
        return obs, info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step in the environment."""
        # Convert action to GAMA format using the optimized fast path
        gama_action = self._fast_act_encode(action)
        action_json = json.dumps(gama_action, separators=(',', ':'))
        # Execute step in GAMA
        step_data = self.gama_client.execute_step(self.experiment_id, action_json)
        
        # Extract and convert observation using the optimized fast path
        try:
            state = self._fast_obs_convert(step_data["State"])
        except Exception as e:
            print(f"Step conversion error: {e}")
            traceback.print_exc()
            raise GamaEnvironmentError(f"Failed to convert step observation: {e}")
        
        reward = step_data["Reward"]
        terminated = step_data["Terminated"]
        truncated = step_data["Truncated"]
        info = step_data["Info"]

        return state, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Render the environment (placeholder)."""
        print("Rendering the environment... (not implemented)")

    def close(self):
        """Close the environment and cleanup resources."""
        if hasattr(self, 'gama_client') and self.gama_client:
            try:
                self.gama_client.close()
            except Exception as e:
                print(f"Warning: Error closing GAMA environment: {e}")
            finally:
                self.gama_client = None