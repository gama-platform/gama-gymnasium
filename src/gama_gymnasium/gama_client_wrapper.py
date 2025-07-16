"""
Wrapper around GamaSyncClient to provide high-level operations for the environment.
"""
import time
from typing import Any, Dict

from gama_client.sync_client import GamaSyncClient
from gama_client.message_types import MessageTypes

from .exceptions import GamaEnvironmentError, GamaConnectionError, GamaCommandError


class GamaClientWrapper:
    """
    High-level wrapper around GamaSyncClient for environment operations.
    
    This class encapsulates all direct communication with the GAMA server
    and provides clean, typed methods for environment operations.
    """

    def __init__(self, ip_address: str = None, port: int = 6868):
        """
        Initialize the GAMA client wrapper.
        
        Args:
            ip_address: IP address of GAMA server
            port: Port of GAMA server
        """
        self.ip_address = ip_address
        self.port = port
        self.client = None
        self.experiment_id = None
        
        # Connect to GAMA server
        self._connect()

    def _connect(self):
        """Establish connection to GAMA server."""
        try:
            self.client = GamaSyncClient(
                self.ip_address, 
                self.port, 
                self._async_command_handler,
                self._server_message_handler
            )
            self.client.connect()
        except Exception as e:
            raise GamaConnectionError(f"Failed to connect to GAMA server: {e}")

    async def _async_command_handler(self, message: dict):
        """Handle async command responses."""
        print("Async command response:", message)

    async def _server_message_handler(self, message: dict):
        """Handle server messages."""
        print("Server message:", message)

    def load_experiment(self, gaml_path: str, experiment_name: str, 
                       parameters: list = None) -> str:
        """
        Load a GAMA experiment.
        
        Args:
            gaml_path: Path to the .gaml file
            experiment_name: Name of the experiment
            parameters: Optional experiment parameters
            
        Returns:
            Experiment ID
        """
        response = self.client.load(
            gaml_path, 
            experiment_name, 
            console=False, 
            runtime=True, 
            parameters=parameters or []
        )
        
        if response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise GamaCommandError(f"Failed to load experiment: {response}")
        
        experiment_id = response["content"]
        
        # Wait for initialization if using port 1000
        if self.port == 1000:
            time.sleep(8)
            
        return experiment_id

    def get_observation_space(self, experiment_id: str) -> Dict:
        """Get observation space definition from GAMA."""
        return self._execute_expression(
            experiment_id, 
            r"GymAgent[0].observation_space"
        )

    def get_action_space(self, experiment_id: str) -> Dict:
        """Get action space definition from GAMA."""
        return self._execute_expression(
            experiment_id, 
            r"GymAgent[0].action_space"
        )

    def reset_experiment(self, experiment_id: str, seed: int = None):
        """Reset the GAMA experiment."""
        # Reload experiment
        response = self.client.reload(experiment_id)
        if response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise GamaCommandError(f"Failed to reload experiment: {response}")
        
        # Set seed
        if seed is not None:
            self._execute_expression(experiment_id, f"seed <- {seed};")
        else:
            import numpy as np
            random_seed = np.random.random()
            self._execute_expression(experiment_id, f"seed <- {random_seed};")

    def get_state(self, experiment_id: str):
        """Get current state from GAMA."""
        return self._execute_expression(experiment_id, r"GymAgent[0].state")

    def get_info(self, experiment_id: str):
        """Get current info from GAMA."""
        return self._execute_expression(experiment_id, r"GymAgent[0].info")

    def execute_step(self, experiment_id: str, action) -> Dict:
        """
        Execute one step in GAMA with the given action.
        
        Args:
            experiment_id: GAMA experiment ID
            action: Action to execute
            
        Returns:
            Dictionary with State, Reward, Terminated, Truncated, Info
        """
        # Set action
        self._execute_expression(
            experiment_id, 
            f"GymAgent[0].next_action <- {action};"
        )
        
        # Execute step
        response = self.client.step(experiment_id, sync=True)
        if response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise GamaCommandError(f"Failed to execute step: {response}")
        
        # Get results
        data = self._execute_expression(experiment_id, r"GymAgent[0].data")
        return data

    def _execute_expression(self, experiment_id: str, expression: str):
        """Execute a GAMA expression and return the result."""
        response = self.client.expression(experiment_id, expression)
        if response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise GamaCommandError(f"Failed to execute expression '{expression}': {response}")
        return response["content"]

    def close(self):
        """Close the connection to GAMA server."""
        if self.client:
            try:
                self.client.close_connection()
            except Exception as e:
                # Log the error but don't raise it (graceful cleanup)
                print(f"Warning: Error closing GAMA connection: {e}")
            finally:
                self.client = None