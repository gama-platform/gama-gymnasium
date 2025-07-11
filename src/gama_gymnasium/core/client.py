"""
GAMA Client Module

This module provides a high-level interface for communicating with GAMA servers.
It handles connection management, message sending/receiving, and error handling.
"""

import asyncio
from typing import Any, Callable, Optional

from gama_client.sync_client import GamaSyncClient
from gama_client.message_types import MessageTypes


async def async_command_answer_handler(message: dict) -> None:
    """
    Default handler for asynchronous command responses from GAMA server.
    
    Args:
        message (dict): The response message from GAMA server
    """
    print("Received async command response:", message)


async def gama_server_message_handler(message: dict) -> None:
    """
    Default handler for unsolicited messages from GAMA server.
    
    Args:
        message (dict): The message from GAMA server
    """
    print("Received unsolicited GAMA server message:", message)


class GamaClient:
    """
    High-level client for GAMA server communication.
    
    This class wraps the low-level GamaSyncClient and provides a more
    convenient interface for common operations like loading experiments,
    executing expressions, and managing simulation steps.
    
    Attributes:
        client (GamaSyncClient): Underlying GAMA client
        is_connected (bool): Connection status
    """
    
    def __init__(
        self,
        ip_address: Optional[str] = "localhost",
        port: int = 6868,
        async_handler: Optional[Callable] = None,
        message_handler: Optional[Callable] = None
    ):
        """
        Initialize the GAMA client.
        
        Args:
            ip_address (str, optional): IP address of GAMA server
            port (int): Port number for GAMA server communication
            async_handler (Callable, optional): Handler for async responses
            message_handler (Callable, optional): Handler for server messages
        """
        self.client = GamaSyncClient(
            ip_address, 
            port,
            async_handler or async_command_answer_handler,
            message_handler or gama_server_message_handler
        )
        self.is_connected = False
    
    def connect(self) -> None:
        """Establish connection to GAMA server."""
        self.client.connect()
        self.is_connected = True
    
    def disconnect(self) -> None:
        """Close connection to GAMA server."""
        if self.is_connected:
            self.client.close_connection()
            self.is_connected = False
    
    def load_experiment(
        self,
        gaml_file_path: str,
        experiment_name: str,
        parameters: Optional[list[dict[str, Any]]] = None,
        console: bool = False,
        runtime: bool = True
    ) -> str:
        """
        Load a GAML experiment on the GAMA server.
        
        Args:
            gaml_file_path (str): Path to the GAML model file
            experiment_name (str): Name of the experiment to load
            parameters (list, optional): Experiment parameters
            console (bool): Enable console output
            runtime (bool): Enable runtime mode
            
        Returns:
            str: Experiment ID
            
        Raises:
            Exception: If loading fails
        """
        if not self.is_connected:
            raise RuntimeError("Client not connected to GAMA server")
        
        parameters = parameters or []
        response = self.client.load(
            gaml_file_path, experiment_name,
            console=console, runtime=runtime, parameters=parameters
        )
        
        if response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception(f"Failed to load experiment: {response}")
        
        return response["content"]
    
    def execute_expression(self, experiment_id: str, expression: str) -> Any:
        """
        Execute a GAML expression in the given experiment.
        
        Args:
            experiment_id (str): ID of the experiment
            expression (str): GAML expression to execute
            
        Returns:
            Any: Result of the expression
            
        Raises:
            Exception: If execution fails
        """
        if not self.is_connected:
            raise RuntimeError("Client not connected to GAMA server")
        
        response = self.client.expression(experiment_id, expression)
        
        if response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception(f"Failed to execute expression '{expression}': {response}")
        
        return response["content"]
    
    def step_simulation(self, experiment_id: str, sync: bool = True) -> None:
        """
        Execute one simulation step.
        
        Args:
            experiment_id (str): ID of the experiment
            sync (bool): Whether to wait for step completion
            
        Raises:
            Exception: If step execution fails
        """
        if not self.is_connected:
            raise RuntimeError("Client not connected to GAMA server")
        
        response = self.client.step(experiment_id, sync=sync)
        
        if response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception(f"Failed to execute simulation step: {response}")
    
    def stop_experiment(self, experiment_id: str) -> None:
        """
        Stop a running experiment.
        
        Args:
            experiment_id (str): ID of the experiment to stop
            
        Raises:
            Exception: If stopping fails
        """
        if not self.is_connected:
            raise RuntimeError("Client not connected to GAMA server")
        
        response = self.client.stop(experiment_id)
        
        if response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception(f"Failed to stop experiment: {response}")
    
    def reload_experiment(
        self,
        experiment_id: str,
        gaml_file_path: str,
        experiment_name: str,
        parameters: Optional[list[dict[str, Any]]] = None
    ) -> str:
        """
        Reload an experiment (stop and load again).
        
        Args:
            experiment_id (str): Current experiment ID
            gaml_file_path (str): Path to the GAML model file
            experiment_name (str): Name of the experiment
            parameters (list, optional): Experiment parameters
            
        Returns:
            str: New experiment ID
        """
        self.stop_experiment(experiment_id)
        return self.load_experiment(
            gaml_file_path, experiment_name, parameters
        )
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
