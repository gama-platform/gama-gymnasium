"""
Message Handler Module

This module provides utilities for processing and validating messages
exchanged between GAMA and the Gymnasium environment.
"""

import json
from typing import Any, Dict, Union

from ..utils.exceptions import MessageValidationError


class MessageHandler:
    """
    Handler for GAMA-Gymnasium message processing.
    
    This class provides methods for validating, processing, and converting
    messages between GAMA and Gymnasium formats.
    """
    
    @staticmethod
    def validate_step_data(data: Dict[str, Any]) -> None:
        """
        Validate step data received from GAMA.
        
        Args:
            data (dict): Step data from GAMA
            
        Raises:
            MessageValidationError: If data is invalid
        """
        required_fields = ["State", "Reward", "Terminated", "Truncated", "Info"]
        
        if not isinstance(data, dict):
            raise MessageValidationError("Step data must be a dictionary")
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise MessageValidationError(f"Missing required fields: {missing_fields}")
        
        # Validate field types
        if not isinstance(data["Reward"], (int, float)):
            raise MessageValidationError("Reward must be a number")
        
        if not isinstance(data["Terminated"], bool):
            raise MessageValidationError("Terminated must be a boolean")
        
        if not isinstance(data["Truncated"], bool):
            raise MessageValidationError("Truncated must be a boolean")
        
        if not isinstance(data["Info"], dict):
            raise MessageValidationError("Info must be a dictionary")
    
    @staticmethod
    def validate_space_definition(space_def: Dict[str, Any]) -> None:
        """
        Validate space definition received from GAMA.
        
        Args:
            space_def (dict): Space definition from GAMA
            
        Raises:
            MessageValidationError: If space definition is invalid
        """
        if not isinstance(space_def, dict):
            raise MessageValidationError("Space definition must be a dictionary")
        
        if "type" not in space_def:
            raise MessageValidationError("Space definition must contain 'type' field")
        
        space_type = space_def["type"]
        
        if space_type == "Box":
            required_fields = ["low", "high", "shape"]
            missing_fields = [field for field in required_fields if field not in space_def]
            if missing_fields:
                raise MessageValidationError(
                    f"Box space missing required fields: {missing_fields}"
                )
        
        elif space_type == "Discrete":
            if "n" not in space_def:
                raise MessageValidationError("Discrete space must contain 'n' field")
            
            if not isinstance(space_def["n"], int) or space_def["n"] <= 0:
                raise MessageValidationError("Discrete space 'n' must be a positive integer")
        
        elif space_type == "MultiDiscrete":
            if "nvec" not in space_def:
                raise MessageValidationError("MultiDiscrete space must contain 'nvec' field")
        
        else:
            raise MessageValidationError(f"Unsupported space type: {space_type}")
    
    @staticmethod
    def format_action_for_gama(action: Any) -> str:
        """
        Format an action for sending to GAMA.
        
        Args:
            action: Action from Gymnasium
            
        Returns:
            str: Formatted action string for GAMA
        """
        if isinstance(action, (list, tuple)):
            # Convert to GAMA list format
            action_str = "[" + ", ".join(str(a) for a in action) + "]"
        elif isinstance(action, dict):
            # Convert to GAMA map format
            pairs = [f'"{k}"::{v}' for k, v in action.items()]
            action_str = "[" + ", ".join(pairs) + "]"
        else:
            # Simple value
            action_str = str(action)
        
        return action_str
    
    @staticmethod
    def parse_gama_response(response: Dict[str, Any]) -> Any:
        """
        Parse a response from GAMA.
        
        Args:
            response (dict): Response from GAMA server
            
        Returns:
            Any: Parsed content
            
        Raises:
            MessageValidationError: If response is invalid
        """
        if not isinstance(response, dict):
            raise MessageValidationError("GAMA response must be a dictionary")
        
        if "type" not in response:
            raise MessageValidationError("GAMA response must contain 'type' field")
        
        if "content" not in response:
            raise MessageValidationError("GAMA response must contain 'content' field")
        
        return response["content"]
    
    @staticmethod
    def create_experiment_parameters(params: Dict[str, Any]) -> list[Dict[str, Any]]:
        """
        Create experiment parameters in GAMA format.
        
        Args:
            params (dict): Parameters as key-value pairs
            
        Returns:
            list: Parameters in GAMA format
        """
        gama_params = []
        for name, value in params.items():
            gama_params.append({
                "name": name,
                "value": value
            })
        return gama_params
    
    @staticmethod
    def serialize_for_json(obj: Any) -> Any:
        """
        Serialize an object for JSON transmission.
        
        Args:
            obj: Object to serialize
            
        Returns:
            Any: JSON-serializable object
        """
        if hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, '__dict__'):  # custom objects
            return obj.__dict__
        else:
            return obj
