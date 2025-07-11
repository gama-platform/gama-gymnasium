"""
Space Validation Utilities for GAMA-Gymnasium Integration

This module provides validation functions for space definitions
and space compatibility checks.
"""

from typing import Dict, Any, List, Union
from gymnasium import Space
from gymnasium.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Text

from ..utils.exceptions import SpaceConversionError


def validate_space_definition(space_def: Dict[str, Any]) -> None:
    """
    Validate a space definition from GAMA.
    
    Args:
        space_def (dict): Space definition to validate
        
    Raises:
        SpaceConversionError: If space definition is invalid
    """
    if not isinstance(space_def, dict):
        raise SpaceConversionError("Space definition must be a dictionary")
    
    if "type" not in space_def:
        raise SpaceConversionError("Space definition must contain 'type' field")
    
    space_type = space_def["type"]
    
    # Validate based on space type
    validators = {
        "Discrete": _validate_discrete_space,
        "Box": _validate_box_space,
        "MultiBinary": _validate_multi_binary_space,
        "MultiDiscrete": _validate_multi_discrete_space,
        "Text": _validate_text_space
    }
    
    if space_type in validators:
        validators[space_type](space_def)
    else:
        raise SpaceConversionError(f"Unknown space type: {space_type}")


def _validate_discrete_space(space_def: Dict[str, Any]) -> None:
    """Validate Discrete space definition."""
    if "n" not in space_def:
        raise SpaceConversionError("Discrete space must contain 'n' field")
    
    n = space_def["n"]
    if not isinstance(n, int) or n <= 0:
        raise SpaceConversionError("Discrete space 'n' must be a positive integer")
    
    if "start" in space_def:
        start = space_def["start"]
        if not isinstance(start, int):
            raise SpaceConversionError("Discrete space 'start' must be an integer")


def _validate_box_space(space_def: Dict[str, Any]) -> None:
    """Validate Box space definition."""
    # Check for required fields
    if "shape" not in space_def and "low" not in space_def and "high" not in space_def:
        raise SpaceConversionError("Box space must specify at least one of: shape, low, high")
    
    # Validate low bounds
    if "low" in space_def:
        low = space_def["low"]
        if not isinstance(low, (int, float, list)):
            raise SpaceConversionError("Box space 'low' must be a number or list of numbers")
        if isinstance(low, list) and not all(isinstance(x, (int, float, str)) for x in low):
            raise SpaceConversionError("Box space 'low' list must contain only numbers or 'Infinity'")
    
    # Validate high bounds
    if "high" in space_def:
        high = space_def["high"]
        if not isinstance(high, (int, float, list)):
            raise SpaceConversionError("Box space 'high' must be a number or list of numbers")
        if isinstance(high, list) and not all(isinstance(x, (int, float, str)) for x in high):
            raise SpaceConversionError("Box space 'high' list must contain only numbers or 'Infinity'")
    
    # Validate shape
    if "shape" in space_def:
        shape = space_def["shape"]
        if not isinstance(shape, (int, list, tuple)):
            raise SpaceConversionError("Box space 'shape' must be an integer or list of integers")
        if isinstance(shape, (list, tuple)) and not all(isinstance(x, int) and x > 0 for x in shape):
            raise SpaceConversionError("Box space 'shape' must contain positive integers")
    
    # Validate dtype
    if "dtype" in space_def:
        dtype = space_def["dtype"]
        if not isinstance(dtype, str):
            raise SpaceConversionError("Box space 'dtype' must be a string")


def _validate_multi_binary_space(space_def: Dict[str, Any]) -> None:
    """Validate MultiBinary space definition."""
    if "n" not in space_def:
        raise SpaceConversionError("MultiBinary space must contain 'n' field")
    
    n = space_def["n"]
    if isinstance(n, int):
        if n <= 0:
            raise SpaceConversionError("MultiBinary space 'n' must be positive")
    elif isinstance(n, list):
        if not all(isinstance(x, int) and x > 0 for x in n):
            raise SpaceConversionError("MultiBinary space 'n' list must contain positive integers")
    else:
        raise SpaceConversionError("MultiBinary space 'n' must be an integer or list of integers")


def _validate_multi_discrete_space(space_def: Dict[str, Any]) -> None:
    """Validate MultiDiscrete space definition."""
    if "nvec" not in space_def:
        raise SpaceConversionError("MultiDiscrete space must contain 'nvec' field")
    
    nvec = space_def["nvec"]
    if not isinstance(nvec, list):
        raise SpaceConversionError("MultiDiscrete space 'nvec' must be a list")
    
    if not all(isinstance(x, int) and x > 0 for x in nvec):
        raise SpaceConversionError("MultiDiscrete space 'nvec' must contain positive integers")
    
    if "start" in space_def:
        start = space_def["start"]
        if not isinstance(start, list):
            raise SpaceConversionError("MultiDiscrete space 'start' must be a list")
        if len(start) != len(nvec):
            raise SpaceConversionError("MultiDiscrete space 'start' must have same length as 'nvec'")
        if not all(isinstance(x, int) for x in start):
            raise SpaceConversionError("MultiDiscrete space 'start' must contain integers")


def _validate_text_space(space_def: Dict[str, Any]) -> None:
    """Validate Text space definition."""
    if "min_length" in space_def:
        min_length = space_def["min_length"]
        if not isinstance(min_length, int) or min_length < 0:
            raise SpaceConversionError("Text space 'min_length' must be a non-negative integer")
    
    if "max_length" in space_def:
        max_length = space_def["max_length"]
        if not isinstance(max_length, int) or max_length < 0:
            raise SpaceConversionError("Text space 'max_length' must be a non-negative integer")
    
    if "min_length" in space_def and "max_length" in space_def:
        if space_def["min_length"] > space_def["max_length"]:
            raise SpaceConversionError("Text space 'min_length' must be <= 'max_length'")


def validate_action_in_space(action: Any, space: Space) -> bool:
    """
    Validate that an action is valid for the given space.
    
    Args:
        action: Action to validate
        space (Space): Action space
        
    Returns:
        bool: True if action is valid, False otherwise
    """
    try:
        return space.contains(action)
    except:
        return False


def validate_observation_in_space(observation: Any, space: Space) -> bool:
    """
    Validate that an observation is valid for the given space.
    
    Args:
        observation: Observation to validate
        space (Space): Observation space
        
    Returns:
        bool: True if observation is valid, False otherwise
    """
    try:
        return space.contains(observation)
    except:
        return False


def get_space_info(space: Space) -> Dict[str, Any]:
    """
    Get information about a Gymnasium space.
    
    Args:
        space (Space): Space to analyze
        
    Returns:
        dict: Information about the space
    """
    info = {
        "type": type(space).__name__,
        "shape": getattr(space, 'shape', None),
        "dtype": getattr(space, 'dtype', None)
    }
    
    if isinstance(space, Discrete):
        info.update({
            "n": space.n,
            "start": getattr(space, 'start', 0)
        })
    elif isinstance(space, Box):
        info.update({
            "low": space.low.tolist() if hasattr(space.low, 'tolist') else space.low,
            "high": space.high.tolist() if hasattr(space.high, 'tolist') else space.high,
            "bounded_below": space.bounded_below.tolist() if hasattr(space.bounded_below, 'tolist') else space.bounded_below,
            "bounded_above": space.bounded_above.tolist() if hasattr(space.bounded_above, 'tolist') else space.bounded_above
        })
    elif isinstance(space, MultiBinary):
        info.update({
            "n": space.n.tolist() if hasattr(space.n, 'tolist') else space.n
        })
    elif isinstance(space, MultiDiscrete):
        info.update({
            "nvec": space.nvec.tolist() if hasattr(space.nvec, 'tolist') else space.nvec,
            "start": getattr(space, 'start', None)
        })
    elif isinstance(space, Text):
        info.update({
            "min_length": space.min_length,
            "max_length": space.max_length
        })
    
    return info
