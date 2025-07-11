"""
Space Conversion Utilities for GAMA-Gymnasium Integration

This module contains functions to convert GAMA space definitions to Gymnasium spaces.
It handles the mapping between GAMA's space format and Gymnasium's standard space types.
"""

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Text
from typing import Dict, Any, Union, List

from ..utils.exceptions import SpaceConversionError


def map_to_space(space_map: Dict[str, Any]) -> Space:
    """
    Convert a GAMA space definition to a Gymnasium space.
    
    Args:
        space_map (dict): Space definition from GAMA
        
    Returns:
        Space: Corresponding Gymnasium space
        
    Raises:
        SpaceConversionError: If space conversion fails
    """
    if not isinstance(space_map, dict):
        raise SpaceConversionError("Space map must be a dictionary")
    
    if "type" not in space_map:
        raise SpaceConversionError("No type specified in the space map")
    
    space_type = space_map["type"]
    
    # Map different space types to their respective converters
    space_converters = {
        "Discrete": map_to_discrete,
        "Box": map_to_box,
        "MultiBinary": map_to_multi_binary,
        "MultiDiscrete": map_to_multi_discrete,
        "Text": map_to_text,
        # TODO: Add support for complex spaces
        # "Tuple": map_to_tuple,
        # "Dict": map_to_dict,
        # "Sequence": map_to_sequence,
        # "Graph": map_to_graph,
        # "OneOf": map_to_one_of,
    }
    
    if space_type not in space_converters:
        raise SpaceConversionError(f"Unknown space type '{space_type}'")
    
    try:
        return space_converters[space_type](space_map)
    except Exception as e:
        raise SpaceConversionError(f"Failed to convert {space_type} space: {e}")


def map_to_box(box_map: Dict[str, Any]) -> Box:
    """
    Convert GAMA Box space definition to Gymnasium Box space.
    
    Args:
        box_map (dict): Box space definition from GAMA
        
    Returns:
        Box: Gymnasium Box space
        
    Raises:
        SpaceConversionError: If conversion fails
    """
    try:
        # Handle lower bounds
        if "low" in box_map:
            if isinstance(box_map["low"], list):
                low = np.array(replace_infinity(box_map["low"]))
            else:
                low = replace_infinity(box_map["low"])
        else:
            low = -np.inf
        
        # Handle upper bounds
        if "high" in box_map:
            if isinstance(box_map["high"], list):
                high = np.array(replace_infinity(box_map["high"]))
            else:
                high = replace_infinity(box_map["high"])
        else:
            high = np.inf

        # Handle shape
        shape = box_map.get("shape", None)
        if shape is not None and not isinstance(shape, (list, tuple)):
            shape = [shape]

        # Handle data type
        dtype = _parse_dtype(box_map.get("dtype", "float32"))

        return Box(low=low, high=high, shape=shape, dtype=dtype)
        
    except Exception as e:
        raise SpaceConversionError(f"Failed to create Box space: {e}")


def map_to_discrete(discrete_map: Dict[str, Any]) -> Discrete:
    """
    Convert GAMA Discrete space definition to Gymnasium Discrete space.
    
    Args:
        discrete_map (dict): Discrete space definition from GAMA
        
    Returns:
        Discrete: Gymnasium Discrete space
        
    Raises:
        SpaceConversionError: If conversion fails
    """
    if "n" not in discrete_map:
        raise SpaceConversionError("Discrete space must have 'n' parameter")
    
    n = discrete_map["n"]
    if not isinstance(n, int) or n <= 0:
        raise SpaceConversionError("Discrete space 'n' must be a positive integer")
    
    start = discrete_map.get("start", 0)
    
    try:
        if "start" in discrete_map:
            return Discrete(n, start=start)
        else:
            return Discrete(n)
    except Exception as e:
        raise SpaceConversionError(f"Failed to create Discrete space: {e}")


def map_to_multi_binary(mb_map: Dict[str, Any]) -> MultiBinary:
    """
    Convert GAMA MultiBinary space definition to Gymnasium MultiBinary space.
    
    Args:
        mb_map (dict): MultiBinary space definition from GAMA
        
    Returns:
        MultiBinary: Gymnasium MultiBinary space
        
    Raises:
        SpaceConversionError: If conversion fails
    """
    if "n" not in mb_map:
        raise SpaceConversionError("MultiBinary space must have 'n' parameter")
    
    n = mb_map["n"]
    
    try:
        # Handle both single integer and list of integers
        if isinstance(n, list):
            if len(n) == 1:
                return MultiBinary(n[0])
            else:
                return MultiBinary(n)
        else:
            return MultiBinary(n)
    except Exception as e:
        raise SpaceConversionError(f"Failed to create MultiBinary space: {e}")


def map_to_multi_discrete(md_map: Dict[str, Any]) -> MultiDiscrete:
    """
    Convert GAMA MultiDiscrete space definition to Gymnasium MultiDiscrete space.
    
    Args:
        md_map (dict): MultiDiscrete space definition from GAMA
        
    Returns:
        MultiDiscrete: Gymnasium MultiDiscrete space
        
    Raises:
        SpaceConversionError: If conversion fails
    """
    if "nvec" not in md_map:
        raise SpaceConversionError("MultiDiscrete space must have 'nvec' parameter")
    
    nvec = md_map["nvec"]
    
    try:
        if "start" in md_map:
            start = md_map["start"]
            return MultiDiscrete(nvec, start=start)
        else:
            return MultiDiscrete(nvec)
    except Exception as e:
        raise SpaceConversionError(f"Failed to create MultiDiscrete space: {e}")


def map_to_text(text_map: Dict[str, Any]) -> Text:
    """
    Convert GAMA Text space definition to Gymnasium Text space.
    
    Args:
        text_map (dict): Text space definition from GAMA
        
    Returns:
        Text: Gymnasium Text space
        
    Raises:
        SpaceConversionError: If conversion fails
    """
    min_length = text_map.get("min_length", 0)
    max_length = text_map.get("max_length", 1000)
    
    if not isinstance(min_length, int) or min_length < 0:
        raise SpaceConversionError("Text space 'min_length' must be a non-negative integer")
    
    if not isinstance(max_length, int) or max_length < min_length:
        raise SpaceConversionError("Text space 'max_length' must be >= min_length")
    
    try:
        return Text(min_length=min_length, max_length=max_length)
    except Exception as e:
        raise SpaceConversionError(f"Failed to create Text space: {e}")


def replace_infinity(data: Union[Any, List[Any]]) -> Union[Any, List[Any]]:
    """
    Replace string representations of infinity with Python float infinity.
    
    This function recursively processes data structures to convert
    GAMA's string representations of infinity to Python's float infinity.
    
    Args:
        data: Data that may contain infinity strings
        
    Returns:
        Data with infinity strings replaced by float infinity
    """
    if isinstance(data, list):
        return [replace_infinity(item) for item in data]
    elif data == "Infinity":
        return float('inf')
    elif data == "-Infinity":
        return float('-inf')
    else:
        return data


def _parse_dtype(dtype_str: str) -> np.dtype:
    """
    Parse dtype string to numpy dtype.
    
    Args:
        dtype_str (str): String representation of dtype
        
    Returns:
        np.dtype: Corresponding numpy dtype
    """
    dtype_map = {
        "int": np.int64,
        "int32": np.int32,
        "int64": np.int64,
        "float": np.float64,
        "float32": np.float32,
        "float64": np.float64,
        "bool": np.bool_,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64
    }
    
    if dtype_str in dtype_map:
        return dtype_map[dtype_str]
    else:
        # Try to parse as numpy dtype directly
        try:
            return np.dtype(dtype_str)
        except TypeError:
            # Default to float32 if parsing fails
            return np.float32


def validate_space_compatibility(gama_space: Dict[str, Any], gym_space: Space) -> bool:
    """
    Validate that a GAMA space definition is compatible with a Gymnasium space.
    
    Args:
        gama_space (dict): GAMA space definition
        gym_space (Space): Gymnasium space
        
    Returns:
        bool: True if compatible, False otherwise
    """
    try:
        converted_space = map_to_space(gama_space)
        return type(converted_space) == type(gym_space)
    except SpaceConversionError:
        return False
