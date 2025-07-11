"""
Space Conversion Utilities for GAMA-Gymnasium Integration

This module contains functions to convert GAMA space definitions to Gymnasium spaces.
It handles the mapping between GAMA's space format and Gymnasium's standard space types.
"""

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Text


def map_to_space(space_map: dict) -> Space:
    """
    Convert a GAMA space definition to a Gymnasium space.
    
    Args:
        space_map (dict): Space definition from GAMA
        
    Returns:
        Space: Corresponding Gymnasium space
    """
    if "type" not in space_map:
        print("No type specified in the space map, cannot map to space.")
        return None
    
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
    
    if space_type in space_converters:
        return space_converters[space_type](space_map)
    else:
        print(f"Unknown space type '{space_type}', cannot map to space.")
        return None


def map_to_box(box_map: dict) -> Box:
    """
    Convert GAMA Box space definition to Gymnasium Box space.
    
    Args:
        box_map (dict): Box space definition from GAMA
        
    Returns:
        Box: Gymnasium Box space
    """
    # Handle lower bounds
    if "low" in box_map:
        if isinstance(box_map["low"], list):
            low = np.array(replace_infinity(box_map["low"]))
        else:
            low = box_map["low"]
    else:
        low = -np.inf
    
    # Handle upper bounds
    if "high" in box_map:
        if isinstance(box_map["high"], list):
            high = np.array(replace_infinity(box_map["high"]))
        else:
            high = box_map["high"]
    else:
        high = np.inf

    # Handle shape
    shape = box_map.get("shape", None)

    # Handle data type
    if "dtype" in box_map:
        dtype_map = {
            "int": np.int64,
            "float": np.float64
        }
        dtype = dtype_map.get(box_map["dtype"], np.float32)
        if box_map["dtype"] not in dtype_map:
            print(f"Unknown dtype '{box_map['dtype']}' in box, defaulting to float32.")
    else:
        dtype = np.float32

    return Box(low=low, high=high, shape=shape, dtype=dtype)


def map_to_discrete(discrete_map: dict) -> Discrete:
    """
    Convert GAMA Discrete space definition to Gymnasium Discrete space.
    
    Args:
        discrete_map (dict): Discrete space definition from GAMA
        
    Returns:
        Discrete: Gymnasium Discrete space
    """
    n = discrete_map["n"]
    start = discrete_map.get("start", 0)
    
    if "start" in discrete_map:
        return Discrete(n, start=start)
    else:
        return Discrete(n)


def map_to_multi_binary(mb_map: dict) -> MultiBinary:
    """
    Convert GAMA MultiBinary space definition to Gymnasium MultiBinary space.
    
    Args:
        mb_map (dict): MultiBinary space definition from GAMA
        
    Returns:
        MultiBinary: Gymnasium MultiBinary space
    """
    n = mb_map["n"]
    
    # Handle both single integer and list of integers
    if len(n) == 1:
        return MultiBinary(n[0])
    else:
        return MultiBinary(n)


def map_to_multi_discrete(md_map: dict) -> MultiDiscrete:
    """
    Convert GAMA MultiDiscrete space definition to Gymnasium MultiDiscrete space.
    
    Args:
        md_map (dict): MultiDiscrete space definition from GAMA
        
    Returns:
        MultiDiscrete: Gymnasium MultiDiscrete space
    """
    nvec = md_map["nvec"]
    
    if "start" in md_map:
        start = md_map["start"]
        return MultiDiscrete(nvec, start=start)
    else:
        return MultiDiscrete(nvec)


def map_to_text(text_map: dict) -> Text:
    """
    Convert GAMA Text space definition to Gymnasium Text space.
    
    Args:
        text_map (dict): Text space definition from GAMA
        
    Returns:
        Text: Gymnasium Text space
    """
    min_length = text_map.get("min_length", 0)
    max_length = text_map.get("max_length", 1000)
    
    return Text(min_length=min_length, max_length=max_length)


def replace_infinity(data):
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
