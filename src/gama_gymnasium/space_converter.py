"""
Utilities for converting between GAMA space definitions and Gymnasium spaces.
"""
from typing import Dict, Any, Union
import numpy as np

from gymnasium.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Text, Dict, Tuple, Sequence
from .exceptions import SpaceConversionError


class SpaceConverter:
    """
    Converter between GAMA space definitions and Gymnasium spaces.
    
    This class handles the mapping between the space format used by GAMA
    and the standard Gymnasium space types.
    """

    def map_to_space(self, space_map: dict[str, Any]):
        """
        Convert a GAMA space definition to a Gymnasium space.
        
        Args:
            space_map: Dictionary containing space definition from GAMA
            
        Returns:
            Gymnasium space object
            
        Raises:
            SpaceConversionError: If space type is unknown or invalid
        """
        if "type" not in space_map:
            raise SpaceConversionError("No type specified in space definition")
        
        space_type = space_map["type"]
        
        converters = {
            "Discrete": self._map_to_discrete,
            "Box": self._map_to_box,
            "MultiBinary": self._map_to_multi_binary,
            "MultiDiscrete": self._map_to_multi_discrete,
            "Text": self._map_to_text,
            "Dict": self._map_to_dict,
            "Tuple": self._map_to_tuple,
            "Sequence": self._map_to_sequence,
        }
        
        if space_type not in converters:
            raise SpaceConversionError(f"Unknown space type: {space_type}")
        
        try:
            return converters[space_type](space_map)
        except Exception as e:
            raise SpaceConversionError(f"Failed to convert {space_type} space: {e}")
        
    #---------- Fundamental Space Converters ----------#

    def _map_to_discrete(self, discrete_map: dict) -> Discrete:
        """Convert GAMA discrete space to Gymnasium Discrete."""
        n = discrete_map["n"]
        start = discrete_map.get("start", 0)
        return Discrete(n, start=start)

    def _map_to_box(self, box_map: dict) -> Box:
        """Convert GAMA box space to Gymnasium Box."""
        # Handle low bound
        low = box_map.get("low", -np.inf)
        if isinstance(low, list):
            low = np.array(self._replace_infinity(low))
        
        # Handle high bound
        high = box_map.get("high", np.inf)
        if isinstance(high, list):
            high = np.array(self._replace_infinity(high))
        
        # Handle shape
        shape = box_map.get("shape", None)
        
        # Handle dtype
        dtype_map = {
            "int": np.int64,
            "float": np.float64,
        }
        dtype = dtype_map.get(box_map.get("dtype"), np.float32)
        
        if isinstance(low, np.ndarray):
            low = low.astype(dtype)
        if isinstance(high, np.ndarray):
            high = high.astype(dtype)
        
        return Box(low=low, high=high, shape=shape, dtype=dtype)

    def _map_to_multi_binary(self, mb_map: dict) -> MultiBinary:
        """Convert GAMA multibinary space to Gymnasium MultiBinary."""
        n = mb_map["n"]
        if isinstance(n, list) and len(n) == 1:
            return MultiBinary(n[0])
        return MultiBinary(n)

    def _map_to_multi_discrete(self, md_map: dict) -> MultiDiscrete:
        """Convert GAMA multidiscrete space to Gymnasium MultiDiscrete."""
        nvec = md_map["nvec"]
        start = md_map.get("start", None)
        
        if start is not None:
            return MultiDiscrete(nvec, start=start)
        return MultiDiscrete(nvec)

    def _map_to_text(self, text_map: dict) -> Text:
        """Convert GAMA text space to Gymnasium Text."""
        min_length = text_map.get("min_length", 0)
        max_length = text_map.get("max_length", 1000)
        return Text(min_length=min_length, max_length=max_length)
    
    #---------- Composite Space Converters ----------#
    
    def _map_to_dict(self, dict_map: dict) -> Dict:
        """Convert GAMA dict space to Gymnasium Dict."""
        if "spaces" not in dict_map:
            raise SpaceConversionError("Dict space must contain 'spaces' field")
        
        spaces = {}
        for key, space_def in dict_map["spaces"].items():
            spaces[key] = self.map_to_space(space_def)
        
        return Dict(spaces)
    
    def _map_to_tuple(self, tuple_map: dict) -> Tuple:
        """Convert GAMA tuple space to Gymnasium Tuple."""
        if "spaces" not in tuple_map:
            raise SpaceConversionError("Tuple space must contain 'spaces' field")
        
        spaces = []
        for space_def in tuple_map["spaces"]:
            spaces.append(self.map_to_space(space_def))
        
        return Tuple(spaces)
    
    def _map_to_sequence(self, seq_map: dict) -> Sequence:
        """Convert GAMA sequence space to Gymnasium Sequence."""
        if "space" not in seq_map:
            raise SpaceConversionError("Sequence space must contain 'space' field")
        
        space = self.map_to_space(seq_map["space"])
        
        return Sequence(space)
    
    #---------- Observation Conversion Methods ----------#
    
    def convert_gama_to_gym_observation(self, space, gama_observation):
        """
        Convert a GAMA observation to a Gymnasium-compatible observation.
        Handles the problem with from_jsonable() expecting lists recursively.
        
        Args:
            space: The Gymnasium space to convert to
            gama_observation: The observation data from GAMA
            
        Returns:
            Gymnasium-compatible observation
        """
        return self._convert_space_value(space, gama_observation)
    
    def _convert_space_value(self, space, value):
        """Convert a value according to the Gymnasium space type."""
        
        if hasattr(space, 'spaces') and isinstance(space.spaces, dict):  # Dict space
            if not isinstance(value, dict):
                raise SpaceConversionError(f"Expected dict for Dict space, got {type(value)}")
            
            converted = {}
            for key, subspace in space.spaces.items():
                if key in value:
                    converted[key] = self._convert_space_value(subspace, value[key])
                else:
                    raise SpaceConversionError(f"Missing key '{key}' in observation")
            return converted
            
        elif hasattr(space, 'feature_space'):  # Sequence space
            if not isinstance(value, (list, tuple)):
                raise SpaceConversionError(f"Expected list/tuple for Sequence space, got {type(value)}")
            
            converted_items = []
            for item in value:
                converted_items.append(self._convert_space_value(space.feature_space, item))
            return tuple(converted_items)  # Sequence expects a tuple
            
        elif hasattr(space, 'spaces') and isinstance(space.spaces, (list, tuple)):  # Tuple space
            if not isinstance(value, (list, tuple)):
                raise SpaceConversionError(f"Expected list/tuple for Tuple space, got {type(value)}")
            
            converted_items = []
            for i, (subspace, item) in enumerate(zip(space.spaces, value)):
                converted_items.append(self._convert_space_value(subspace, item))
            return tuple(converted_items)
            
        else:  # Simple spaces (Box, Discrete, etc.)
            return self._convert_simple_space_value(space, value)
    
    def _convert_simple_space_value(self, space, value):
        """Convert value for simple spaces (Box, Discrete, etc.)."""
        
        # Box space conversion
        if hasattr(space, 'low') and hasattr(space, 'high'):  # Box space
            if isinstance(value, (list, tuple)):
                # Ensure correct dimension
                if hasattr(space, 'shape') and space.shape:
                    expected_size = np.prod(space.shape)
                    if len(value) > expected_size:
                        value = value[:expected_size]
                    elif len(value) < expected_size:
                        # Extend with zeros if necessary
                        value = list(value) + [0.0] * (expected_size - len(value))
                
                # Reshape according to expected shape
                array = np.array(value, dtype=space.dtype)
                if hasattr(space, 'shape') and space.shape:
                    array = array.reshape(space.shape)
                return array
            else:
                # Scalar value for 1D Box
                if hasattr(space, 'shape') and space.shape == (1,):
                    return np.array([value], dtype=space.dtype)
                return np.array(value, dtype=space.dtype)
        
        # Discrete space conversion
        elif hasattr(space, 'n'):  # Discrete space
            return int(value)
        
        # MultiBinary space conversion
        elif hasattr(space, 'n') and hasattr(space, 'shape'):  # MultiBinary space
            if isinstance(value, (list, tuple)):
                return np.array(value, dtype=np.int8)
            return value
        
        # MultiDiscrete space conversion
        elif hasattr(space, 'nvec'):  # MultiDiscrete space
            if isinstance(value, (list, tuple)):
                return np.array(value, dtype=np.int64)
            return value
        
        # Text space conversion
        elif hasattr(space, 'character_set'):  # Text space
            return str(value)
        
        # Default conversion
        else:
            return value

    #---------- Utility Methods ----------#

    def _replace_infinity(self, data: Union[list, Any]):
        """Replace string infinity values with float infinity."""
        if isinstance(data, list):
            return [self._replace_infinity(item) for item in data]
        elif data == "Infinity":
            return float('inf')
        elif data == "-Infinity":
            return float('-inf')
        return data