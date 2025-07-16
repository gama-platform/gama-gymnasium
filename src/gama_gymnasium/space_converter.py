"""
Utilities for converting between GAMA space definitions and Gymnasium spaces.
"""
from typing import Dict, Any, Union
import numpy as np

from gymnasium.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Text
from .exceptions import SpaceConversionError


class SpaceConverter:
    """
    Converter between GAMA space definitions and Gymnasium spaces.
    
    This class handles the mapping between the space format used by GAMA
    and the standard Gymnasium space types.
    """

    def map_to_space(self, space_map: Dict[str, Any]):
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
        }
        
        if space_type not in converters:
            raise SpaceConversionError(f"Unknown space type: {space_type}")
        
        try:
            return converters[space_type](space_map)
        except Exception as e:
            raise SpaceConversionError(f"Failed to convert {space_type} space: {e}")

    def _map_to_discrete(self, discrete_map: Dict) -> Discrete:
        """Convert GAMA discrete space to Gymnasium Discrete."""
        n = discrete_map["n"]
        start = discrete_map.get("start", 0)
        return Discrete(n, start=start)

    def _map_to_box(self, box_map: Dict) -> Box:
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

    def _map_to_multi_binary(self, mb_map: Dict) -> MultiBinary:
        """Convert GAMA multibinary space to Gymnasium MultiBinary."""
        n = mb_map["n"]
        if isinstance(n, list) and len(n) == 1:
            return MultiBinary(n[0])
        return MultiBinary(n)

    def _map_to_multi_discrete(self, md_map: Dict) -> MultiDiscrete:
        """Convert GAMA multidiscrete space to Gymnasium MultiDiscrete."""
        nvec = md_map["nvec"]
        start = md_map.get("start", None)
        
        if start is not None:
            return MultiDiscrete(nvec, start=start)
        return MultiDiscrete(nvec)

    def _map_to_text(self, text_map: Dict) -> Text:
        """Convert GAMA text space to Gymnasium Text."""
        min_length = text_map.get("min_length", 0)
        max_length = text_map.get("max_length", 1000)
        return Text(min_length=min_length, max_length=max_length)

    def _replace_infinity(self, data: Union[list, Any]):
        """Replace string infinity values with float infinity."""
        if isinstance(data, list):
            return [self._replace_infinity(item) for item in data]
        elif data == "Infinity":
            return float('inf')
        elif data == "-Infinity":
            return float('-inf')
        return data