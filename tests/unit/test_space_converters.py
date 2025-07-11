"""
Tests for space conversion utilities.
"""

import pytest
import numpy as np
from gymnasium.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Text

from gama_gymnasium.spaces.converters import (
    map_to_space, map_to_box, map_to_discrete, 
    replace_infinity, validate_space_compatibility
)
from gama_gymnasium.utils.exceptions import SpaceConversionError


class TestSpaceConverters:
    """Test space conversion functions."""
    
    def test_map_to_discrete_valid(self):
        """Test converting valid discrete space."""
        space_def = {"type": "Discrete", "n": 4}
        space = map_to_space(space_def)
        
        assert isinstance(space, Discrete)
        assert space.n == 4
    
    def test_map_to_discrete_with_start(self):
        """Test discrete space with start parameter."""
        space_def = {"type": "Discrete", "n": 4, "start": 1}
        space = map_to_space(space_def)
        
        assert isinstance(space, Discrete)
        assert space.n == 4
        assert space.start == 1
    
    def test_map_to_box_simple(self):
        """Test converting simple box space."""
        space_def = {
            "type": "Box",
            "low": [0.0, 0.0],
            "high": [1.0, 1.0],
            "shape": [2]
        }
        space = map_to_space(space_def)
        
        assert isinstance(space, Box)
        assert np.array_equal(space.low, [0.0, 0.0])
        assert np.array_equal(space.high, [1.0, 1.0])
        assert space.shape == (2,)
    
    def test_map_to_box_with_infinity(self):
        """Test box space with infinity values."""
        space_def = {
            "type": "Box",
            "low": ["-Infinity", 0.0],
            "high": ["Infinity", 1.0],
            "shape": [2]
        }
        space = map_to_space(space_def)
        
        assert isinstance(space, Box)
        assert space.low[0] == float('-inf')
        assert space.high[0] == float('inf')
        assert space.low[1] == 0.0
        assert space.high[1] == 1.0
    
    def test_map_to_box_with_infinity_strings(self):
        """Test box space conversion with infinity strings."""
        space_def = {
            "type": "Box",
            "low": ["-Infinity", 0.0],
            "high": ["Infinity", 10.0],
            "shape": [2]
        }
        space = map_to_space(space_def)
        
        assert isinstance(space, Box)
        assert space.low[0] == float('-inf')
        assert space.high[0] == float('inf')
        assert space.low[1] == 0.0
        assert space.high[1] == 10.0
    
    def test_map_to_text_with_custom_lengths(self):
        """Test text space with custom min/max lengths."""
        space_def = {
            "type": "Text",
            "min_length": 5,
            "max_length": 50
        }
        space = map_to_space(space_def)
        
        assert isinstance(space, Text)
        assert space.min_length == 5
        assert space.max_length == 50
    
    def test_map_to_multi_binary_integer_n(self):
        """Test MultiBinary space with integer n."""
        space_def = {"type": "MultiBinary", "n": 8}
        space = map_to_space(space_def)
        
        assert isinstance(space, MultiBinary)
        assert space.n == 8
    
    def test_error_handling_invalid_discrete_n(self):
        """Test error handling for invalid discrete n."""
        space_def = {"type": "Discrete", "n": 0}
        
        with pytest.raises(SpaceConversionError, match="positive integer"):
            map_to_space(space_def)
    
    def test_error_handling_missing_nvec(self):
        """Test error handling for missing nvec in MultiDiscrete."""
        space_def = {"type": "MultiDiscrete"}
        
        with pytest.raises(SpaceConversionError, match="must have 'nvec'"):
            map_to_space(space_def)
    
    def test_box_space_with_different_dtypes(self):
        """Test box space conversion with different data types."""
        test_cases = [
            ("int32", np.int32),
            ("int64", np.int64),
            ("float32", np.float32),
            ("float64", np.float64),
            ("uint8", np.uint8)
        ]
        
        for dtype_str, expected_dtype in test_cases:
            space_def = {
                "type": "Box",
                "low": 0,
                "high": 1,
                "shape": [2],
                "dtype": dtype_str
            }
            space = map_to_space(space_def)
            assert space.dtype == expected_dtype
    
    def test_multi_discrete_start_length_mismatch(self):
        """Test MultiDiscrete with mismatched start and nvec lengths."""
        space_def = {
            "type": "MultiDiscrete",
            "nvec": [3, 4, 5],
            "start": [1, 0]  # Wrong length
        }
        
        # This should still work with our current implementation
        # but let's test that it doesn't crash
        space = map_to_space(space_def)
        assert isinstance(space, MultiDiscrete)
    
    def test_replace_infinity_edge_cases(self):
        """Test replace_infinity with edge cases."""
        # Empty list
        assert replace_infinity([]) == []
        
        # None values
        assert replace_infinity(None) == None
        
        # Mixed types
        mixed_data = [1, "Infinity", None, "-Infinity", "normal"]
        expected = [1, float('inf'), None, float('-inf'), "normal"]
        assert replace_infinity(mixed_data) == expected
    
    def test_invalid_space_type(self):
        """Test error handling for invalid space type."""
        space_def = {"type": "InvalidType", "n": 4}
        
        with pytest.raises(SpaceConversionError):
            map_to_space(space_def)
    
    def test_missing_type(self):
        """Test error handling for missing type."""
        space_def = {"n": 4}
        
        with pytest.raises(SpaceConversionError):
            map_to_space(space_def)
    
    def test_discrete_missing_n(self):
        """Test error handling for discrete space missing n."""
        space_def = {"type": "Discrete"}
        
        with pytest.raises(SpaceConversionError):
            map_to_space(space_def)
    
    def test_discrete_invalid_n(self):
        """Test error handling for invalid n value."""
        space_def = {"type": "Discrete", "n": -1}
        
        with pytest.raises(SpaceConversionError):
            map_to_space(space_def)


class TestSpaceValidation:
    """Test space validation functions."""
    
    def test_validate_compatibility_true(self):
        """Test compatible spaces return True."""
        gama_space = {"type": "Discrete", "n": 4}
        gym_space = Discrete(4)
        
        assert validate_space_compatibility(gama_space, gym_space) == True
    
    def test_validate_compatibility_false(self):
        """Test incompatible spaces return False."""
        gama_space = {"type": "Discrete", "n": 4}
        gym_space = Box(low=0, high=1, shape=(2,))
        
        assert validate_space_compatibility(gama_space, gym_space) == False
    
    def test_validate_compatibility_invalid_gama(self):
        """Test invalid GAMA space returns False."""
        gama_space = {"type": "InvalidType"}
        gym_space = Discrete(4)
        
        assert validate_space_compatibility(gama_space, gym_space) == False
