"""
Tests for space validation functionality.
"""

import pytest
import numpy as np
from gymnasium.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Text

from gama_gymnasium.spaces.validators import (
    validate_space_definition, validate_action_in_space, 
    validate_observation_in_space, get_space_info
)
from gama_gymnasium.utils.exceptions import SpaceConversionError


class TestSpaceValidators:
    """Test space validation functionality."""
    
    def test_validate_discrete_space_valid(self):
        """Test validation of valid discrete space."""
        space_def = {
            "type": "Discrete",
            "n": 4
        }
        
        # Should not raise any exception
        validate_space_definition(space_def)
    
    def test_validate_discrete_space_with_start(self):
        """Test validation of discrete space with start parameter."""
        space_def = {
            "type": "Discrete",
            "n": 5,
            "start": 1
        }
        
        # Should not raise any exception
        validate_space_definition(space_def)
    
    def test_validate_discrete_space_missing_n(self):
        """Test validation of discrete space missing n."""
        space_def = {
            "type": "Discrete"
        }
        
        with pytest.raises(SpaceConversionError, match="must contain 'n' field"):
            validate_space_definition(space_def)
    
    def test_validate_discrete_space_invalid_n(self):
        """Test validation of discrete space with invalid n."""
        space_def = {
            "type": "Discrete",
            "n": 0
        }
        
        with pytest.raises(SpaceConversionError, match="must be a positive integer"):
            validate_space_definition(space_def)
    
    def test_validate_discrete_space_invalid_start(self):
        """Test validation of discrete space with invalid start."""
        space_def = {
            "type": "Discrete",
            "n": 4,
            "start": "invalid"
        }
        
        with pytest.raises(SpaceConversionError, match="must be an integer"):
            validate_space_definition(space_def)
    
    def test_validate_box_space_valid(self):
        """Test validation of valid box space."""
        space_def = {
            "type": "Box",
            "low": [0.0, -1.0],
            "high": [1.0, 1.0],
            "shape": [2]
        }
        
        # Should not raise any exception
        validate_space_definition(space_def)
    
    def test_validate_box_space_minimal(self):
        """Test validation of minimal box space."""
        space_def = {
            "type": "Box",
            "low": 0.0
        }
        
        # Should not raise any exception
        validate_space_definition(space_def)
    
    def test_validate_box_space_no_fields(self):
        """Test validation of box space with no required fields."""
        space_def = {
            "type": "Box"
        }
        
        with pytest.raises(SpaceConversionError, match="must specify at least one"):
            validate_space_definition(space_def)
    
    def test_validate_box_space_invalid_low_type(self):
        """Test validation of box space with invalid low type."""
        space_def = {
            "type": "Box",
            "low": "invalid",
            "high": 1.0
        }
        
        with pytest.raises(SpaceConversionError, match="must be a number or list"):
            validate_space_definition(space_def)
    
    def test_validate_box_space_invalid_low_list(self):
        """Test validation of box space with invalid low list."""
        space_def = {
            "type": "Box",
            "low": [0.0, "invalid"],
            "high": [1.0, 1.0]
        }
        
        with pytest.raises(SpaceConversionError, match="must contain only numbers"):
            validate_space_definition(space_def)
    
    def test_validate_box_space_invalid_shape(self):
        """Test validation of box space with invalid shape."""
        space_def = {
            "type": "Box",
            "low": 0.0,
            "high": 1.0,
            "shape": [0, -1]  # Invalid dimensions
        }
        
        with pytest.raises(SpaceConversionError, match="must contain positive integers"):
            validate_space_definition(space_def)
    
    def test_validate_multi_binary_space_valid(self):
        """Test validation of valid MultiBinary space."""
        space_def = {
            "type": "MultiBinary",
            "n": 8
        }
        
        # Should not raise any exception
        validate_space_definition(space_def)
    
    def test_validate_multi_binary_space_list(self):
        """Test validation of MultiBinary space with list n."""
        space_def = {
            "type": "MultiBinary",
            "n": [2, 3, 4]
        }
        
        # Should not raise any exception
        validate_space_definition(space_def)
    
    def test_validate_multi_binary_space_missing_n(self):
        """Test validation of MultiBinary space missing n."""
        space_def = {
            "type": "MultiBinary"
        }
        
        with pytest.raises(SpaceConversionError, match="must contain 'n' field"):
            validate_space_definition(space_def)
    
    def test_validate_multi_binary_space_invalid_n(self):
        """Test validation of MultiBinary space with invalid n."""
        space_def = {
            "type": "MultiBinary",
            "n": 0
        }
        
        with pytest.raises(SpaceConversionError, match="must be positive"):
            validate_space_definition(space_def)
    
    def test_validate_multi_discrete_space_valid(self):
        """Test validation of valid MultiDiscrete space."""
        space_def = {
            "type": "MultiDiscrete",
            "nvec": [3, 4, 2]
        }
        
        # Should not raise any exception
        validate_space_definition(space_def)
    
    def test_validate_multi_discrete_space_with_start(self):
        """Test validation of MultiDiscrete space with start."""
        space_def = {
            "type": "MultiDiscrete",
            "nvec": [3, 4],
            "start": [1, 0]
        }
        
        # Should not raise any exception
        validate_space_definition(space_def)
    
    def test_validate_multi_discrete_space_missing_nvec(self):
        """Test validation of MultiDiscrete space missing nvec."""
        space_def = {
            "type": "MultiDiscrete"
        }
        
        with pytest.raises(SpaceConversionError, match="must contain 'nvec' field"):
            validate_space_definition(space_def)
    
    def test_validate_multi_discrete_space_invalid_nvec(self):
        """Test validation of MultiDiscrete space with invalid nvec."""
        space_def = {
            "type": "MultiDiscrete",
            "nvec": [3, 0, 2]  # Contains zero
        }
        
        with pytest.raises(SpaceConversionError, match="must contain positive integers"):
            validate_space_definition(space_def)
    
    def test_validate_multi_discrete_space_start_length_mismatch(self):
        """Test validation of MultiDiscrete space with mismatched start length."""
        space_def = {
            "type": "MultiDiscrete",
            "nvec": [3, 4, 2],
            "start": [1, 0]  # Wrong length
        }
        
        with pytest.raises(SpaceConversionError, match="must have same length"):
            validate_space_definition(space_def)
    
    def test_validate_text_space_valid(self):
        """Test validation of valid Text space."""
        space_def = {
            "type": "Text",
            "min_length": 1,
            "max_length": 100
        }
        
        # Should not raise any exception
        validate_space_definition(space_def)
    
    def test_validate_text_space_invalid_min_length(self):
        """Test validation of Text space with invalid min_length."""
        space_def = {
            "type": "Text",
            "min_length": -1
        }
        
        with pytest.raises(SpaceConversionError, match="must be a non-negative integer"):
            validate_space_definition(space_def)
    
    def test_validate_text_space_invalid_order(self):
        """Test validation of Text space with min > max."""
        space_def = {
            "type": "Text",
            "min_length": 10,
            "max_length": 5
        }
        
        with pytest.raises(SpaceConversionError, match="must be <= 'max_length'"):
            validate_space_definition(space_def)
    
    def test_validate_action_in_space_valid(self):
        """Test validation of valid action in discrete space."""
        space = Discrete(4)
        assert validate_action_in_space(2, space) == True
        assert validate_action_in_space(0, space) == True
        assert validate_action_in_space(3, space) == True
    
    def test_validate_action_in_space_invalid(self):
        """Test validation of invalid action in discrete space."""
        space = Discrete(4)
        assert validate_action_in_space(4, space) == False
        assert validate_action_in_space(-1, space) == False
        assert validate_action_in_space("invalid", space) == False
    
    def test_validate_observation_in_space_valid(self):
        """Test validation of valid observation in box space."""
        space = Box(low=0, high=1, shape=(2,))
        assert validate_observation_in_space(np.array([0.5, 0.8]), space) == True
        assert validate_observation_in_space(np.array([0.0, 1.0]), space) == True
    
    def test_validate_observation_in_space_invalid(self):
        """Test validation of invalid observation in box space."""
        space = Box(low=0, high=1, shape=(2,))
        assert validate_observation_in_space(np.array([1.5, 0.8]), space) == False
        assert validate_observation_in_space(np.array([0.5]), space) == False  # Wrong shape
    
    def test_get_space_info_discrete(self):
        """Test getting info for discrete space."""
        space = Discrete(5, start=1)
        info = get_space_info(space)
        
        expected = {
            "type": "Discrete",
            "shape": None,
            "dtype": space.dtype,
            "n": 5,
            "start": 1
        }
        assert info == expected
    
    def test_get_space_info_box(self):
        """Test getting info for box space."""
        space = Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]))
        info = get_space_info(space)
        
        assert info["type"] == "Box"
        assert info["low"] == [0.0, -1.0]
        assert info["high"] == [1.0, 1.0]
        assert info["shape"] == (2,)
        assert info["dtype"] == space.dtype
    
    def test_get_space_info_multi_binary(self):
        """Test getting info for MultiBinary space."""
        space = MultiBinary([2, 3])
        info = get_space_info(space)
        
        assert info["type"] == "MultiBinary"
        assert info["n"] == [2, 3]
    
    def test_get_space_info_text(self):
        """Test getting info for Text space."""
        space = Text(min_length=5, max_length=50)
        info = get_space_info(space)
        
        expected = {
            "type": "Text",
            "shape": None,
            "dtype": None,
            "min_length": 5,
            "max_length": 50
        }
        assert info == expected
    
    def test_validate_unknown_space_type(self):
        """Test validation with unknown space type."""
        space_def = {
            "type": "UnknownSpace"
        }
        
        with pytest.raises(SpaceConversionError, match="Unknown space type"):
            validate_space_definition(space_def)
    
    def test_validate_space_definition_not_dict(self):
        """Test validation when space definition is not a dict."""
        with pytest.raises(SpaceConversionError, match="must be a dictionary"):
            validate_space_definition("not_a_dict")
