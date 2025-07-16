"""
Unit tests for SpaceConverter class.

This module contains comprehensive tests for the SpaceConverter class,
based on the test scenarios from spaces_test.py and Spaces test.gaml.
"""

import pytest
import numpy as np
from typing import Dict, Any

from gymnasium.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Text

# Import the classes to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gama_gymnasium.space_converter import SpaceConverter
from gama_gymnasium.exceptions import SpaceConversionError


class TestSpaceConverter:
    """Test suite for SpaceConverter class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.converter = SpaceConverter()

    def test_init(self):
        """Test SpaceConverter initialization."""
        assert isinstance(self.converter, SpaceConverter)

    # Tests for main map_to_space method
    def test_map_to_space_missing_type(self):
        """Test that missing type raises SpaceConversionError."""
        space_map = {"n": 5}  # Missing "type" key
        
        with pytest.raises(SpaceConversionError, match="No type specified in space definition"):
            self.converter.map_to_space(space_map)

    def test_map_to_space_unknown_type(self):
        """Test that unknown space type raises SpaceConversionError."""
        space_map = {"type": "UnknownSpaceType", "n": 5}
        
        with pytest.raises(SpaceConversionError, match="Unknown space type: UnknownSpaceType"):
            self.converter.map_to_space(space_map)

    def test_map_to_space_conversion_error(self):
        """Test that conversion errors are properly wrapped."""
        # Create a malformed discrete space (missing required 'n' parameter)
        space_map = {"type": "Discrete"}
        
        with pytest.raises(SpaceConversionError, match="Failed to convert Discrete space"):
            self.converter.map_to_space(space_map)

    # Tests for Discrete space conversion (based on spaces_test.py examples)
    def test_map_to_discrete_basic(self):
        """Test basic discrete space conversion (discrete_space1 from GAMA)."""
        discrete_map = {"type": "Discrete", "n": 3}
        space = self.converter.map_to_space(discrete_map)
        
        assert isinstance(space, Discrete)
        assert space.n == 3
        assert space.start == 0  # Default start value

    def test_map_to_discrete_with_start(self):
        """Test discrete space conversion with custom start value (discrete_space2 from GAMA)."""
        discrete_map = {"type": "Discrete", "n": 5, "start": 10}
        space = self.converter.map_to_space(discrete_map)
        
        assert isinstance(space, Discrete)
        assert space.n == 5
        assert space.start == 10

    def test_map_to_discrete_missing_n(self):
        """Test discrete space conversion with missing n parameter."""
        discrete_map = {"type": "Discrete", "start": 0}
        
        with pytest.raises(SpaceConversionError):
            self.converter.map_to_space(discrete_map)

    # Tests for Box space conversion (based on spaces_test.py examples)
    def test_map_to_box_space1_from_gama(self):
        """Test box_space1 from GAMA: ["type"::"Box", "low"::0.0, "high"::100.0, "shape"::[2], "dtype"::"float"]"""
        box_map = {
            "type": "Box",
            "low": 0.0,
            "high": 100.0,
            "shape": [2],
            "dtype": "float"
        }
        space = self.converter.map_to_space(box_map)
        
        assert isinstance(space, Box)
        assert space.low[0] == 0.0
        assert space.high[0] == 100.0
        assert space.shape == (2,)
        assert space.dtype == np.float64

    def test_map_to_box_space2_from_gama(self):
        """Test box_space2 from GAMA: ["type"::"Box", "low"::0, "high"::255, "shape"::[64, 64], "dtype"::"int"]"""
        box_map = {
            "type": "Box",
            "low": 0,
            "high": 255,
            "shape": [64, 64],
            "dtype": "int"
        }
        space = self.converter.map_to_space(box_map)
        
        assert isinstance(space, Box)
        assert space.low[0][0] == 0
        assert space.high[0][0] == 255
        assert space.shape == (64, 64)
        assert space.dtype == np.int64

    def test_map_to_box_space3_from_gama(self):
        """Test box_space3 from GAMA: ["type"::"Box", "low"::-1.0, "high"::1.0, "shape"::[4, 4, 4], "dtype"::"float"]"""
        box_map = {
            "type": "Box",
            "low": -1.0,
            "high": 1.0,
            "shape": [4, 4, 4],
            "dtype": "float"
        }
        space = self.converter.map_to_space(box_map)
        
        assert isinstance(space, Box)
        assert space.low[0][0][0] == -1.0
        assert space.high[0][0][0] == 1.0
        assert space.shape == (4, 4, 4)
        assert space.dtype == np.float64

    def test_map_to_box_space4_from_gama(self):
        """Test box_space4 from GAMA: ["type"::"Box", "low"::[-5, -10], "high"::[5, 10], "dtype"::"int"]"""
        box_map = {
            "type": "Box",
            "low": [-5, -10],
            "high": [5, 10],
            "dtype": "int"
        }
        space = self.converter.map_to_space(box_map)
        
        assert isinstance(space, Box)
        np.testing.assert_array_equal(space.low, np.array([-5, -10]))
        np.testing.assert_array_equal(space.high, np.array([5, 10]))
        assert space.dtype == np.int64

    def test_map_to_box_with_defaults(self):
        """Test box space conversion with default values."""
        box_map = {"type": "Box"}
        space = self.converter.map_to_space(box_map)
        
        assert isinstance(space, Box)
        assert space.low == -np.inf
        assert space.high == np.inf
        assert space.dtype == np.float32

    def test_map_to_box_with_infinity_strings(self):
        """Test box space conversion with string infinity values."""
        box_map = {
            "type": "Box",
            "low": ["-Infinity", -1.0],
            "high": ["Infinity", 1.0]
        }
        space = self.converter.map_to_space(box_map)
        
        assert isinstance(space, Box)
        np.testing.assert_array_equal(space.low, np.array([-np.inf, -1.0]))
        np.testing.assert_array_equal(space.high, np.array([np.inf, 1.0]))

    def test_map_to_box_unknown_dtype(self):
        """Test box space conversion with unknown dtype defaults to float32."""
        box_map = {
            "type": "Box",
            "dtype": "unknown_type"
        }
        space = self.converter.map_to_space(box_map)
        
        assert isinstance(space, Box)
        assert space.dtype == np.float32

    # Tests for MultiBinary space conversion (based on spaces_test.py examples)
    def test_map_to_multi_binary_space1_from_gama(self):
        """Test mb_space1 from GAMA: ["type"::"MultiBinary", "n"::[4]]"""
        mb_map = {"type": "MultiBinary", "n": [4]}
        space = self.converter.map_to_space(mb_map)
        
        assert isinstance(space, MultiBinary)
        assert space.n == 4

    def test_map_to_multi_binary_space2_from_gama(self):
        """Test mb_space2 from GAMA: ["type"::"MultiBinary", "n"::[2, 2]]"""
        mb_map = {"type": "MultiBinary", "n": [2, 2]}
        space = self.converter.map_to_space(mb_map)
        
        assert isinstance(space, MultiBinary)
        np.testing.assert_array_equal(space.n, [2, 2])

    def test_map_to_multi_binary_space3_from_gama(self):
        """Test mb_space3 from GAMA: ["type"::"MultiBinary", "n"::[4, 8, 8]]"""
        mb_map = {"type": "MultiBinary", "n": [4, 8, 8]}
        space = self.converter.map_to_space(mb_map)
        
        assert isinstance(space, MultiBinary)
        np.testing.assert_array_equal(space.n, [4, 8, 8])

    def test_map_to_multi_binary_single_value(self):
        """Test multibinary space conversion with single integer."""
        mb_map = {"type": "MultiBinary", "n": 5}
        space = self.converter.map_to_space(mb_map)
        
        assert isinstance(space, MultiBinary)
        assert space.n == 5

    # Tests for MultiDiscrete space conversion (based on spaces_test.py examples)
    def test_map_to_multi_discrete_space1_from_gama(self):
        """Test md_space1 from GAMA: ["type"::"MultiDiscrete", "nvec"::[3, 5, 2]]"""
        md_map = {"type": "MultiDiscrete", "nvec": [3, 5, 2]}
        space = self.converter.map_to_space(md_map)
        
        assert isinstance(space, MultiDiscrete)
        np.testing.assert_array_equal(space.nvec, [3, 5, 2])

    def test_map_to_multi_discrete_space2_from_gama(self):
        """Test md_space2 from GAMA: ["type"::"MultiDiscrete", "nvec"::[10, 5], "start"::[100, 200]]"""
        md_map = {
            "type": "MultiDiscrete", 
            "nvec": [10, 5],
            "start": [100, 200]
        }
        space = self.converter.map_to_space(md_map)
        
        assert isinstance(space, MultiDiscrete)
        np.testing.assert_array_equal(space.nvec, [10, 5])
        np.testing.assert_array_equal(space.start, [100, 200])

    def test_map_to_multi_discrete_space3_from_gama(self):
        """Test md_space3 from GAMA: ["type"::"MultiDiscrete", "nvec"::[[2, 3], [4, 5]]]"""
        md_map = {"type": "MultiDiscrete", "nvec": [[2, 3], [4, 5]]}
        space = self.converter.map_to_space(md_map)
        
        assert isinstance(space, MultiDiscrete)
        np.testing.assert_array_equal(space.nvec, [[2, 3], [4, 5]])

    def test_map_to_multi_discrete_no_start(self):
        """Test multidiscrete space conversion without start values."""
        md_map = {"type": "MultiDiscrete", "nvec": [2, 3]}
        space = self.converter.map_to_space(md_map)
        
        assert isinstance(space, MultiDiscrete)
        np.testing.assert_array_equal(space.nvec, [2, 3])

    # Tests for Text space conversion (based on spaces_test.py examples)
    def test_map_to_text_space_from_gama(self):
        """Test text_space from GAMA: ["type"::"Text", "min_length"::0, "max_length"::12]"""
        text_map = {
            "type": "Text",
            "min_length": 0,
            "max_length": 12
        }
        space = self.converter.map_to_space(text_map)
        
        assert isinstance(space, Text)
        assert space.min_length == 0
        assert space.max_length == 12

    def test_map_to_text_basic(self):
        """Test basic text space conversion with defaults."""
        text_map = {"type": "Text"}
        space = self.converter.map_to_space(text_map)
        
        assert isinstance(space, Text)
        assert space.min_length == 0
        assert space.max_length == 1000

    def test_map_to_text_with_lengths(self):
        """Test text space conversion with custom lengths."""
        text_map = {
            "type": "Text",
            "min_length": 5,
            "max_length": 100
        }
        space = self.converter.map_to_space(text_map)
        
        assert isinstance(space, Text)
        assert space.min_length == 5
        assert space.max_length == 100

    # Tests for _replace_infinity helper method
    def test_replace_infinity_single_values(self):
        """Test infinity replacement for single values."""
        assert self.converter._replace_infinity("Infinity") == float('inf')
        assert self.converter._replace_infinity("-Infinity") == float('-inf')
        assert self.converter._replace_infinity(5.0) == 5.0
        assert self.converter._replace_infinity("normal") == "normal"

    def test_replace_infinity_lists(self):
        """Test infinity replacement in lists."""
        input_list = ["Infinity", -1.0, "-Infinity", 2.0]
        expected = [float('inf'), -1.0, float('-inf'), 2.0]
        result = self.converter._replace_infinity(input_list)
        
        assert result == expected

    def test_replace_infinity_nested_lists(self):
        """Test infinity replacement in nested lists."""
        input_nested = [["Infinity", 1.0], [2.0, "-Infinity"]]
        expected = [[float('inf'), 1.0], [2.0, float('-inf')]]
        result = self.converter._replace_infinity(input_nested)
        
        assert result == expected

    def test_replace_infinity_empty_list(self):
        """Test infinity replacement with empty list."""
        result = self.converter._replace_infinity([])
        assert result == []


class TestSpaceConverterCompatibility:
    """Test compatibility with actual GAMA space definitions from the test files."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.converter = SpaceConverter()

    def test_all_gama_box_spaces(self):
        """Test all box space definitions from Spaces test.gaml."""
        gama_box_spaces = [
            {"type": "Box", "low": 0.0, "high": 100.0, "shape": [2], "dtype": "float"},
            {"type": "Box", "low": 0, "high": 255, "shape": [64, 64], "dtype": "int"},
            {"type": "Box", "low": -1.0, "high": 1.0, "shape": [4, 4, 4], "dtype": "float"},
            {"type": "Box", "low": [-5, -10], "high": [5, 10], "dtype": "int"}
        ]
        
        # All should convert without errors
        for i, box_def in enumerate(gama_box_spaces):
            space = self.converter.map_to_space(box_def)
            assert isinstance(space, Box), f"Box space {i} failed to convert"

    def test_all_gama_discrete_spaces(self):
        """Test all discrete space definitions from Spaces test.gaml."""
        gama_discrete_spaces = [
            {"type": "Discrete", "n": 3},
            {"type": "Discrete", "n": 5, "start": 10}
        ]
        
        # All should convert without errors
        for i, discrete_def in enumerate(gama_discrete_spaces):
            space = self.converter.map_to_space(discrete_def)
            assert isinstance(space, Discrete), f"Discrete space {i} failed to convert"

    def test_all_gama_multibinary_spaces(self):
        """Test all multibinary space definitions from Spaces test.gaml."""
        gama_mb_spaces = [
            {"type": "MultiBinary", "n": [4]},
            {"type": "MultiBinary", "n": [2, 2]},
            {"type": "MultiBinary", "n": [4, 8, 8]}
        ]
        
        # All should convert without errors
        for i, mb_def in enumerate(gama_mb_spaces):
            space = self.converter.map_to_space(mb_def)
            assert isinstance(space, MultiBinary), f"MultiBinary space {i} failed to convert"

    def test_all_gama_multidiscrete_spaces(self):
        """Test all multidiscrete space definitions from Spaces test.gaml."""
        gama_md_spaces = [
            {"type": "MultiDiscrete", "nvec": [3, 5, 2]},
            {"type": "MultiDiscrete", "nvec": [10, 5], "start": [100, 200]},
            {"type": "MultiDiscrete", "nvec": [[2, 3], [4, 5]]}
        ]
        
        # All should convert without errors
        for i, md_def in enumerate(gama_md_spaces):
            space = self.converter.map_to_space(md_def)
            assert isinstance(space, MultiDiscrete), f"MultiDiscrete space {i} failed to convert"

    def test_all_gama_text_spaces(self):
        """Test all text space definitions from Spaces test.gaml."""
        gama_text_spaces = [
            {"type": "Text", "min_length": 0, "max_length": 12}
        ]
        
        # All should convert without errors
        for i, text_def in enumerate(gama_text_spaces):
            space = self.converter.map_to_space(text_def)
            assert isinstance(space, Text), f"Text space {i} failed to convert"


class TestSpaceConverterIntegration:
    """Integration tests for SpaceConverter with complex scenarios."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.converter = SpaceConverter()

    def test_complex_box_space_with_all_parameters(self):
        """Test conversion of complex box space with all parameters."""
        complex_box = {
            "type": "Box",
            "low": ["-Infinity", -10.0, 0.0],
            "high": ["Infinity", 10.0, 1.0],
            "shape": [3],
            "dtype": "float"
        }
        
        space = self.converter.map_to_space(complex_box)
        
        assert isinstance(space, Box)
        assert space.shape == (3,)
        assert space.dtype == np.float64
        np.testing.assert_array_equal(
            space.low, 
            np.array([-np.inf, -10.0, 0.0])
        )
        np.testing.assert_array_equal(
            space.high, 
            np.array([np.inf, 10.0, 1.0])
        )

    def test_all_space_types_conversion(self):
        """Test that all supported space types can be converted."""
        space_definitions = [
            {"type": "Discrete", "n": 5},
            {"type": "Box", "low": -1.0, "high": 1.0},
            {"type": "MultiBinary", "n": 3},
            {"type": "MultiDiscrete", "nvec": [2, 3]},
            {"type": "Text", "min_length": 1, "max_length": 10}
        ]
        
        expected_types = [Discrete, Box, MultiBinary, MultiDiscrete, Text]
        
        for space_def, expected_type in zip(space_definitions, expected_types):
            space = self.converter.map_to_space(space_def)
            assert isinstance(space, expected_type)

    def test_error_handling_chain(self):
        """Test that errors are properly wrapped and chained."""
        # Test with a space type that exists but has invalid parameters
        invalid_box = {
            "type": "Box",
            "low": "not_a_number",  # This should cause an error
            "high": 1.0
        }
        
        with pytest.raises(SpaceConversionError) as exc_info:
            self.converter.map_to_space(invalid_box)
        
        # Verify the error message contains useful information
        assert "Failed to convert Box space" in str(exc_info.value)


# Parametrized tests for comprehensive coverage
class TestSpaceConverterParametrized:
    """Parametrized tests for SpaceConverter."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.converter = SpaceConverter()

    @pytest.mark.parametrize("space_def,expected_n,expected_start", [
        ({"type": "Discrete", "n": 2}, 2, 0),
        ({"type": "Discrete", "n": 10, "start": 1}, 10, 1),
        ({"type": "Discrete", "n": 100}, 100, 0),
        ({"type": "Discrete", "n": 3}, 3, 0),  # From GAMA test
        ({"type": "Discrete", "n": 5, "start": 10}, 5, 10),  # From GAMA test
    ])
    def test_discrete_spaces_parametrized(self, space_def, expected_n, expected_start):
        """Parametrized test for discrete space conversions."""
        space = self.converter.map_to_space(space_def)
        
        assert isinstance(space, Discrete)
        assert space.n == expected_n
        assert space.start == expected_start

    @pytest.mark.parametrize("space_def,expected_low,expected_high,expected_dtype", [
        ({"type": "Box"}, -np.inf, np.inf, np.float32),
        ({"type": "Box", "low": 0, "high": 1}, 0, 1, np.float32),
        ({"type": "Box", "dtype": "int"}, np.iinfo(np.int64).min, np.iinfo(np.int64).max, np.int64),
        ({"type": "Box", "dtype": "float"}, -np.inf, np.inf, np.float64),
        ({"type": "Box", "low": 0.0, "high": 100.0, "dtype": "float"}, 0.0, 100.0, np.float64),  # From GAMA
    ])
    def test_box_spaces_parametrized(self, space_def, expected_low, expected_high, expected_dtype):
        """Parametrized test for box space conversions."""
        space = self.converter.map_to_space(space_def)
        
        assert isinstance(space, Box)
        assert space.low[0] == expected_low
        assert space.high[0] == expected_high
        assert space.dtype == expected_dtype

    @pytest.mark.parametrize("invalid_space", [
        {},  # No type
        {"type": "InvalidType"},  # Unknown type
        {"type": "Discrete"},  # Missing required parameter
        {"type": "MultiBinary"},  # Missing required parameter
        {"type": "MultiDiscrete"},  # Missing required parameter
    ])
    def test_invalid_spaces_parametrized(self, invalid_space):
        """Parametrized test for invalid space definitions."""
        with pytest.raises(SpaceConversionError):
            self.converter.map_to_space(invalid_space)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
