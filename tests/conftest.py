"""
Configuration file for pytest.

This file contains global pytest configurations, fixtures, and utilities
for the gama-gymnasium test suite.
"""

import pytest
import sys
import os
from pathlib import Path
import numpy as np

# Add the source directories to the Python path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# Global pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gama: mark test as requiring GAMA server"
    )
    config.addinivalue_line(
        "markers", "asyncio: mark test as async test"
    )


@pytest.fixture(scope="session")
def project_paths():
    """Provide commonly used project paths."""
    root = Path(__file__).parent.parent
    return {
        "root": root,
        "src": root / "src",
        "tests": root / "tests",
        "examples": root / "examples",
    }


@pytest.fixture
def sample_space_definitions():
    """Provide sample space definitions for testing based on GAMA test files."""
    return {
        # From Spaces test.gaml
        "discrete_space1": {"type": "Discrete", "n": 3},
        "discrete_space2": {"type": "Discrete", "n": 5, "start": 10},
        
        "box_space1": {"type": "Box", "low": 0.0, "high": 100.0, "shape": [2], "dtype": "float"},
        "box_space2": {"type": "Box", "low": 0, "high": 255, "shape": [64, 64], "dtype": "int"},
        "box_space3": {"type": "Box", "low": -1.0, "high": 1.0, "shape": [4, 4, 4], "dtype": "float"},
        "box_space4": {"type": "Box", "low": [-5, -10], "high": [5, 10], "dtype": "int"},
        
        "mb_space1": {"type": "MultiBinary", "n": [4]},
        "mb_space2": {"type": "MultiBinary", "n": [2, 2]},
        "mb_space3": {"type": "MultiBinary", "n": [4, 8, 8]},
        
        "md_space1": {"type": "MultiDiscrete", "nvec": [3, 5, 2]},
        "md_space2": {"type": "MultiDiscrete", "nvec": [10, 5], "start": [100, 200]},
        "md_space3": {"type": "MultiDiscrete", "nvec": [[2, 3], [4, 5]]},
        
        "text_space": {"type": "Text", "min_length": 0, "max_length": 12},
        
        # Additional test spaces
        "box_basic": {"type": "Box"},
        "box_bounded": {
            "type": "Box",
            "low": -1.0,
            "high": 1.0,
            "shape": [2, 2],
            "dtype": "float"
        },
        "box_with_arrays": {
            "type": "Box",
            "low": [-1.0, -2.0],
            "high": [1.0, 2.0],
            "dtype": "int"
        },
        "multibinary_single": {"type": "MultiBinary", "n": 3},
        "multibinary_array": {"type": "MultiBinary", "n": [2, 3]},
        "multidiscrete": {
            "type": "MultiDiscrete",
            "nvec": [2, 3, 4],
            "start": [0, 1, 0]
        },
        "text_basic": {"type": "Text"},
        "text_bounded": {
            "type": "Text",
            "min_length": 5,
            "max_length": 100
        }
    }


@pytest.fixture
def invalid_space_definitions():
    """Provide invalid space definitions for error testing."""
    return {
        "no_type": {"n": 5},
        "unknown_type": {"type": "UnknownSpace", "n": 5},
        "discrete_no_n": {"type": "Discrete", "start": 0},
        "multibinary_no_n": {"type": "MultiBinary"},
        "multidiscrete_no_nvec": {"type": "MultiDiscrete", "start": [0, 1]},
        "empty": {},
        "box_invalid_low": {"type": "Box", "low": "not_a_number", "high": 1.0},
        "box_invalid_dtype": {"type": "Box", "dtype": "invalid_dtype"},
    }


@pytest.fixture
def sample_gama_responses():
    """Provide sample GAMA response messages for testing."""
    from gama_client.message_types import MessageTypes
    
    return {
        "load_success": {
            "type": MessageTypes.CommandExecutedSuccessfully.value,
            "content": "experiment_123"
        },
        "load_failure": {
            "type": "Error",
            "content": "Failed to load experiment: file not found"
        },
        "expression_success": {
            "type": MessageTypes.CommandExecutedSuccessfully.value,
            "content": "[0.1, 0.2, 0.3]"
        },
        "expression_failure": {
            "type": "Error",
            "content": "Invalid expression syntax"
        },
        "step_success": {
            "type": MessageTypes.CommandExecutedSuccessfully.value,
            "content": "Step executed successfully"
        },
        "step_failure": {
            "type": "Error",
            "content": "Step execution failed"
        },
        "reload_success": {
            "type": MessageTypes.CommandExecutedSuccessfully.value,
            "content": "Experiment reloaded"
        }
    }


@pytest.fixture
def sample_step_data():
    """Provide sample step data from GAMA for testing."""
    return {
        "basic_step": {
            "State": [0.1, 0.2, 0.3],
            "Reward": 1.0,
            "Terminated": False,
            "Truncated": False,
            "Info": {"step": 1, "episode_reward": 1.0}
        },
        "terminal_step": {
            "State": [1.0, 1.0, 1.0],
            "Reward": 10.0,
            "Terminated": True,
            "Truncated": False,
            "Info": {"step": 100, "episode_reward": 150.0, "reason": "goal_reached"}
        },
        "truncated_step": {
            "State": [0.5, -0.5, 0.0],
            "Reward": -1.0,
            "Terminated": False,
            "Truncated": True,
            "Info": {"step": 1000, "episode_reward": 50.0, "reason": "time_limit"}
        },
        "negative_reward_step": {
            "State": [-0.1, -0.2, -0.3],
            "Reward": -5.0,
            "Terminated": False,
            "Truncated": False,
            "Info": {"step": 50, "episode_reward": -25.0}
        }
    }


@pytest.fixture
def sample_actions():
    """Provide sample actions for different space types."""
    return {
        "discrete_actions": [0, 1, 2, 3, 4],
        "box_actions": [
            np.array([0.5, -0.3]),
            np.array([1.0, 1.0]),
            np.array([-1.0, 0.0]),
        ],
        "multibinary_actions": [
            np.array([1, 0, 1]),
            np.array([0, 1, 0]),
            np.array([1, 1, 1]),
        ],
        "multidiscrete_actions": [
            np.array([0, 1, 2]),
            np.array([1, 0, 1]),
            np.array([2, 2, 0]),
        ],
        "text_actions": [
            "move_north",
            "move_south",
            "attack",
            "defend"
        ]
    }


# Test utilities
def assert_space_type(space, expected_type):
    """Utility function to assert space type."""
    assert isinstance(space, expected_type), f"Expected {expected_type}, got {type(space)}"


def assert_space_properties(space, **expected_properties):
    """Utility function to assert multiple space properties."""
    for prop_name, expected_value in expected_properties.items():
        actual_value = getattr(space, prop_name)
        assert actual_value == expected_value, (
            f"Property {prop_name}: expected {expected_value}, got {actual_value}"
        )


def create_mock_gama_response(success=True, content=None, message_type=None):
    """Create a mock GAMA response message."""
    if message_type is None:
        from gama_client.message_types import MessageTypes
        message_type = MessageTypes.CommandExecutedSuccessfully.value if success else "Error"
    
    return {
        "type": message_type,
        "content": content or ("Success" if success else "Error occurred")
    }


@pytest.fixture
def mock_gama_client():
    """Provide a mock GAMA client for testing."""
    from unittest.mock import MagicMock
    
    client = MagicMock()
    
    # Set up default successful responses
    client.connect.return_value = None
    client.load.return_value = create_mock_gama_response(True, "exp_123")
    client.reload.return_value = create_mock_gama_response(True, "Reloaded")
    client.step.return_value = create_mock_gama_response(True, "Step executed")
    client.expression.return_value = create_mock_gama_response(True, "Expression result")
    client.close_connection.return_value = None
    
    return client


@pytest.fixture
def mock_space_converter():
    """Provide a mock SpaceConverter for testing."""
    from unittest.mock import MagicMock
    from gymnasium.spaces import Discrete, Box
    
    converter = MagicMock()
    
    # Set up default space conversions
    converter.map_to_space.side_effect = lambda space_def: {
        "Discrete": Discrete(space_def.get("n", 2)),
        "Box": Box(
            low=space_def.get("low", -1.0),
            high=space_def.get("high", 1.0),
            shape=tuple(space_def.get("shape", [1]))
        )
    }.get(space_def.get("type"), Discrete(2))
    
    return converter


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Provide a timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed
        
        @property
        def elapsed(self):
            if self.start_time is None:
                return 0
            end = self.end_time if self.end_time else time.time()
            return end - self.start_time
    
    return Timer()


# Parameterized test data
@pytest.fixture(params=[
    {"type": "Discrete", "n": 3},
    {"type": "Discrete", "n": 5, "start": 10},
    {"type": "Box", "low": 0.0, "high": 1.0},
    {"type": "MultiBinary", "n": [4]},
    {"type": "MultiDiscrete", "nvec": [2, 3]},
    {"type": "Text", "min_length": 0, "max_length": 10}
])
def valid_space_definition(request):
    """Parametrized fixture for valid space definitions."""
    return request.param


@pytest.fixture(params=[
    {},  # No type
    {"type": "InvalidType"},  # Unknown type
    {"type": "Discrete"},  # Missing required parameter
    {"type": "MultiBinary"},  # Missing required parameter
    {"type": "MultiDiscrete"},  # Missing required parameter
])
def invalid_space_definition(request):
    """Parametrized fixture for invalid space definitions."""
    return request.param
