"""
Configuration for pytest tests.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Test configuration
pytest_plugins = []


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def temp_gaml_file(tmp_path_factory):
    """Create a temporary GAML file for testing."""
    # Use the actual test.gaml file instead of creating a temporary one
    test_gaml_path = Path(__file__).parent / "test.gaml"
    if test_gaml_path.exists():
        return str(test_gaml_path)
    
    # Fallback to temporary file if test.gaml doesn't exist
    temp_dir = tmp_path_factory.mktemp("gaml")
    gaml_file = temp_dir / "test_experiment.gaml"
    
    gaml_content = """
model TestModel

import "../gama/gama gymnasium.gaml"

global {
    int observation_space_n <- 4;
    int action_space_n <- 2;
    int current_state <- 0;
    
    init {
        create TestAgent number: 1;
        create GymnasiumManager number: 1;
    }
}

species TestAgent {
    int state <- 0;
    
    action test_action {
        state <- rnd(observation_space_n);
    }
}

species GymnasiumManager parent: GymnasiumCommunication {
    action define_observation_space {
        observation_space <- map(["type"::"Discrete", "n"::observation_space_n]);
    }
    
    action define_action_space {
        action_space <- map(["type"::"Discrete", "n"::action_space_n]);
    }
    
    action get_observation {
        observation <- current_state;
    }
    
    action get_info {
        info <- map([]);
    }
    
    bool is_episode_terminated { return false; }
    bool is_episode_truncated { return false; }
    float get_reward { return 1.0; }
    
    action execute_action(unknown action_data) {
        ask world { current_state <- rnd(observation_space_n - 1); }
    }
    
    action reset_episode {
        ask world { current_state <- 0; }
    }
}

experiment TestExperiment type: headless {
    
}
"""
    
    gaml_file.write_text(gaml_content)
    return str(gaml_file)


@pytest.fixture(scope="session")
def test_gaml_files():
    """Paths to various test GAML files."""
    test_dir = Path(__file__).parent
    return {
        "basic": str(test_dir / "test.gaml"),
        "box_space": str(test_dir / "test_box_space.gaml"),
        "multi_discrete": str(test_dir / "test_multi_discrete.gaml"),
        "multi_binary": str(test_dir / "test_multi_binary.gaml"),
        "text_space": str(test_dir / "test_text_space.gaml"),
        "error_handling": str(test_dir / "test_error_handling.gaml"),
        "nonexistent": str(test_dir / "nonexistent.gaml")
    }


@pytest.fixture
def box_space_gaml_file(test_gaml_files):
    """GAML file for Box space testing."""
    return test_gaml_files["box_space"]


@pytest.fixture
def multi_discrete_gaml_file(test_gaml_files):
    """GAML file for MultiDiscrete space testing."""
    return test_gaml_files["multi_discrete"]


@pytest.fixture
def multi_binary_gaml_file(test_gaml_files):
    """GAML file for MultiBinary space testing."""
    return test_gaml_files["multi_binary"]


@pytest.fixture
def text_space_gaml_file(test_gaml_files):
    """GAML file for Text space testing."""
    return test_gaml_files["text_space"]


@pytest.fixture
def error_gaml_file(test_gaml_files):
    """GAML file for error handling testing."""
    return test_gaml_files["error_handling"]


@pytest.fixture
def nonexistent_gaml_file(test_gaml_files):
    """GAML file that should cause errors."""
    return test_gaml_files["nonexistent"]


@pytest.fixture
def mock_gama_server_responses():
    """Standard mock responses for GAMA server."""
    return {
        "load_success": {
            "type": "CommandExecutedSuccessfully",
            "content": "experiment_123"
        },
        "load_error": {
            "type": "CommandExecutionError", 
            "content": "Failed to load experiment"
        },
        "observation_space": {
            "type": "CommandExecutedSuccessfully",
            "content": {
                "type": "Discrete",
                "n": 4
            }
        },
        "action_space": {
            "type": "CommandExecutedSuccessfully",
            "content": {
                "type": "Discrete",
                "n": 2
            }
        },
        "box_observation_space": {
            "type": "CommandExecutedSuccessfully",
            "content": {
                "type": "Box",
                "low": [0.0, -1.0],
                "high": [1.0, 1.0],
                "shape": [2]
            }
        },
        "reset_observation": {
            "type": "CommandExecutedSuccessfully",
            "content": 0
        },
        "reset_info": {
            "type": "CommandExecutedSuccessfully",
            "content": {"episode": 0}
        },
        "step_response": {
            "type": "CommandExecutedSuccessfully",
            "content": {
                "State": 1,
                "Reward": 1.0,
                "Terminated": False,
                "Truncated": False,
                "Info": {"step": 1}
            }
        },
        "terminated_step": {
            "type": "CommandExecutedSuccessfully",
            "content": {
                "State": 3,
                "Reward": 10.0,
                "Terminated": True,
                "Truncated": False,
                "Info": {"episode_end": True}
            }
        },
        "step_success": {
            "type": "CommandExecutedSuccessfully",
            "content": ""
        },
        "stop_success": {
            "type": "CommandExecutedSuccessfully",
            "content": ""
        }
    }


@pytest.fixture
def sample_space_definitions():
    """Sample space definitions for testing."""
    return {
        "discrete": {"type": "Discrete", "n": 5},
        "box_bounded": {
            "type": "Box",
            "low": [0.0, -1.0],
            "high": [1.0, 1.0], 
            "shape": [2]
        },
        "box_unbounded": {
            "type": "Box",
            "low": "-Infinity",
            "high": "Infinity",
            "shape": [3]
        },
        "multi_binary": {"type": "MultiBinary", "n": 8},
        "multi_discrete": {"type": "MultiDiscrete", "nvec": [3, 4, 2]},
        "text": {
            "type": "Text",
            "max_length": 100,
            "charset": "abcdefghijklmnopqrstuvwxyz "
        }
    }


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark performance tests
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Mark slow tests
        if "test_concurrent" in item.name or "test_memory" in item.name:
            item.add_marker(pytest.mark.slow)


# Test utilities
class MockGamaClient:
    """Mock GAMA client for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.call_count = {}
        self.last_calls = {}
        self.is_connected = False
        
    def connect(self):
        self.is_connected = True
        
    def disconnect(self):
        self.is_connected = False
        
    def load_experiment(self, gaml_path, experiment_name):
        self._record_call("load_experiment", (gaml_path, experiment_name))
        return self.responses.get("load", {"type": "CommandExecutedSuccessfully", "content": "exp_123"})
    
    def execute_expression(self, expression, experiment_id=None):
        self._record_call("execute_expression", (expression, experiment_id))
        
        # Return appropriate response based on expression
        if "observation_space" in expression:
            return self.responses.get("observation_space", {"type": "CommandExecutedSuccessfully", "content": {"type": "Discrete", "n": 4}})
        elif "action_space" in expression:
            return self.responses.get("action_space", {"type": "CommandExecutedSuccessfully", "content": {"type": "Discrete", "n": 2}})
        else:
            return self.responses.get("expression", {"type": "CommandExecutedSuccessfully", "content": "result"})
    
    def step_experiment(self, action, experiment_id):
        self._record_call("step_experiment", (action, experiment_id))
        return self.responses.get("step", {"type": "CommandExecutedSuccessfully", "content": ""})
    
    def stop_experiment(self, experiment_id):
        self._record_call("stop_experiment", (experiment_id,))
        return self.responses.get("stop", {"type": "CommandExecutedSuccessfully", "content": ""})
    
    def _record_call(self, method, args):
        self.call_count[method] = self.call_count.get(method, 0) + 1
        self.last_calls[method] = args


@pytest.fixture
def mock_gama_client():
    """Fixture providing a mock GAMA client."""
    return MockGamaClient()


# Performance testing utilities
def measure_time(func, *args, **kwargs):
    """Measure execution time of a function."""
    import time
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start


def measure_memory(func, *args, **kwargs):
    """Measure memory usage of a function."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Measure initial memory
    initial_memory = process.memory_info().rss
    
    # Execute function
    result = func(*args, **kwargs)
    
    # Force garbage collection again
    gc.collect()
    
    # Measure final memory
    final_memory = process.memory_info().rss
    
    memory_diff = final_memory - initial_memory
    
    return result, memory_diff


# Test data generators
def generate_test_actions(action_space_def, count=10):
    """Generate test actions for a given action space."""
    import numpy as np
    from gama_gymnasium.spaces.converters import map_to_space
    
    space = map_to_space(action_space_def)
    actions = []
    
    for _ in range(count):
        actions.append(space.sample())
    
    return actions


def generate_test_observations(observation_space_def, count=10):
    """Generate test observations for a given observation space."""
    import numpy as np
    from gama_gymnasium.spaces.converters import map_to_space
    
    space = map_to_space(observation_space_def)
    observations = []
    
    for _ in range(count):
        observations.append(space.sample())
    
    return observations
