"""
Unit tests for GamaEnv class.

This module contains comprehensive tests for the main GamaEnv class,
testing the Gymnasium interface implementation for GAMA simulations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

import gymnasium as gym
from gymnasium.spaces import Discrete, Box, MultiBinary

# Import the classes to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gama_gymnasium.gama_env import GamaEnv
from gama_gymnasium.gama_client_wrapper import GamaClientWrapper
from gama_gymnasium.space_converter import SpaceConverter
from gama_gymnasium.exceptions import GamaEnvironmentError


class TestGamaEnvInitialization:
    """Test GamaEnv initialization and setup."""

    @patch('gama_gymnasium.gama_env.GamaClientWrapper')
    @patch('gama_gymnasium.gama_env.SpaceConverter')
    def test_init_minimal_params(self, mock_space_converter_class, mock_client_wrapper_class):
        """Test initialization with minimal required parameters."""
        # Mock the dependencies
        mock_client = MagicMock()
        mock_client_wrapper_class.return_value = mock_client
        mock_converter = MagicMock()
        mock_space_converter_class.return_value = mock_converter
        
        # Mock setup methods
        with patch.object(GamaEnv, '_setup_experiment') as mock_setup_exp, \
             patch.object(GamaEnv, '_setup_spaces') as mock_setup_spaces:
            
            env = GamaEnv(
                gaml_experiment_path="/path/to/model.gaml",
                gaml_experiment_name="test_experiment"
            )
            
            # Verify basic attributes
            assert env.gaml_file_path == "/path/to/model.gaml"
            assert env.experiment_name == "test_experiment"
            assert env.experiment_parameters == []
            assert env.render_mode is None
            
            # Verify dependencies were created
            assert env.gama_client == mock_client
            assert env.space_converter == mock_converter
            
            # Verify setup methods were called
            mock_setup_exp.assert_called_once()
            mock_setup_spaces.assert_called_once()

    @patch('gama_gymnasium.gama_env.GamaClientWrapper')
    @patch('gama_gymnasium.gama_env.SpaceConverter')
    def test_init_with_all_params(self, mock_space_converter_class, mock_client_wrapper_class):
        """Test initialization with all optional parameters."""
        mock_client_wrapper_class.return_value = MagicMock()
        mock_space_converter_class.return_value = MagicMock()
        
        with patch.object(GamaEnv, '_setup_experiment'), \
             patch.object(GamaEnv, '_setup_spaces'):
            
            env = GamaEnv(
                gaml_experiment_path="/path/to/complex_model.gaml",
                gaml_experiment_name="complex_experiment",
                gaml_experiment_parameters=[{"param1": "value1", "param2": 42}],
                gama_ip_address="192.168.1.100",
                gama_port=8080,
                render_mode="human"
            )
            
            assert env.gaml_file_path == "/path/to/complex_model.gaml"
            assert env.experiment_name == "complex_experiment"
            assert env.experiment_parameters == [{"param1": "value1", "param2": 42}]
            assert env.render_mode == "human"
            
            # Verify client was created with correct parameters
            mock_client_wrapper_class.assert_called_once_with("192.168.1.100", 8080)

    @patch('gama_gymnasium.gama_env.GamaClientWrapper')
    @patch('gama_gymnasium.gama_env.SpaceConverter')
    def test_init_with_default_params(self, mock_space_converter_class, mock_client_wrapper_class):
        """Test initialization with default parameter values."""
        mock_client_wrapper_class.return_value = MagicMock()
        mock_space_converter_class.return_value = MagicMock()
        
        with patch.object(GamaEnv, '_setup_experiment'), \
             patch.object(GamaEnv, '_setup_spaces'):
            
            env = GamaEnv("/path/to/model.gaml", "test")
            
            # Verify default values
            assert env.experiment_parameters == []
            assert env.render_mode is None
            
            # Verify client was created with default parameters
            mock_client_wrapper_class.assert_called_once_with(None, 6868)


class TestGamaEnvSetup:
    """Test GamaEnv setup methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch('gama_gymnasium.gama_env.GamaClientWrapper'), \
             patch('gama_gymnasium.gama_env.SpaceConverter'), \
             patch.object(GamaEnv, '_setup_experiment'), \
             patch.object(GamaEnv, '_setup_spaces'):
            
            self.env = GamaEnv("/path/to/model.gaml", "test")
        
        # Mock the dependencies
        self.mock_client = MagicMock()
        self.env.gama_client = self.mock_client
        self.mock_converter = MagicMock()
        self.env.space_converter = self.mock_converter

    def test_setup_experiment(self):
        """Test experiment setup."""
        # Mock load_experiment return value
        self.mock_client.load_experiment.return_value = "exp_123"
        
        # Call the actual method
        self.env._setup_experiment()
        
        # Verify experiment was loaded correctly
        self.mock_client.load_experiment.assert_called_once_with(
            "/path/to/model.gaml",
            "test",
            []
        )
        assert self.env.experiment_id == "exp_123"

    def test_setup_spaces(self):
        """Test space setup."""
        # Mock space data from GAMA
        obs_space_data = {"type": "Box", "low": -1.0, "high": 1.0, "shape": [4]}
        action_space_data = {"type": "Discrete", "n": 3}
        
        self.mock_client.get_observation_space.return_value = obs_space_data
        self.mock_client.get_action_space.return_value = action_space_data
        
        # Mock converted spaces
        obs_space = Box(-1.0, 1.0, shape=(4,))
        action_space = Discrete(3)
        self.mock_converter.map_to_space.side_effect = [obs_space, action_space]
        
        # Set experiment_id for the test
        self.env.experiment_id = "exp_123"
        
        # Call the actual method
        self.env._setup_spaces()
        
        # Verify spaces were retrieved from GAMA
        self.mock_client.get_observation_space.assert_called_once_with("exp_123")
        self.mock_client.get_action_space.assert_called_once_with("exp_123")
        
        # Verify spaces were converted
        assert self.mock_converter.map_to_space.call_count == 2
        self.mock_converter.map_to_space.assert_any_call(obs_space_data)
        self.mock_converter.map_to_space.assert_any_call(action_space_data)
        
        # Verify spaces were set
        assert self.env.observation_space == obs_space
        assert self.env.action_space == action_space


class TestGamaEnvReset:
    """Test GamaEnv reset functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch('gama_gymnasium.gama_env.GamaClientWrapper'), \
             patch('gama_gymnasium.gama_env.SpaceConverter'), \
             patch.object(GamaEnv, '_setup_experiment'), \
             patch.object(GamaEnv, '_setup_spaces'):
            
            self.env = GamaEnv("/path/to/model.gaml", "test")
        
        # Mock dependencies
        self.mock_client = MagicMock()
        self.env.gama_client = self.mock_client
        self.env.experiment_id = "exp_123"
        
        # Set up mock observation space
        self.mock_obs_space = MagicMock()
        self.env.observation_space = self.mock_obs_space

    def test_reset_basic(self):
        """Test basic reset functionality."""
        # Mock GAMA responses
        raw_state = [0.1, 0.2, 0.3]
        converted_state = np.array([0.1, 0.2, 0.3])
        info = {"episode": 1, "step": 0}
        
        self.mock_client.get_state.return_value = raw_state
        self.mock_client.get_info.return_value = info
        self.mock_obs_space.from_jsonable.return_value = [converted_state]
        
        # Reset environment
        state, returned_info = self.env.reset()
        
        # Verify GAMA client calls
        self.mock_client.reset_experiment.assert_called_once_with("exp_123", None)
        self.mock_client.get_state.assert_called_once_with("exp_123")
        self.mock_client.get_info.assert_called_once_with("exp_123")
        
        # Verify state conversion
        self.mock_obs_space.from_jsonable.assert_called_once_with([raw_state])
        
        # Verify return values
        np.testing.assert_array_equal(state, converted_state)
        assert returned_info == info

    def test_reset_with_seed(self):
        """Test reset with explicit seed."""
        seed = 42
        raw_state = [0.0, 0.0]
        converted_state = np.array([0.0, 0.0])
        
        self.mock_client.get_state.return_value = raw_state
        self.mock_client.get_info.return_value = {}
        self.mock_obs_space.from_jsonable.return_value = [converted_state]
        
        # Reset with seed
        state, info = self.env.reset(seed=seed)
        
        # Verify seed was passed to GAMA client
        self.mock_client.reset_experiment.assert_called_once_with("exp_123", seed)

    def test_reset_with_options(self):
        """Test reset with options parameter."""
        options = {"option1": "value1", "option2": 42}
        
        self.mock_client.get_state.return_value = [0.0]
        self.mock_client.get_info.return_value = {}
        self.mock_obs_space.from_jsonable.return_value = [np.array([0.0])]
        
        # Reset with options
        state, info = self.env.reset(seed=None, options=options)
        
        # Options are not currently used in implementation, but should not cause errors
        assert state is not None
        assert isinstance(info, dict)

    def test_reset_calls_super(self):
        """Test that reset calls parent class reset method."""
        self.mock_client.get_state.return_value = [0.0]
        self.mock_client.get_info.return_value = {}
        self.mock_obs_space.from_jsonable.return_value = [np.array([0.0])]
        
        with patch.object(gym.Env, 'reset') as mock_super_reset:
            self.env.reset(seed=123, options={"test": True})
            
            # Verify super().reset() was called with correct parameters
            mock_super_reset.assert_called_once_with(seed=123, options={"test": True})


class TestGamaEnvStep:
    """Test GamaEnv step functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch('gama_gymnasium.gama_env.GamaClientWrapper'), \
             patch('gama_gymnasium.gama_env.SpaceConverter'), \
             patch.object(GamaEnv, '_setup_experiment'), \
             patch.object(GamaEnv, '_setup_spaces'):
            
            self.env = GamaEnv("/path/to/model.gaml", "test")
        
        # Mock dependencies
        self.mock_client = MagicMock()
        self.env.gama_client = self.mock_client
        self.env.experiment_id = "exp_123"
        
        # Set up mock spaces
        self.mock_action_space = MagicMock()
        self.mock_obs_space = MagicMock()
        self.env.action_space = self.mock_action_space
        self.env.observation_space = self.mock_obs_space

    def test_step_basic(self):
        """Test basic step functionality."""
        # Input action
        action = np.array([1, 0, 1])
        converted_action = [1, 0, 1]
        
        # Expected GAMA response
        step_data = {
            "State": [0.2, 0.3, 0.4],
            "Reward": 1.5,
            "Terminated": False,
            "Truncated": False,
            "Info": {"step": 1, "score": 10}
        }
        
        # Expected converted state
        converted_state = np.array([0.2, 0.3, 0.4])
        
        # Mock conversions
        self.mock_action_space.to_jsonable.return_value = [converted_action]
        self.mock_client.execute_step.return_value = step_data
        self.mock_obs_space.from_jsonable.return_value = [converted_state]
        
        # Execute step
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Verify action conversion
        self.mock_action_space.to_jsonable.assert_called_once_with([action])
        
        # Verify GAMA step execution
        self.mock_client.execute_step.assert_called_once_with("exp_123", converted_action)
        
        # Verify state conversion
        self.mock_obs_space.from_jsonable.assert_called_once_with([step_data["State"]])
        
        # Verify return values
        np.testing.assert_array_equal(state, converted_state)
        assert reward == 1.5
        assert terminated == False
        assert truncated == False
        assert info == {"step": 1, "score": 10}

    def test_step_with_different_action_types(self):
        """Test step with different action types."""
        test_actions = [
            2,  # Integer action
            [1, 0, 1],  # List action
            np.array([0.5, -0.3]),  # Numpy array action
        ]
        
        # Mock standard response
        step_data = {
            "State": [0.0],
            "Reward": 0.0,
            "Terminated": False,
            "Truncated": False,
            "Info": {}
        }
        self.mock_client.execute_step.return_value = step_data
        self.mock_obs_space.from_jsonable.return_value = [np.array([0.0])]
        
        for action in test_actions:
            self.mock_action_space.to_jsonable.return_value = [action]
            
            state, reward, terminated, truncated, info = self.env.step(action)
            
            # Verify each action type works
            assert state is not None
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

    def test_step_with_termination(self):
        """Test step that results in episode termination."""
        action = 1
        step_data = {
            "State": [0.0, 0.0],
            "Reward": 10.0,
            "Terminated": True,
            "Truncated": False,
            "Info": {"reason": "goal_reached"}
        }
        
        self.mock_action_space.to_jsonable.return_value = [action]
        self.mock_client.execute_step.return_value = step_data
        self.mock_obs_space.from_jsonable.return_value = [np.array([0.0, 0.0])]
        
        state, reward, terminated, truncated, info = self.env.step(action)
        
        assert terminated == True
        assert truncated == False
        assert reward == 10.0
        assert info["reason"] == "goal_reached"

    def test_step_with_truncation(self):
        """Test step that results in episode truncation."""
        action = 0
        step_data = {
            "State": [1.0, 1.0],
            "Reward": -1.0,
            "Terminated": False,
            "Truncated": True,
            "Info": {"reason": "time_limit"}
        }
        
        self.mock_action_space.to_jsonable.return_value = [action]
        self.mock_client.execute_step.return_value = step_data
        self.mock_obs_space.from_jsonable.return_value = [np.array([1.0, 1.0])]
        
        state, reward, terminated, truncated, info = self.env.step(action)
        
        assert terminated == False
        assert truncated == True
        assert reward == -1.0
        assert info["reason"] == "time_limit"


class TestGamaEnvRenderAndClose:
    """Test GamaEnv render and close methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch('gama_gymnasium.gama_env.GamaClientWrapper'), \
             patch('gama_gymnasium.gama_env.SpaceConverter'), \
             patch.object(GamaEnv, '_setup_experiment'), \
             patch.object(GamaEnv, '_setup_spaces'):
            
            self.env = GamaEnv("/path/to/model.gaml", "test")
        
        self.mock_client = MagicMock()
        self.env.gama_client = self.mock_client

    def test_render_placeholder(self):
        """Test render method (currently a placeholder)."""
        # Should not raise an exception
        self.env.render()
        
        # Test with mode parameter
        self.env.render(mode='human')

    def test_close(self):
        """Test close method."""
        self.env.close()
        
        # Verify client close was called
        self.mock_client.close.assert_called_once()

    def test_close_without_client(self):
        """Test close when client doesn't have close method."""
        # Remove gama_client attribute
        delattr(self.env, 'gama_client')
        
        # Should not raise an exception
        self.env.close()

    def test_close_with_client_error(self):
        """Test close when client raises an error."""
        self.mock_client.close.side_effect = Exception("Close error")
        
        # Should handle error gracefully (no exception raised)
        self.env.close()
        
        # Verify cleanup happened (client set to None)
        assert self.env.gama_client is None


class TestGamaEnvIntegration:
    """Integration tests for GamaEnv workflows."""

    @patch('gama_gymnasium.gama_env.GamaClientWrapper')
    @patch('gama_gymnasium.gama_env.SpaceConverter')
    def test_full_environment_lifecycle(self, mock_space_converter_class, mock_client_wrapper_class):
        """Test complete environment lifecycle: init -> reset -> step -> close."""
        # Set up mocks
        mock_client = MagicMock()
        mock_client_wrapper_class.return_value = mock_client
        mock_converter = MagicMock()
        mock_space_converter_class.return_value = mock_converter
        
        # Mock experiment setup
        mock_client.load_experiment.return_value = "exp_123"
        
        # Mock space setup
        obs_space_data = {"type": "Box", "low": -1.0, "high": 1.0, "shape": [2]}
        action_space_data = {"type": "Discrete", "n": 3}
        mock_client.get_observation_space.return_value = obs_space_data
        mock_client.get_action_space.return_value = action_space_data
        
        obs_space = Box(-1.0, 1.0, shape=(2,))
        action_space = Discrete(3)
        mock_converter.map_to_space.side_effect = [obs_space, action_space]
        
        # Create environment
        env = GamaEnv("/path/to/model.gaml", "test")
        
        # Mock reset
        mock_client.get_state.return_value = [0.0, 0.0]
        mock_client.get_info.return_value = {"episode": 1}
        obs_space.from_jsonable = MagicMock(return_value=[np.array([0.0, 0.0])])
        
        # Mock step
        step_data = {
            "State": [0.1, -0.1],
            "Reward": 1.0,
            "Terminated": False,
            "Truncated": False,
            "Info": {"step": 1}
        }
        mock_client.execute_step.return_value = step_data
        action_space.to_jsonable = MagicMock(return_value=[2])
        obs_space.from_jsonable = MagicMock(return_value=[np.array([0.1, -0.1])])
        
        # Test the lifecycle
        # Reset
        state, info = env.reset()
        assert state is not None
        assert isinstance(info, dict)
        
        # Step
        state, reward, terminated, truncated, info = env.step(2)
        assert state is not None
        assert reward == 1.0
        assert terminated == False
        assert truncated == False
        
        # Close
        env.close()
        
        # Verify all components were used correctly
        mock_client.load_experiment.assert_called()
        mock_client.get_observation_space.assert_called()
        mock_client.get_action_space.assert_called()
        mock_client.reset_experiment.assert_called()
        mock_client.execute_step.assert_called()
        mock_client.close.assert_called()

    @patch('gama_gymnasium.gama_env.GamaClientWrapper')
    @patch('gama_gymnasium.gama_env.SpaceConverter')
    def test_multiple_episodes(self, mock_space_converter_class, mock_client_wrapper_class):
        """Test running multiple episodes."""
        # Set up mocks similar to previous test
        mock_client = MagicMock()
        mock_client_wrapper_class.return_value = mock_client
        mock_converter = MagicMock()
        mock_space_converter_class.return_value = mock_converter
        
        # Mock setup
        mock_client.load_experiment.return_value = "exp_123"
        mock_client.get_observation_space.return_value = {"type": "Discrete", "n": 4}
        mock_client.get_action_space.return_value = {"type": "Discrete", "n": 2}
        
        discrete_space = Discrete(4)
        action_space = Discrete(2)
        mock_converter.map_to_space.side_effect = [discrete_space, action_space]
        
        env = GamaEnv("/path/to/model.gaml", "test")
        
        # Run multiple episodes
        num_episodes = 3
        for episode in range(num_episodes):
            # Reset
            mock_client.get_state.return_value = episode
            mock_client.get_info.return_value = {"episode": episode}
            discrete_space.from_jsonable = MagicMock(return_value=[episode])
            
            state, info = env.reset()
            assert info["episode"] == episode
            
            # Run a few steps
            for step in range(5):
                step_data = {
                    "State": step,
                    "Reward": 0.1 * step,
                    "Terminated": step >= 4,  # Terminate on last step
                    "Truncated": False,
                    "Info": {"step": step}
                }
                mock_client.execute_step.return_value = step_data
                action_space.to_jsonable = MagicMock(return_value=[1])
                discrete_space.from_jsonable = MagicMock(return_value=[step])
                
                state, reward, terminated, truncated, info = env.step(1)
                
                if terminated:
                    break
        
        env.close()
        
        # Verify reset was called for each episode
        assert mock_client.reset_experiment.call_count == num_episodes


class TestGamaEnvErrorHandling:
    """Test error handling in GamaEnv."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch('gama_gymnasium.gama_env.GamaClientWrapper'), \
             patch('gama_gymnasium.gama_env.SpaceConverter'), \
             patch.object(GamaEnv, '_setup_experiment'), \
             patch.object(GamaEnv, '_setup_spaces'):
            
            self.env = GamaEnv("/path/to/model.gaml", "test")

    def test_initialization_error_propagation(self):
        """Test that initialization errors are properly propagated."""
        with patch('gama_gymnasium.gama_env.GamaClientWrapper') as mock_client_class:
            mock_client_class.side_effect = GamaEnvironmentError("Connection failed")
            
            with pytest.raises(GamaEnvironmentError):
                GamaEnv("/path/to/model.gaml", "test")

    def test_step_error_handling(self):
        """Test error handling during step execution."""
        self.mock_client = MagicMock()
        self.env.gama_client = self.mock_client
        self.env.experiment_id = "exp_123"
        
        # Mock action space
        self.mock_action_space = MagicMock()
        self.env.action_space = self.mock_action_space
        self.mock_action_space.to_jsonable.return_value = [1]
        
        # Mock client error
        self.mock_client.execute_step.side_effect = GamaEnvironmentError("Step failed")
        
        with pytest.raises(GamaEnvironmentError):
            self.env.step(1)

    def test_reset_error_handling(self):
        """Test error handling during reset."""
        self.mock_client = MagicMock()
        self.env.gama_client = self.mock_client
        self.env.experiment_id = "exp_123"
        
        # Mock client error
        self.mock_client.reset_experiment.side_effect = GamaEnvironmentError("Reset failed")
        
        with pytest.raises(GamaEnvironmentError):
            self.env.reset()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
