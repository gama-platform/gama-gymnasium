"""
Integration tests for GAMA-Gymnasium.

These tests verify that different components work together correctly.
They may require a running GAMA server or use more realistic mocks.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from gymnasium.spaces import Discrete, Box

from gama_gymnasium import GamaEnv, SyncWrapper, MonitoringWrapper
from gama_gymnasium.core.client import GamaClient
from gama_gymnasium.spaces.converters import map_to_space


class TestGamaGymnasiumIntegration:
    """Integration tests for the full GAMA-Gymnasium stack."""
    
    @pytest.fixture
    def mock_gama_responses(self):
        """Mock GAMA server responses."""
        return {
            "load": {
                "type": "CommandExecutedSuccessfully",
                "content": "experiment_123"
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
            "reset_state": {
                "type": "CommandExecutedSuccessfully",
                "content": 0
            },
            "reset_info": {
                "type": "CommandExecutedSuccessfully",
                "content": {"episode": 0}
            },
            "step_data": {
                "type": "CommandExecutedSuccessfully",
                "content": {
                    "State": 1,
                    "Reward": 1.0,
                    "Terminated": False,
                    "Truncated": False,
                    "Info": {"step": 1}
                }
            }
        }
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    @patch('gama_gymnasium.core.client.MessageTypes')
    def test_gama_env_full_workflow(self, mock_message_types, mock_sync_client, mock_gama_responses):
        """Test complete GamaEnv workflow with mocked GAMA server."""
        # Setup mocks
        mock_message_types.CommandExecutedSuccessfully.value = "CommandExecutedSuccessfully"
        
        mock_client = mock_sync_client.return_value
        mock_client.load.return_value = mock_gama_responses["load"]
        mock_client.expression.side_effect = [
            mock_gama_responses["observation_space"],
            mock_gama_responses["action_space"],
            mock_gama_responses["reset_state"],
            mock_gama_responses["reset_info"],
            mock_gama_responses["step_data"]
        ]
        mock_client.step.return_value = {"type": "CommandExecutedSuccessfully", "content": ""}
        mock_client.stop.return_value = {"type": "CommandExecutedSuccessfully", "content": ""}
        
        # Test GamaEnv
        env = GamaEnv(
            gaml_experiment_path="test.gaml",
            gaml_experiment_name="test_experiment"
        )
        
        # Verify spaces were created correctly
        assert isinstance(env.observation_space, Discrete)
        assert env.observation_space.n == 4
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 2
        
        # Test reset
        obs, info = env.reset(seed=42)
        assert obs == 0
        assert info == {"episode": 0}
        
        # Test step
        obs, reward, terminated, truncated, info = env.step(1)
        assert obs == 1
        assert reward == 1.0
        assert terminated == False
        assert truncated == False
        assert info == {"step": 1}
        
        # Test close
        env.close()
        mock_client.close_connection.assert_called_once()
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    @patch('gama_gymnasium.core.client.MessageTypes')
    def test_sync_wrapper_integration(self, mock_message_types, mock_sync_client, mock_gama_responses):
        """Test SyncWrapper with GamaEnv integration."""
        # Setup mocks (same as above)
        mock_message_types.CommandExecutedSuccessfully.value = "CommandExecutedSuccessfully"
        mock_client = mock_sync_client.return_value
        mock_client.load.return_value = mock_gama_responses["load"]
        mock_client.expression.side_effect = [
            mock_gama_responses["observation_space"],
            mock_gama_responses["action_space"],
            mock_gama_responses["reset_state"],
            mock_gama_responses["reset_info"],
            mock_gama_responses["step_data"]
        ]
        mock_client.step.return_value = {"type": "CommandExecutedSuccessfully", "content": ""}
        mock_client.stop.return_value = {"type": "CommandExecutedSuccessfully", "content": ""}
        
        # Test SyncWrapper
        with SyncWrapper(
            gaml_experiment_path="test.gaml",
            gaml_experiment_name="test_experiment"
        ) as env:
            # Should work synchronously
            obs, info = env.reset()
            assert obs == 0
            
            obs, reward, terminated, truncated, info = env.step(1)
            assert reward == 1.0
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    @patch('gama_gymnasium.core.client.MessageTypes')
    def test_monitoring_wrapper_integration(self, mock_message_types, mock_sync_client, mock_gama_responses):
        """Test MonitoringWrapper with GamaEnv integration."""
        # Setup mocks
        mock_message_types.CommandExecutedSuccessfully.value = "CommandExecutedSuccessfully"
        mock_client = mock_sync_client.return_value
        mock_client.load.return_value = mock_gama_responses["load"]
        mock_client.expression.side_effect = [
            mock_gama_responses["observation_space"],
            mock_gama_responses["action_space"],
            mock_gama_responses["reset_state"],
            mock_gama_responses["reset_info"],
            mock_gama_responses["step_data"]
        ]
        mock_client.step.return_value = {"type": "CommandExecutedSuccessfully", "content": ""}
        mock_client.stop.return_value = {"type": "CommandExecutedSuccessfully", "content": ""}
        
        # Create GamaEnv
        base_env = GamaEnv(
            gaml_experiment_path="test.gaml",
            gaml_experiment_name="test_experiment"
        )
        
        # Wrap with monitoring
        env = MonitoringWrapper(base_env)
        
        # Test that monitoring tracks interactions
        env.reset()
        assert env.episode_count == 1
        
        env.step(1)
        assert env.step_count == 1
        assert len(env.episode_stats["actions"]) == 1
        
        env.close()
    
    def test_space_conversion_integration(self):
        """Test space conversion with various GAMA space definitions."""
        test_cases = [
            # Discrete space
            {
                "definition": {"type": "Discrete", "n": 5},
                "expected_type": Discrete,
                "expected_n": 5
            },
            # Box space
            {
                "definition": {
                    "type": "Box",
                    "low": [0.0, -1.0],
                    "high": [1.0, 1.0],
                    "shape": [2]
                },
                "expected_type": Box,
                "expected_shape": (2,)
            },
            # Box with infinity
            {
                "definition": {
                    "type": "Box",
                    "low": "-Infinity",
                    "high": "Infinity",
                    "shape": [3]
                },
                "expected_type": Box,
                "expected_shape": (3,)
            }
        ]
        
        for case in test_cases:
            space = map_to_space(case["definition"])
            assert isinstance(space, case["expected_type"])
            
            if "expected_n" in case:
                assert space.n == case["expected_n"]
            if "expected_shape" in case:
                assert space.shape == case["expected_shape"]
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    @patch('gama_gymnasium.core.client.MessageTypes')
    def test_error_propagation(self, mock_message_types, mock_sync_client):
        """Test that errors are properly propagated through the stack."""
        # Setup mock to fail on load
        mock_message_types.CommandExecutedSuccessfully.value = "CommandExecutedSuccessfully"
        mock_client = mock_sync_client.return_value
        mock_client.load.return_value = {
            "type": "CommandExecutionError",
            "content": "Failed to load experiment"
        }
        
        # Should raise exception during initialization
        with pytest.raises(Exception, match="Failed to load experiment"):
            GamaEnv(
                gaml_experiment_path="nonexistent.gaml",
                gaml_experiment_name="test_experiment"
            )
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    @patch('gama_gymnasium.core.client.MessageTypes')
    def test_complex_action_handling(self, mock_message_types, mock_sync_client, mock_gama_responses):
        """Test handling of complex action types."""
        # Setup mocks for Box action space
        box_action_space = {
            "type": "Box",
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
            "shape": [2]
        }
        
        mock_message_types.CommandExecutedSuccessfully.value = "CommandExecutedSuccessfully"
        mock_client = mock_sync_client.return_value
        mock_client.load.return_value = mock_gama_responses["load"]
        mock_client.expression.side_effect = [
            mock_gama_responses["observation_space"],
            {"type": "CommandExecutedSuccessfully", "content": box_action_space},
            mock_gama_responses["reset_state"],
            mock_gama_responses["reset_info"],
            mock_gama_responses["step_data"]
        ]
        mock_client.step.return_value = {"type": "CommandExecutedSuccessfully", "content": ""}
        
        env = GamaEnv(
            gaml_experiment_path="test.gaml",
            gaml_experiment_name="test_experiment"
        )
        
        # Test with continuous action
        env.reset()
        continuous_action = np.array([0.5, -0.3])
        obs, reward, terminated, truncated, info = env.step(continuous_action)
        
        # Should handle continuous actions without error
        assert reward == 1.0
        
        env.close()
    
    def test_client_context_manager(self):
        """Test GamaClient as context manager."""
        with patch('gama_gymnasium.core.client.GamaSyncClient') as mock_sync_client:
            mock_client = mock_sync_client.return_value
            
            with GamaClient() as client:
                assert client.is_connected == True
                mock_client.connect.assert_called_once()
            
            # Should automatically disconnect
            mock_client.close_connection.assert_called_once()
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    @patch('gama_gymnasium.core.client.MessageTypes')
    def test_experiment_reload(self, mock_message_types, mock_sync_client, mock_gama_responses):
        """Test experiment reload functionality."""
        mock_message_types.CommandExecutedSuccessfully.value = "CommandExecutedSuccessfully"
        mock_client = mock_sync_client.return_value
        
        # Setup responses for load, then reload
        mock_client.load.side_effect = [
            mock_gama_responses["load"],
            {"type": "CommandExecutedSuccessfully", "content": "new_experiment_456"}
        ]
        mock_client.stop.return_value = {"type": "CommandExecutedSuccessfully", "content": ""}
        mock_client.expression.side_effect = [
            mock_gama_responses["observation_space"],
            mock_gama_responses["action_space"],
            mock_gama_responses["reset_state"],
            mock_gama_responses["reset_info"]
        ]
        
        env = GamaEnv(
            gaml_experiment_path="test.gaml",
            gaml_experiment_name="test_experiment"
        )
        
        original_id = env.experiment_id
        
        # Reset should reload the experiment
        env.reset()
        
        # Should have called stop and load
        mock_client.stop.assert_called_with(original_id)
        assert mock_client.load.call_count == 2
        
        env.close()
