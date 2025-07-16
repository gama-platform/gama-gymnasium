"""
Unit tests for GamaClientWrapper class.

This module contains comprehensive tests for the GamaClientWrapper class,
testing GAMA server communication and experiment management.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any

# Import the classes to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gama_gymnasium.gama_client_wrapper import GamaClientWrapper
from gama_gymnasium.exceptions import (
    GamaEnvironmentError,
    GamaConnectionError,
    GamaCommandError
)


class TestGamaClientWrapperInitialization:
    """Test GamaClientWrapper initialization and connection."""

    @patch('gama_gymnasium.gama_client_wrapper.GamaSyncClient')
    def test_init_with_default_params(self, mock_client_class):
        """Test initialization with default parameters."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        wrapper = GamaClientWrapper()
        
        assert wrapper.ip_address is None
        assert wrapper.port == 6868
        assert wrapper.client == mock_client
        
        # Verify client was created and connected
        mock_client_class.assert_called_once()
        mock_client.connect.assert_called_once()

    @patch('gama_gymnasium.gama_client_wrapper.GamaSyncClient')
    def test_init_with_custom_params(self, mock_client_class):
        """Test initialization with custom parameters."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        wrapper = GamaClientWrapper(ip_address="192.168.1.100", port=8080)
        
        assert wrapper.ip_address == "192.168.1.100"
        assert wrapper.port == 8080
        assert wrapper.client == mock_client

    @patch('gama_gymnasium.gama_client_wrapper.GamaSyncClient')
    def test_connection_failure(self, mock_client_class):
        """Test connection failure handling."""
        mock_client = MagicMock()
        mock_client.connect.side_effect = ConnectionError("Connection failed")
        mock_client_class.return_value = mock_client
        
        with pytest.raises(GamaConnectionError, match="Failed to connect to GAMA server"):
            GamaClientWrapper()

    @patch('gama_gymnasium.gama_client_wrapper.GamaSyncClient')
    def test_async_handlers_setup(self, mock_client_class):
        """Test that async handlers are properly set up."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        wrapper = GamaClientWrapper()
        
        # Verify handlers were passed to client
        args, kwargs = mock_client_class.call_args
        assert len(args) >= 3  # ip, port, and at least one handler
        
        # Check that handlers are callable
        assert callable(args[2])  # async_command_handler
        assert callable(args[3])  # server_message_handler


class TestGamaClientWrapperExperimentManagement:
    """Test experiment loading and management."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch('gama_gymnasium.gama_client_wrapper.GamaSyncClient'):
            self.wrapper = GamaClientWrapper()
        
        # Mock the internal client
        self.mock_client = MagicMock()
        self.wrapper.client = self.mock_client

    def test_load_experiment_success(self):
        """Test successful experiment loading."""
        # Mock successful response
        from gama_client.message_types import MessageTypes
        self.mock_client.load.return_value = {
            "type": MessageTypes.CommandExecutedSuccessfully.value,
            "content": "experiment_123"
        }
        
        exp_id = self.wrapper.load_experiment(
            gaml_path="/path/to/model.gaml",
            experiment_name="test_experiment",
            parameters=[{"param1": "value1"}]
        )
        
        assert exp_id == "experiment_123"
        
        # Verify client.load was called correctly
        self.mock_client.load.assert_called_once_with(
            "/path/to/model.gaml",
            "test_experiment",
            console=False,
            runtime=True,
            parameters=[{"param1": "value1"}]
        )

    def test_load_experiment_failure(self):
        """Test experiment loading failure."""
        # Mock failed response
        self.mock_client.load.return_value = {
            "type": "Error",
            "content": "File not found"
        }
        
        with pytest.raises(GamaCommandError, match="Failed to load experiment"):
            self.wrapper.load_experiment("/invalid/path.gaml", "test")

    def test_load_experiment_with_default_params(self):
        """Test experiment loading with default parameters."""
        from gama_client.message_types import MessageTypes
        self.mock_client.load.return_value = {
            "type": MessageTypes.CommandExecutedSuccessfully.value,
            "content": "experiment_456"
        }
        
        exp_id = self.wrapper.load_experiment("/path/to/model.gaml", "test")
        
        # Verify default parameters were used
        self.mock_client.load.assert_called_once_with(
            "/path/to/model.gaml",
            "test",
            console=False,
            runtime=True,
            parameters=[]
        )

    @patch('gama_gymnasium.gama_client_wrapper.time.sleep')
    def test_load_experiment_with_port_1000_wait(self, mock_sleep):
        """Test that loading waits when using port 1000."""
        from gama_client.message_types import MessageTypes
        
        # Set port to 1000
        self.wrapper.port = 1000
        
        self.mock_client.load.return_value = {
            "type": MessageTypes.CommandExecutedSuccessfully.value,
            "content": "experiment_789"
        }
        
        self.wrapper.load_experiment("/path/to/model.gaml", "test")
        
        # Verify sleep was called for port 1000
        mock_sleep.assert_called_once_with(8)


class TestGamaClientWrapperSpaceManagement:
    """Test space retrieval methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch('gama_gymnasium.gama_client_wrapper.GamaSyncClient'):
            self.wrapper = GamaClientWrapper()
        
        self.mock_client = MagicMock()
        self.wrapper.client = self.mock_client

    def test_get_observation_space(self):
        """Test getting observation space from GAMA."""
        expected_space = {"type": "Box", "low": -1.0, "high": 1.0}
        
        with patch.object(self.wrapper, '_execute_expression') as mock_execute:
            mock_execute.return_value = expected_space
            
            result = self.wrapper.get_observation_space("exp_123")
            
            assert result == expected_space
            mock_execute.assert_called_once_with("exp_123", r"GymAgent[0].observation_space")

    def test_get_action_space(self):
        """Test getting action space from GAMA."""
        expected_space = {"type": "Discrete", "n": 4}
        
        with patch.object(self.wrapper, '_execute_expression') as mock_execute:
            mock_execute.return_value = expected_space
            
            result = self.wrapper.get_action_space("exp_123")
            
            assert result == expected_space
            mock_execute.assert_called_once_with("exp_123", r"GymAgent[0].action_space")


class TestGamaClientWrapperExperimentControl:
    """Test experiment control methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch('gama_gymnasium.gama_client_wrapper.GamaSyncClient'):
            self.wrapper = GamaClientWrapper()
        
        self.mock_client = MagicMock()
        self.wrapper.client = self.mock_client

    def test_reset_experiment_success(self):
        """Test successful experiment reset."""
        from gama_client.message_types import MessageTypes
        
        # Mock reload response
        self.mock_client.reload.return_value = {
            "type": MessageTypes.CommandExecutedSuccessfully.value,
            "content": "Reloaded"
        }
        
        with patch.object(self.wrapper, '_execute_expression') as mock_execute:
            self.wrapper.reset_experiment("exp_123", seed=42)
            
            # Verify reload was called
            self.mock_client.reload.assert_called_once_with("exp_123")
            
            # Verify seed was set
            mock_execute.assert_called_once_with("exp_123", "seed <- 42;")

    def test_reset_experiment_without_seed(self):
        """Test experiment reset without explicit seed."""
        from gama_client.message_types import MessageTypes
        
        self.mock_client.reload.return_value = {
            "type": MessageTypes.CommandExecutedSuccessfully.value,
            "content": "Reloaded"
        }
        
        with patch.object(self.wrapper, '_execute_expression') as mock_execute:
            with patch('numpy.random.random', return_value=0.12345):
                self.wrapper.reset_experiment("exp_123")
                
                # Verify random seed was set
                mock_execute.assert_called_once_with("exp_123", "seed <- 0.12345;")

    def test_reset_experiment_reload_failure(self):
        """Test experiment reset with reload failure."""
        # Mock failed reload
        self.mock_client.reload.return_value = {
            "type": "Error",
            "content": "Reload failed"
        }
        
        with pytest.raises(GamaCommandError, match="Failed to reload experiment"):
            self.wrapper.reset_experiment("exp_123")

    def test_get_state(self):
        """Test getting current state from GAMA."""
        expected_state = [0.1, 0.2, 0.3]
        
        with patch.object(self.wrapper, '_execute_expression') as mock_execute:
            mock_execute.return_value = expected_state
            
            result = self.wrapper.get_state("exp_123")
            
            assert result == expected_state
            mock_execute.assert_called_once_with("exp_123", r"GymAgent[0].state")

    def test_get_info(self):
        """Test getting current info from GAMA."""
        expected_info = {"step": 5, "episode": 1}
        
        with patch.object(self.wrapper, '_execute_expression') as mock_execute:
            mock_execute.return_value = expected_info
            
            result = self.wrapper.get_info("exp_123")
            
            assert result == expected_info
            mock_execute.assert_called_once_with("exp_123", r"GymAgent[0].info")


class TestGamaClientWrapperStepExecution:
    """Test step execution methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch('gama_gymnasium.gama_client_wrapper.GamaSyncClient'):
            self.wrapper = GamaClientWrapper()
        
        self.mock_client = MagicMock()
        self.wrapper.client = self.mock_client

    def test_execute_step_success(self):
        """Test successful step execution."""
        from gama_client.message_types import MessageTypes
        
        # Mock step response
        self.mock_client.step.return_value = {
            "type": MessageTypes.CommandExecutedSuccessfully.value,
            "content": "Step executed"
        }
        
        expected_data = {
            "State": [0.5, -0.3],
            "Reward": 1.0,
            "Terminated": False,
            "Truncated": False,
            "Info": {"step": 1}
        }
        
        with patch.object(self.wrapper, '_execute_expression') as mock_execute:
            mock_execute.side_effect = [None, expected_data]  # Set action, then get data
            
            result = self.wrapper.execute_step("exp_123", [1, 0, 1])
            
            assert result == expected_data
            
            # Verify action was set and step was executed
            assert mock_execute.call_count == 2
            mock_execute.assert_any_call("exp_123", "GymAgent[0].next_action <- [1, 0, 1];")
            mock_execute.assert_any_call("exp_123", r"GymAgent[0].data")
            
            # Verify step was called
            self.mock_client.step.assert_called_once_with("exp_123", sync=True)

    def test_execute_step_failure(self):
        """Test step execution failure."""
        # Mock step failure
        self.mock_client.step.return_value = {
            "type": "Error",
            "content": "Step failed"
        }
        
        with patch.object(self.wrapper, '_execute_expression'):
            with pytest.raises(GamaCommandError, match="Failed to execute step"):
                self.wrapper.execute_step("exp_123", [1, 0])

    def test_execute_step_with_different_action_types(self):
        """Test step execution with different action types."""
        from gama_client.message_types import MessageTypes
        
        self.mock_client.step.return_value = {
            "type": MessageTypes.CommandExecutedSuccessfully.value,
            "content": "Step executed"
        }
        
        test_actions = [
            42,  # Single integer
            [1, 2, 3],  # List
            {"action": "move", "direction": "north"},  # Dictionary
            "text_action"  # String
        ]
        
        with patch.object(self.wrapper, '_execute_expression') as mock_execute:
            mock_execute.return_value = {"State": [], "Reward": 0, "Terminated": False, "Truncated": False, "Info": {}}
            
            for action in test_actions:
                self.wrapper.execute_step("exp_123", action)
                
                # Verify action was set correctly
                expected_call = f"GymAgent[0].next_action <- {action};"
                mock_execute.assert_any_call("exp_123", expected_call)


class TestGamaClientWrapperExpressionExecution:
    """Test expression execution methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch('gama_gymnasium.gama_client_wrapper.GamaSyncClient'):
            self.wrapper = GamaClientWrapper()
        
        self.mock_client = MagicMock()
        self.wrapper.client = self.mock_client

    def test_execute_expression_success(self):
        """Test successful expression execution."""
        from gama_client.message_types import MessageTypes
        
        # Mock expression response
        self.mock_client.expression.return_value = {
            "type": MessageTypes.CommandExecutedSuccessfully.value,
            "content": "expression_result"
        }
        
        result = self.wrapper._execute_expression("exp_123", "world.agents.length")
        
        assert result == "expression_result"
        self.mock_client.expression.assert_called_once_with("exp_123", "world.agents.length")

    def test_execute_expression_failure(self):
        """Test expression execution failure."""
        # Mock expression failure
        self.mock_client.expression.return_value = {
            "type": "Error",
            "content": "Invalid expression"
        }
        
        with pytest.raises(GamaCommandError, match="Failed to execute expression"):
            self.wrapper._execute_expression("exp_123", "invalid.expression")

    def test_execute_expression_with_complex_expressions(self):
        """Test execution of complex GAMA expressions."""
        from gama_client.message_types import MessageTypes
        
        complex_expressions = [
            "GymAgent[0].state",
            "world.agents where (each.species = GymAgent)",
            "sum(GymAgent collect each.energy)",
            "matrix([[1,2],[3,4]])",
            "map(['key1'::10, 'key2'::20])"
        ]
        
        for expr in complex_expressions:
            self.mock_client.expression.return_value = {
                "type": MessageTypes.CommandExecutedSuccessfully.value,
                "content": f"result_for_{expr}"
            }
            
            result = self.wrapper._execute_expression("exp_123", expr)
            assert result == f"result_for_{expr}"


class TestGamaClientWrapperCleanup:
    """Test cleanup and resource management."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch('gama_gymnasium.gama_client_wrapper.GamaSyncClient'):
            self.wrapper = GamaClientWrapper()
        
        self.mock_client = MagicMock()
        self.wrapper.client = self.mock_client

    def test_close_connection(self):
        """Test closing the connection to GAMA server."""
        self.wrapper.close()
        
        # Verify close_connection was called
        self.mock_client.close_connection.assert_called_once()

    def test_close_without_client(self):
        """Test closing when no client exists."""
        self.wrapper.client = None
        
        # Should not raise an exception
        self.wrapper.close()

    def test_close_with_client_error(self):
        """Test closing when client raises an error."""
        self.mock_client.close_connection.side_effect = Exception("Close error")
        
        # Should not raise an exception (graceful cleanup)
        self.wrapper.close()

        # Verify client was set to None for cleanup
        assert self.wrapper.client is None

class TestGamaClientWrapperAsyncHandlers:
    """Test async message handlers."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch('gama_gymnasium.gama_client_wrapper.GamaSyncClient'):
            self.wrapper = GamaClientWrapper()

    @pytest.mark.asyncio
    async def test_async_command_handler(self):
        """Test async command handler."""
        test_message = {"type": "command_response", "content": "test"}
        
        # Should not raise exception
        await self.wrapper._async_command_handler(test_message)

    @pytest.mark.asyncio
    async def test_server_message_handler(self):
        """Test server message handler."""
        test_message = {"type": "server_message", "content": "test"}
        
        # Should not raise exception
        await self.wrapper._server_message_handler(test_message)


class TestGamaClientWrapperIntegration:
    """Integration tests for GamaClientWrapper workflows."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch('gama_gymnasium.gama_client_wrapper.GamaSyncClient'):
            self.wrapper = GamaClientWrapper()
        
        self.mock_client = MagicMock()
        self.wrapper.client = self.mock_client

    def test_full_experiment_lifecycle(self):
        """Test complete experiment lifecycle."""
        from gama_client.message_types import MessageTypes
        
        # Mock all necessary responses
        self.mock_client.load.return_value = {
            "type": MessageTypes.CommandExecutedSuccessfully.value,
            "content": "exp_123"
        }
        
        self.mock_client.reload.return_value = {
            "type": MessageTypes.CommandExecutedSuccessfully.value,
            "content": "Reloaded"
        }
        
        self.mock_client.step.return_value = {
            "type": MessageTypes.CommandExecutedSuccessfully.value,
            "content": "Step executed"
        }
        
        with patch.object(self.wrapper, '_execute_expression') as mock_execute:
            # Mock space definitions
            mock_execute.side_effect = [
                {"type": "Box", "low": -1, "high": 1},  # observation_space
                {"type": "Discrete", "n": 4},  # action_space
                None,  # seed setting
                [0.1, 0.2],  # initial state
                {"episode": 1},  # initial info
                None,  # action setting
                {  # step data
                    "State": [0.2, 0.3],
                    "Reward": 1.0,
                    "Terminated": False,
                    "Truncated": False,
                    "Info": {"step": 1}
                }
            ]
            
            # Load experiment
            exp_id = self.wrapper.load_experiment("/path/to/model.gaml", "test")
            assert exp_id == "exp_123"
            
            # Get spaces
            obs_space = self.wrapper.get_observation_space(exp_id)
            action_space = self.wrapper.get_action_space(exp_id)
            
            # Reset experiment
            self.wrapper.reset_experiment(exp_id, seed=42)
            
            # Get initial state
            state = self.wrapper.get_state(exp_id)
            info = self.wrapper.get_info(exp_id)
            
            # Execute step
            step_data = self.wrapper.execute_step(exp_id, 2)
            
            # Close
            self.wrapper.close()
            
            # Verify all operations completed
            assert obs_space == {"type": "Box", "low": -1, "high": 1}
            assert action_space == {"type": "Discrete", "n": 4}
            assert state == [0.1, 0.2]
            assert info == {"episode": 1}
            assert step_data["Reward"] == 1.0

    def test_error_recovery_scenarios(self):
        """Test error recovery in various scenarios."""
        # Test load failure followed by successful retry
        self.mock_client.load.side_effect = [
            {"type": "Error", "content": "First attempt failed"},
            {"type": "CommandExecutedSuccessfully", "content": "exp_123"}
        ]
        
        # First attempt should fail
        with pytest.raises(GamaCommandError):
            self.wrapper.load_experiment("/path/to/model.gaml", "test")
        
        # Second attempt should succeed
        exp_id = self.wrapper.load_experiment("/path/to/model.gaml", "test")
        assert exp_id == "exp_123"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
