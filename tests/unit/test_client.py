"""
Tests for GAMA client functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

# Mock the gama_client imports since they may not be available in tests
with patch.dict('sys.modules', {
    'gama_client.sync_client': MagicMock(),
    'gama_client.message_types': MagicMock()
}):
    from gama_gymnasium.core.client import GamaClient
    from gama_gymnasium.utils.exceptions import GamaGymnasiumError


class TestGamaClient:
    """Test GAMA client functionality."""
    
    @patch('gama_gymnasium.core.client.GamaClient')
    def test_client_initialization(self, mock_sync_client):
        """Test client initialization."""
        client = GamaClient()
        
        assert client.is_connected == False
        mock_sync_client.assert_called_once()
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    def test_connect(self, mock_sync_client):
        """Test connection to GAMA server."""
        client = GamaClient()
        client.connect()
        
        assert client.is_connected == True
        client.client.connect.assert_called_once()
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    def test_disconnect(self, mock_sync_client):
        """Test disconnection from GAMA server."""
        client = GamaClient()
        client.connect()
        client.disconnect()
        
        assert client.is_connected == False
        client.client.close_connection.assert_called_once()
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    def test_context_manager(self, mock_sync_client):
        """Test client as context manager."""
        with GamaClient() as client:
            assert client.is_connected == True
        
        client.client.close_connection.assert_called_once()
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    def test_load_experiment_success(self, mock_sync_client):
        """Test successful experiment loading."""
        mock_response = {
            "type": "CommandExecutedSuccessfully",
            "content": "experiment_id_123"
        }
        
        client = GamaClient()
        client.connect()
        client.client.load.return_value = mock_response
        
        # Mock MessageTypes
        with patch('gama_gymnasium.core.client.MessageTypes') as mock_types:
            mock_types.CommandExecutedSuccessfully.value = "CommandExecutedSuccessfully"
            
            exp_id = client.load_experiment("test.gaml", "test_exp")
            assert exp_id == "experiment_id_123"
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    def test_load_experiment_failure(self, mock_sync_client):
        """Test failed experiment loading."""
        mock_response = {
            "type": "CommandExecutionError",
            "content": "Error loading experiment"
        }
        
        client = GamaClient()
        client.connect()
        client.client.load.return_value = mock_response
        
        with patch('gama_gymnasium.core.client.MessageTypes') as mock_types:
            mock_types.CommandExecutedSuccessfully.value = "CommandExecutedSuccessfully"
            
            with pytest.raises(Exception):
                client.load_experiment("test.gaml", "test_exp")
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    def test_execute_expression_not_connected(self, mock_sync_client):
        """Test executing expression when not connected."""
        client = GamaClient()
        
        with pytest.raises(RuntimeError):
            client.execute_expression("exp_id", "some_expression")
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    def test_execute_expression_success(self, mock_sync_client):
        """Test successful expression execution."""
        mock_response = {
            "type": "CommandExecutedSuccessfully",
            "content": {"x": 10, "y": 20}
        }
        
        client = GamaClient()
        client.connect()
        client.client.expression.return_value = mock_response
        
        with patch('gama_gymnasium.core.client.MessageTypes') as mock_types:
            mock_types.CommandExecutedSuccessfully.value = "CommandExecutedSuccessfully"
            
            result = client.execute_expression("exp_id", "position")
            assert result == {"x": 10, "y": 20}
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    def test_step_simulation_success(self, mock_sync_client):
        """Test successful simulation step."""
        mock_response = {
            "type": "CommandExecutedSuccessfully",
            "content": "Step completed"
        }
        
        client = GamaClient()
        client.connect()
        client.client.step.return_value = mock_response
        
        with patch('gama_gymnasium.core.client.MessageTypes') as mock_types:
            mock_types.CommandExecutedSuccessfully.value = "CommandExecutedSuccessfully"
            
            # Should not raise any exception
            client.step_simulation("exp_id", sync=True)
            client.client.step.assert_called_once_with("exp_id", sync=True)
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    def test_stop_experiment_success(self, mock_sync_client):
        """Test successful experiment stopping."""
        mock_response = {
            "type": "CommandExecutedSuccessfully",
            "content": "Experiment stopped"
        }
        
        client = GamaClient()
        client.connect()
        client.client.stop.return_value = mock_response
        
        with patch('gama_gymnasium.core.client.MessageTypes') as mock_types:
            mock_types.CommandExecutedSuccessfully.value = "CommandExecutedSuccessfully"
            
            client.stop_experiment("exp_id")
            client.client.stop.assert_called_once_with("exp_id")
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    def test_reload_experiment_success(self, mock_sync_client):
        """Test successful experiment reload."""
        mock_stop_response = {
            "type": "CommandExecutedSuccessfully",
            "content": "Experiment stopped"
        }
        mock_load_response = {
            "type": "CommandExecutedSuccessfully",
            "content": "new_experiment_id"
        }
        
        client = GamaClient()
        client.connect()
        client.client.stop.return_value = mock_stop_response
        client.client.load.return_value = mock_load_response
        
        with patch('gama_gymnasium.core.client.MessageTypes') as mock_types:
            mock_types.CommandExecutedSuccessfully.value = "CommandExecutedSuccessfully"
            
            new_id = client.reload_experiment("old_id", "test.gaml", "test_exp")
            assert new_id == "new_experiment_id"
            
            client.client.stop.assert_called_once_with("old_id")
            client.client.load.assert_called_once()
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    def test_client_with_custom_handlers(self, mock_sync_client):
        """Test client initialization with custom handlers."""
        async def custom_async_handler(msg):
            pass
        
        async def custom_message_handler(msg):
            pass
        
        client = GamaClient(
            ip_address="localhost",
            port=9999,
            async_handler=custom_async_handler,
            message_handler=custom_message_handler
        )
        
        mock_sync_client.assert_called_once_with(
            "localhost", 9999, custom_async_handler, custom_message_handler
        )
