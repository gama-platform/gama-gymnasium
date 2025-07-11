"""
Tests for synchronous wrapper functionality.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import numpy as np
from gymnasium.spaces import Discrete, Box

from gama_gymnasium.wrappers.sync import SyncWrapper
from gama_gymnasium.utils.exceptions import GamaGymnasiumError


class TestSyncWrapper:
    """Test synchronous wrapper functionality."""
    
    @pytest.fixture
    def mock_gama_env(self):
        """Create a mock GamaEnv for testing."""
        env = AsyncMock()
        env.observation_space = Discrete(4)
        env.action_space = Discrete(2)
        env.reset = AsyncMock(return_value=(np.array([1]), {"info": "test"}))
        env.step = AsyncMock(return_value=(
            np.array([2]), 1.0, False, False, {"step": 1}
        ))
        env.close = AsyncMock()
        env.render = AsyncMock()
        return env
    
    @patch('gama_gymnasium.wrappers.sync.GamaEnv')
    def test_sync_wrapper_initialization(self, mock_gama_env_class, mock_gama_env):
        """Test synchronous wrapper initialization."""
        # Mock the async environment creation
        async def mock_create_env(*args, **kwargs):
            return mock_gama_env
        
        with patch.object(SyncWrapper, '_create_env', mock_create_env):
            wrapper = SyncWrapper(
                gaml_experiment_path="test.gaml",
                gaml_experiment_name="test_exp"
            )
            
            assert wrapper.observation_space == mock_gama_env.observation_space
            assert wrapper.action_space == mock_gama_env.action_space
    
    @patch('gama_gymnasium.wrappers.sync.GamaEnv')
    def test_sync_wrapper_reset(self, mock_gama_env_class, mock_gama_env):
        """Test synchronous reset method."""
        async def mock_create_env(*args, **kwargs):
            return mock_gama_env
        
        with patch.object(SyncWrapper, '_create_env', mock_create_env):
            wrapper = SyncWrapper(
                gaml_experiment_path="test.gaml",
                gaml_experiment_name="test_exp"
            )
            
            obs, info = wrapper.reset(seed=42)
            
            assert np.array_equal(obs, np.array([1]))
            assert info == {"info": "test"}
            mock_gama_env.reset.assert_called_once_with(seed=42, options=None)
    
    @patch('gama_gymnasium.wrappers.sync.GamaEnv')
    def test_sync_wrapper_step(self, mock_gama_env_class, mock_gama_env):
        """Test synchronous step method."""
        async def mock_create_env(*args, **kwargs):
            return mock_gama_env
        
        with patch.object(SyncWrapper, '_create_env', mock_create_env):
            wrapper = SyncWrapper(
                gaml_experiment_path="test.gaml",
                gaml_experiment_name="test_exp"
            )
            
            obs, reward, terminated, truncated, info = wrapper.step(1)
            
            assert np.array_equal(obs, np.array([2]))
            assert reward == 1.0
            assert terminated == False
            assert truncated == False
            assert info == {"step": 1}
            mock_gama_env.step.assert_called_once_with(1)
    
    @patch('gama_gymnasium.wrappers.sync.GamaEnv')
    def test_sync_wrapper_render(self, mock_gama_env_class, mock_gama_env):
        """Test synchronous render method."""
        async def mock_create_env(*args, **kwargs):
            return mock_gama_env
        
        with patch.object(SyncWrapper, '_create_env', mock_create_env):
            wrapper = SyncWrapper(
                gaml_experiment_path="test.gaml",
                gaml_experiment_name="test_exp"
            )
            
            wrapper.render()
            mock_gama_env.render.assert_called_once_with('human')
    
    @patch('gama_gymnasium.wrappers.sync.GamaEnv')
    def test_sync_wrapper_close(self, mock_gama_env_class, mock_gama_env):
        """Test synchronous close method."""
        async def mock_create_env(*args, **kwargs):
            return mock_gama_env
        
        with patch.object(SyncWrapper, '_create_env', mock_create_env):
            wrapper = SyncWrapper(
                gaml_experiment_path="test.gaml",
                gaml_experiment_name="test_exp"
            )
            
            wrapper.close()
            mock_gama_env.close.assert_called_once()
    
    @patch('gama_gymnasium.wrappers.sync.GamaEnv')
    def test_sync_wrapper_context_manager(self, mock_gama_env_class, mock_gama_env):
        """Test synchronous wrapper as context manager."""
        async def mock_create_env(*args, **kwargs):
            return mock_gama_env
        
        with patch.object(SyncWrapper, '_create_env', mock_create_env):
            with SyncWrapper(
                gaml_experiment_path="test.gaml",
                gaml_experiment_name="test_exp"
            ) as wrapper:
                assert wrapper is not None
            
            # Close should be called automatically
            mock_gama_env.close.assert_called_once()
    
    @patch('gama_gymnasium.wrappers.sync.GamaEnv')
    def test_sync_wrapper_timeout_error(self, mock_gama_env_class):
        """Test timeout error handling."""
        # Create a mock that times out
        async def slow_create_env(*args, **kwargs):
            await asyncio.sleep(10)  # This will timeout
            return Mock()
        
        with patch.object(SyncWrapper, '_create_env', slow_create_env):
            with pytest.raises(GamaGymnasiumError, match="timed out"):
                SyncWrapper(
                    gaml_experiment_path="test.gaml",
                    gaml_experiment_name="test_exp",
                    timeout=0.1  # Very short timeout
                )
    
    @patch('gama_gymnasium.wrappers.sync.GamaEnv')
    def test_sync_wrapper_async_error(self, mock_gama_env_class):
        """Test async operation error handling."""
        # Create a mock that raises an exception
        async def failing_create_env(*args, **kwargs):
            raise ValueError("Test error")
        
        with patch.object(SyncWrapper, '_create_env', failing_create_env):
            with pytest.raises(GamaGymnasiumError, match="Async operation failed"):
                SyncWrapper(
                    gaml_experiment_path="test.gaml",
                    gaml_experiment_name="test_exp"
                )
    
    @patch('gama_gymnasium.wrappers.sync.GamaEnv')
    def test_sync_wrapper_custom_parameters(self, mock_gama_env_class, mock_gama_env):
        """Test wrapper with custom parameters."""
        async def mock_create_env(*args, **kwargs):
            return mock_gama_env
        
        custom_params = [{"name": "param1", "value": 100}]
        
        with patch.object(SyncWrapper, '_create_env', mock_create_env):
            wrapper = SyncWrapper(
                gaml_experiment_path="test.gaml",
                gaml_experiment_name="test_exp",
                gaml_experiment_parameters=custom_params,
                gama_ip_address="localhost",
                gama_port=9999,
                render_mode="rgb_array",
                initialization_wait=1.0,
                timeout=60.0
            )
            
            assert wrapper.render_mode == "rgb_array"
            assert wrapper.timeout == 60.0
    
    def test_make_sync_factory(self):
        """Test make_sync factory function."""
        from gama_gymnasium.wrappers.sync import make_sync
        from gama_gymnasium.core.gama_env import GamaEnv
        
        SyncEnv = make_sync(GamaEnv)
        
        # Should return a class that inherits from SyncWrapper
        assert issubclass(SyncEnv, SyncWrapper)
    
    @patch('gama_gymnasium.wrappers.sync.GamaEnv')
    def test_sync_wrapper_thread_management(self, mock_gama_env_class, mock_gama_env):
        """Test that background thread is properly managed."""
        async def mock_create_env(*args, **kwargs):
            return mock_gama_env
        
        with patch.object(SyncWrapper, '_create_env', mock_create_env):
            wrapper = SyncWrapper(
                gaml_experiment_path="test.gaml",
                gaml_experiment_name="test_exp"
            )
            
            # Thread should be running
            assert wrapper.thread.is_alive()
            assert wrapper.loop is not None
            
            wrapper.close()
            
            # Give thread time to shut down
            time.sleep(0.1)
    
    @patch('gama_gymnasium.wrappers.sync.GamaEnv')
    def test_sync_wrapper_multiple_operations(self, mock_gama_env_class, mock_gama_env):
        """Test multiple synchronous operations."""
        async def mock_create_env(*args, **kwargs):
            return mock_gama_env
        
        with patch.object(SyncWrapper, '_create_env', mock_create_env):
            wrapper = SyncWrapper(
                gaml_experiment_path="test.gaml",
                gaml_experiment_name="test_exp"
            )
            
            # Perform multiple operations
            obs, info = wrapper.reset()
            obs, reward, terminated, truncated, info = wrapper.step(0)
            obs, reward, terminated, truncated, info = wrapper.step(1)
            wrapper.render()
            
            # All async methods should have been called
            assert mock_gama_env.reset.call_count == 1
            assert mock_gama_env.step.call_count == 2
            assert mock_gama_env.render.call_count == 1
            
            wrapper.close()
