"""
Tests for monitoring wrapper functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from gymnasium.spaces import Discrete, Box

from gama_gymnasium.wrappers.monitoring import MonitoringWrapper


class TestMonitoringWrapper:
    """Test monitoring wrapper functionality."""
    
    @pytest.fixture
    def mock_env(self):
        """Create a mock environment for testing."""
        env = Mock()
        env.observation_space = Discrete(4)
        env.action_space = Discrete(2)
        env.reset.return_value = (np.array([0]), {"initial": True})
        env.step.return_value = (np.array([1]), 1.0, False, False, {"step": 1})
        env.close.return_value = None
        env.render.return_value = None
        return env
    
    def test_monitoring_wrapper_initialization(self, mock_env):
        """Test monitoring wrapper initialization."""
        wrapper = MonitoringWrapper(mock_env)
        
        assert wrapper.env == mock_env
        assert wrapper.track_actions == True
        assert wrapper.track_observations == True
        assert wrapper.track_timing == True
        assert wrapper.episode_count == 0
        assert wrapper.step_count == 0
        assert wrapper.total_steps == 0
    
    def test_monitoring_wrapper_with_log_file(self, mock_env):
        """Test monitoring wrapper with log file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            log_file = tmp.name
        
        wrapper = MonitoringWrapper(
            mock_env,
            log_file=log_file,
            save_frequency=1
        )
        
        assert wrapper.log_file == Path(log_file)
        
        # Clean up
        Path(log_file).unlink(missing_ok=True)
    
    def test_monitoring_wrapper_reset(self, mock_env):
        """Test monitoring wrapper reset method."""
        wrapper = MonitoringWrapper(mock_env)
        
        obs, info = wrapper.reset(seed=42)
        
        assert wrapper.episode_count == 1
        assert wrapper.step_count == 0
        assert wrapper.episode_stats["episode"] == 1
        assert wrapper.episode_start_time is not None
        assert np.array_equal(obs, np.array([0]))
        assert info == {"initial": True}
        
        mock_env.reset.assert_called_once_with(seed=42, options=None)
    
    def test_monitoring_wrapper_step(self, mock_env):
        """Test monitoring wrapper step method."""
        wrapper = MonitoringWrapper(mock_env, track_timing=True)
        
        # Reset first
        wrapper.reset()
        
        # Take a step
        obs, reward, terminated, truncated, info = wrapper.step(1)
        
        assert wrapper.step_count == 1
        assert wrapper.total_steps == 1
        assert wrapper.episode_stats["steps"] == 1
        assert wrapper.episode_stats["total_reward"] == 1.0
        assert len(wrapper.episode_stats["rewards"]) == 1
        assert len(wrapper.episode_stats["actions"]) == 1
        assert "step_duration" in info
        
        mock_env.step.assert_called_once_with(1)
    
    def test_monitoring_wrapper_episode_completion(self, mock_env):
        """Test monitoring wrapper when episode completes."""
        mock_env.step.return_value = (np.array([2]), 5.0, True, False, {"final": True})
        
        wrapper = MonitoringWrapper(mock_env)
        wrapper.reset()
        
        obs, reward, terminated, truncated, info = wrapper.step(0)
        
        assert terminated == True
        assert wrapper.episode_stats["terminated"] == True
        assert wrapper.episode_stats["truncated"] == False
        assert len(wrapper.all_episodes) == 1
        
        # Episode should be finalized
        episode_data = wrapper.all_episodes[0]
        assert episode_data["total_reward"] == 5.0
        assert "mean_reward" in episode_data
        assert "duration" in episode_data
    
    def test_monitoring_wrapper_multiple_episodes(self, mock_env):
        """Test monitoring wrapper across multiple episodes."""
        wrapper = MonitoringWrapper(mock_env)
        
        # Episode 1
        wrapper.reset()
        wrapper.step(0)
        
        # Episode 2 (with termination)
        mock_env.step.return_value = (np.array([2]), 3.0, True, False, {})
        wrapper.reset()
        wrapper.step(1)
        
        assert wrapper.episode_count == 2
        assert len(wrapper.all_episodes) == 1  # Only completed episodes
        assert wrapper.all_episodes[0]["total_reward"] == 3.0
    
    def test_monitoring_wrapper_action_tracking(self, mock_env):
        """Test action tracking functionality."""
        wrapper = MonitoringWrapper(mock_env, track_actions=True)
        wrapper.reset()
        
        wrapper.step(0)
        wrapper.step(1)
        
        actions = wrapper.episode_stats["actions"]
        assert len(actions) == 2
        assert actions[0]["action"] == 0
        assert actions[1]["action"] == 1
        assert actions[0]["step"] == 1
        assert actions[1]["step"] == 2
    
    def test_monitoring_wrapper_observation_tracking(self, mock_env):
        """Test observation tracking functionality."""
        wrapper = MonitoringWrapper(mock_env, track_observations=True)
        wrapper.reset()
        wrapper.step(0)
        
        # Check that observations are tracked
        assert len(wrapper.observation_history) >= 2  # Reset + step
    
    def test_monitoring_wrapper_disabled_tracking(self, mock_env):
        """Test wrapper with disabled tracking features."""
        wrapper = MonitoringWrapper(
            mock_env,
            track_actions=False,
            track_observations=False,
            track_timing=False
        )
        
        wrapper.reset()
        obs, reward, terminated, truncated, info = wrapper.step(0)
        
        assert wrapper.episode_stats["actions"] == []
        assert len(wrapper.observation_history) == 0
        assert "step_duration" not in info
    
    def test_monitoring_wrapper_log_saving(self, mock_env):
        """Test log file saving functionality."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            log_file = tmp.name
        
        try:
            wrapper = MonitoringWrapper(
                mock_env,
                log_file=log_file,
                save_frequency=1
            )
            
            # Complete an episode to trigger save
            wrapper.reset()
            mock_env.step.return_value = (np.array([1]), 2.0, True, False, {})
            wrapper.step(0)
            
            # Check that log file was created and contains data
            assert Path(log_file).exists()
            
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            assert "episodes" in log_data
            assert "total_episodes" in log_data
            assert "metadata" in log_data
            assert len(log_data["episodes"]) == 1
            
        finally:
            Path(log_file).unlink(missing_ok=True)
    
    def test_monitoring_wrapper_get_stats(self, mock_env):
        """Test statistics retrieval methods."""
        wrapper = MonitoringWrapper(mock_env)
        wrapper.reset()
        wrapper.step(0)
        
        # Test current episode stats
        current_stats = wrapper.get_episode_stats()
        assert current_stats["episode"] == 1
        assert current_stats["steps"] == 1
        
        # Complete the episode
        mock_env.step.return_value = (np.array([1]), 1.0, True, False, {})
        wrapper.step(1)
        
        # Test all episodes stats
        all_stats = wrapper.get_all_episodes_stats()
        assert len(all_stats) == 1
        
        # Test summary stats
        summary = wrapper.get_summary_stats()
        assert summary["total_episodes"] == 1
        assert summary["total_steps"] == 2
        assert "mean_episode_reward" in summary
        assert "mean_episode_length" in summary
    
    def test_monitoring_wrapper_numpy_serialization(self, mock_env):
        """Test serialization of numpy arrays for logging."""
        wrapper = MonitoringWrapper(mock_env)
        
        # Test with numpy array
        np_array = np.array([1, 2, 3])
        result = wrapper._serialize_for_logging(np_array)
        assert result == [1, 2, 3]
        
        # Test with numpy scalar
        np_int = np.int64(42)
        result = wrapper._serialize_for_logging(np_int)
        assert result == 42
        
        # Test with regular types
        assert wrapper._serialize_for_logging(42) == 42
        assert wrapper._serialize_for_logging("test") == "test"
        assert wrapper._serialize_for_logging([1, 2, 3]) == [1, 2, 3]
    
    def test_monitoring_wrapper_close(self, mock_env):
        """Test monitoring wrapper close method."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            log_file = tmp.name
        
        try:
            wrapper = MonitoringWrapper(mock_env, log_file=log_file)
            wrapper.reset()
            wrapper.step(0)
            
            wrapper.close()
            
            # Should save logs and call parent close
            assert Path(log_file).exists()
            mock_env.close.assert_called_once()
            
        finally:
            Path(log_file).unlink(missing_ok=True)
    
    def test_monitoring_wrapper_box_space_actions(self, mock_env):
        """Test monitoring with continuous action space."""
        mock_env.action_space = Box(low=-1, high=1, shape=(2,))
        mock_env.step.return_value = (np.array([1]), 1.0, False, False, {})
        
        wrapper = MonitoringWrapper(mock_env)
        wrapper.reset()
        
        action = np.array([0.5, -0.3])
        wrapper.step(action)
        
        # Action should be serialized properly
        logged_action = wrapper.episode_stats["actions"][0]["action"]
        assert logged_action == [0.5, -0.3]
    
    def test_monitoring_wrapper_error_handling(self, mock_env):
        """Test error handling in monitoring wrapper."""
        # Test with invalid log file path
        wrapper = MonitoringWrapper(mock_env, log_file="/invalid/path/log.json")
        wrapper.reset()
        mock_env.step.return_value = (np.array([1]), 1.0, True, False, {})
        
        # Should not crash even if logging fails
        wrapper.step(0)
        assert len(wrapper.all_episodes) == 1
    
    @patch('gama_gymnasium.wrappers.monitoring.get_logger')
    def test_monitoring_wrapper_logging(self, mock_get_logger, mock_env):
        """Test that wrapper properly logs events."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        wrapper = MonitoringWrapper(mock_env)
        wrapper.reset()
        
        # Should log episode start
        mock_logger.info.assert_called_with("Episode 1 started")
        
        # Complete episode
        mock_env.step.return_value = (np.array([1]), 5.0, True, False, {})
        wrapper.step(0)
        
        # Should log episode completion
        mock_logger.info.assert_called_with(
            "Episode 1 completed: 1 steps, total reward: 5.00, duration: 0.00s"
        )
