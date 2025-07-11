"""
Monitoring Wrapper for GAMA-Gymnasium

This module provides monitoring and logging capabilities for GAMA environments,
including performance metrics, action/observation tracking, and episode statistics.
"""

import time
import json
from typing import Any, Dict, Optional, List, SupportsFloat, Tuple
from pathlib import Path
import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from ..utils.logging import get_logger


class MonitoringWrapper(gym.Wrapper):
    """
    Wrapper that monitors and logs environment interactions.
    
    This wrapper tracks episode statistics, action distributions,
    timing information, and can save logs to files for analysis.
    
    Attributes:
        log_file (Path, optional): Path to log file
        track_actions (bool): Whether to track action distributions
        track_observations (bool): Whether to track observation statistics
        track_timing (bool): Whether to track timing information
        episode_stats (dict): Current episode statistics
        all_episodes (list): Statistics for all episodes
    """
    
    def __init__(
        self,
        env: gym.Env,
        log_file: Optional[str] = None,
        track_actions: bool = True,
        track_observations: bool = True,
        track_timing: bool = True,
        save_frequency: int = 10
    ):
        """
        Initialize the monitoring wrapper.
        
        Args:
            env (gym.Env): Environment to wrap
            log_file (str, optional): Path to save logs
            track_actions (bool): Whether to track actions
            track_observations (bool): Whether to track observations
            track_timing (bool): Whether to track timing
            save_frequency (int): How often to save logs (episodes)
        """
        super().__init__(env)
        
        self.log_file = Path(log_file) if log_file else None
        self.track_actions = track_actions
        self.track_observations = track_observations
        self.track_timing = track_timing
        self.save_frequency = save_frequency
        
        self.logger = get_logger(__name__)
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.total_steps = 0
        
        # Current episode stats
        self.episode_stats = self._init_episode_stats()
        
        # Historical data
        self.all_episodes = []
        self.action_history = []
        self.observation_history = []
        
        # Timing
        self.episode_start_time = None
        self.step_start_time = None
        
        self.logger.info("MonitoringWrapper initialized")
    
    def _init_episode_stats(self) -> Dict[str, Any]:
        """Initialize episode statistics."""
        return {
            "episode": self.episode_count,
            "steps": 0,
            "total_reward": 0.0,
            "start_time": None,
            "end_time": None,
            "duration": 0.0,
            "actions": [],
            "rewards": [],
            "terminated": False,
            "truncated": False
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset the environment and start new episode tracking."""
        
        # Save previous episode if it exists
        if self.episode_count > 0:
            self._finalize_episode()
        
        # Start new episode
        self.episode_count += 1
        self.step_count = 0
        self.episode_stats = self._init_episode_stats()
        self.episode_stats["episode"] = self.episode_count
        
        if self.track_timing:
            self.episode_start_time = time.time()
            self.episode_stats["start_time"] = self.episode_start_time
        
        # Reset the environment
        observation, info = self.env.reset(seed=seed, options=options)
        
        # Track initial observation
        if self.track_observations:
            self._track_observation(observation)
        
        self.logger.info(f"Episode {self.episode_count} started")
        
        return observation, info
    
    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute a step and track statistics."""
        
        if self.track_timing:
            self.step_start_time = time.time()
        
        # Track action
        if self.track_actions:
            self._track_action(action)
        
        # Execute step
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Update statistics
        self.step_count += 1
        self.total_steps += 1
        self.episode_stats["steps"] = self.step_count
        self.episode_stats["total_reward"] += reward
        self.episode_stats["rewards"].append(float(reward))
        
        # Track observation
        if self.track_observations:
            self._track_observation(observation)
        
        # Track timing
        if self.track_timing:
            step_duration = time.time() - self.step_start_time
            info["step_duration"] = step_duration
        
        # Check if episode ended
        if terminated or truncated:
            self.episode_stats["terminated"] = terminated
            self.episode_stats["truncated"] = truncated
            self._finalize_episode()
        
        return observation, reward, terminated, truncated, info
    
    def _track_action(self, action: ActType) -> None:
        """Track action statistics."""
        action_data = {
            "step": self.step_count,
            "action": self._serialize_for_logging(action)
        }
        
        self.episode_stats["actions"].append(action_data)
        self.action_history.append(action_data)
    
    def _track_observation(self, observation: ObsType) -> None:
        """Track observation statistics."""
        obs_data = {
            "step": self.step_count,
            "observation": self._serialize_for_logging(observation)
        }
        
        self.observation_history.append(obs_data)
    
    def _finalize_episode(self) -> None:
        """Finalize episode statistics."""
        if self.track_timing and self.episode_start_time:
            self.episode_stats["end_time"] = time.time()
            self.episode_stats["duration"] = (
                self.episode_stats["end_time"] - self.episode_stats["start_time"]
            )
        
        # Calculate episode statistics
        rewards = self.episode_stats["rewards"]
        if rewards:
            self.episode_stats["mean_reward"] = np.mean(rewards)
            self.episode_stats["std_reward"] = np.std(rewards)
            self.episode_stats["min_reward"] = np.min(rewards)
            self.episode_stats["max_reward"] = np.max(rewards)
        
        # Add to history
        self.all_episodes.append(self.episode_stats.copy())
        
        # Log episode summary
        self.logger.info(
            f"Episode {self.episode_count} completed: "
            f"{self.step_count} steps, "
            f"total reward: {self.episode_stats['total_reward']:.2f}, "
            f"duration: {self.episode_stats.get('duration', 0):.2f}s"
        )
        
        # Save logs if needed
        if self.episode_count % self.save_frequency == 0:
            self._save_logs()
    
    def _save_logs(self) -> None:
        """Save logs to file."""
        if self.log_file is None:
            return
        
        log_data = {
            "episodes": self.all_episodes,
            "total_episodes": self.episode_count,
            "total_steps": self.total_steps,
            "metadata": {
                "environment": str(type(self.env).__name__),
                "action_space": str(self.action_space),
                "observation_space": str(self.observation_space),
                "tracking": {
                    "actions": self.track_actions,
                    "observations": self.track_observations,
                    "timing": self.track_timing
                }
            }
        }
        
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
            
            self.logger.info(f"Logs saved to {self.log_file}")
        except Exception as e:
            self.logger.error(f"Failed to save logs: {e}")
    
    def _serialize_for_logging(self, obj: Any) -> Any:
        """Serialize object for JSON logging."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for the current episode."""
        return self.episode_stats.copy()
    
    def get_all_episodes_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all episodes."""
        return self.all_episodes.copy()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all episodes."""
        if not self.all_episodes:
            return {}
        
        total_rewards = [ep["total_reward"] for ep in self.all_episodes]
        episode_lengths = [ep["steps"] for ep in self.all_episodes]
        durations = [ep.get("duration", 0) for ep in self.all_episodes]
        
        return {
            "total_episodes": len(self.all_episodes),
            "total_steps": self.total_steps,
            "mean_episode_reward": np.mean(total_rewards),
            "std_episode_reward": np.std(total_rewards),
            "min_episode_reward": np.min(total_rewards),
            "max_episode_reward": np.max(total_rewards),
            "mean_episode_length": np.mean(episode_lengths),
            "std_episode_length": np.std(episode_lengths),
            "mean_episode_duration": np.mean(durations),
            "terminated_episodes": sum(1 for ep in self.all_episodes if ep["terminated"]),
            "truncated_episodes": sum(1 for ep in self.all_episodes if ep["truncated"])
        }
    
    def close(self) -> None:
        """Close the wrapper and save final logs."""
        if self.episode_count > 0 and self.step_count > 0:
            self._finalize_episode()
        
        self._save_logs()
        self.logger.info("MonitoringWrapper closed")
        
        super().close()
