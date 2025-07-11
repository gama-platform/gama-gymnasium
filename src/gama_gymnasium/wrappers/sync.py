"""
Synchronous Wrapper for GAMA-Gymnasium

This module provides a synchronous wrapper for the asynchronous GamaEnv,
making it compatible with synchronous RL libraries like Stable-Baselines3.
"""

import asyncio
from typing import Any, Dict, Optional, SupportsFloat, Tuple
from threading import Thread
import queue
import time

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from ..core.gama_env import GamaEnv
from ..utils.exceptions import GamaGymnasiumError


class SyncWrapper(gym.Env):
    """
    Synchronous wrapper for GamaEnv.
    
    This wrapper allows the asynchronous GamaEnv to be used with
    synchronous RL libraries by handling the async/await pattern
    internally using a background thread and event loop.
    
    Attributes:
        env (GamaEnv): The wrapped asynchronous environment
        loop (asyncio.AbstractEventLoop): Event loop for async operations
        thread (Thread): Background thread running the event loop
    """
    
    def __init__(
        self,
        gaml_experiment_path: str,
        gaml_experiment_name: str,
        gaml_experiment_parameters: Optional[list[dict[str, Any]]] = None,
        gama_ip_address: Optional[str] = None,
        gama_port: int = 6868,
        render_mode: Optional[str] = None,
        initialization_wait: float = 0.0,
        timeout: float = 30.0
    ):
        """
        Initialize the synchronous wrapper.
        
        Args:
            gaml_experiment_path (str): Path to the GAML model file
            gaml_experiment_name (str): Name of the experiment
            gaml_experiment_parameters (list, optional): Experiment parameters
            gama_ip_address (str, optional): IP address of GAMA server
            gama_port (int): Port number for GAMA server communication
            render_mode (str, optional): Rendering mode
            initialization_wait (float): Time to wait for environment initialization
            timeout (float): Timeout for operations in seconds
        """
        self.timeout = timeout
        self._result_queue = queue.Queue()
        self._command_queue = queue.Queue()
        
        # Start the async event loop in a background thread
        self.loop = None
        self.thread = Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()
        
        # Wait for loop to be ready
        while self.loop is None:
            time.sleep(0.01)
        
        # Create the async environment
        self.env = self._run_sync(
            self._create_env,
            gaml_experiment_path,
            gaml_experiment_name,
            gaml_experiment_parameters,
            gama_ip_address,
            gama_port,
            render_mode,
            initialization_wait
        )
        
        # Expose the spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.render_mode = render_mode
    
    def _run_async_loop(self):
        """Run the async event loop in the background thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def _run_sync(self, coro_func, *args, **kwargs):
        """
        Run an async function synchronously.
        
        Args:
            coro_func: Async function to run
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the async function
            
        Raises:
            GamaGymnasiumError: If operation times out or fails
        """
        if self.loop is None:
            raise GamaGymnasiumError("Event loop not ready")
        
        future = asyncio.run_coroutine_threadsafe(
            coro_func(*args, **kwargs), self.loop
        )
        
        try:
            return future.result(timeout=self.timeout)
        except asyncio.TimeoutError:
            raise GamaGymnasiumError(f"Operation timed out after {self.timeout} seconds")
        except Exception as e:
            raise GamaGymnasiumError(f"Async operation failed: {e}")
    
    async def _create_env(
        self,
        gaml_experiment_path: str,
        gaml_experiment_name: str,
        gaml_experiment_parameters: Optional[list[dict[str, Any]]],
        gama_ip_address: Optional[str],
        gama_port: int,
        render_mode: Optional[str],
        initialization_wait: float
    ) -> GamaEnv:
        """Create the async environment."""
        return GamaEnv(
            gaml_experiment_path=gaml_experiment_path,
            gaml_experiment_name=gaml_experiment_name,
            gaml_experiment_parameters=gaml_experiment_parameters,
            gama_ip_address=gama_ip_address,
            gama_port=gama_port,
            render_mode=render_mode,
            initialization_wait=initialization_wait
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed (int, optional): Random seed
            options (dict, optional): Additional options
            
        Returns:
            tuple: Initial observation and info
        """
        return self._run_sync(self.env.reset, seed=seed, options=options)
    
    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        return self._run_sync(self.env.step, action)
    
    def render(self, mode: str = 'human'):
        """
        Render the environment.
        
        Args:
            mode (str): Rendering mode
        """
        return self._run_sync(self.env.render, mode)
    
    def close(self):
        """Close the environment and clean up resources."""
        if hasattr(self, 'env'):
            self._run_sync(self.env.close)
        
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def make_sync(env_class=GamaEnv):
    """
    Factory function to create a synchronous version of an environment.
    
    Args:
        env_class: Environment class to wrap
        
    Returns:
        Wrapped environment class
    """
    class SyncEnv(SyncWrapper):
        def __init__(self, *args, **kwargs):
            # Create the async env directly
            super().__init__(*args, **kwargs)
    
    return SyncEnv
