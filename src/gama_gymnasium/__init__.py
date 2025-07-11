"""
GAMA-Gymnasium Integration Package

A seamless integration between GAMA Platform simulations and OpenAI Gymnasium
for reinforcement learning research.
"""

from .core.gama_env import GamaEnv
from .core.client import GamaClient
from .spaces.converters import map_to_space
from .wrappers.sync import SyncWrapper
from .wrappers.monitoring import MonitoringWrapper

__version__ = "0.1.0"
__author__ = "GAMA Platform Team"
__email__ = "contact@gama-platform.org"

__all__ = [
    "GamaEnv",
    "GamaClient", 
    "SyncWrapper",
    "MonitoringWrapper",
    "map_to_space"
]

__version__ = "0.1.0"
__author__ = "GAMA Platform Team"
__email__ = "contact@gama-platform.org"

__all__ = [
    "GamaEnv",
    "GamaConnectionError", 
    "GamaExperimentError",
    "converters"
]
