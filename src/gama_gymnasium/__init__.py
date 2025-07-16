"""
GAMA-Gymnasium: Gymnasium environments for GAMA agent-based simulations.
"""

from gymnasium.envs.registration import register

from .gama_env import GamaEnv
from .exceptions import (
    GamaEnvironmentError,
    GamaConnectionError, 
    GamaCommandError,
    SpaceConversionError
)

# Register the environment with Gymnasium
register(
    id='gama_gymnasium_env/GamaEnv-v0',
    entry_point='gama_gymnasium.gama_env:GamaEnv',
)

__version__ = "0.1.0"
__all__ = [
    "GamaEnv",
    "GamaEnvironmentError",
    "GamaConnectionError",
    "GamaCommandError", 
    "SpaceConversionError",
]