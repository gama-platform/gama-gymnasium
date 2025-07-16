"""
Custom exceptions for GAMA-Gymnasium integration.
"""


class GamaEnvironmentError(Exception):
    """Base exception for GAMA environment errors."""
    pass


class GamaConnectionError(GamaEnvironmentError):
    """Raised when connection to GAMA server fails."""
    pass


class GamaCommandError(GamaEnvironmentError):
    """Raised when a GAMA command fails to execute."""
    pass


class SpaceConversionError(GamaEnvironmentError):
    """Raised when space conversion between GAMA and Gymnasium fails."""
    pass