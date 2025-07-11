"""
Custom Exceptions for GAMA-Gymnasium Integration

This module defines custom exception classes used throughout the package.
"""


class GamaGymnasiumError(Exception):
    """Base exception for all GAMA-Gymnasium errors."""
    pass


class ConnectionError(GamaGymnasiumError):
    """Raised when there are issues connecting to GAMA server."""
    pass


class MessageValidationError(GamaGymnasiumError):
    """Raised when message validation fails."""
    pass


class SpaceConversionError(GamaGymnasiumError):
    """Raised when space conversion between GAMA and Gymnasium fails."""
    pass


class ExperimentError(GamaGymnasiumError):
    """Raised when there are issues with GAMA experiment execution."""
    pass


class ConfigurationError(GamaGymnasiumError):
    """Raised when there are configuration issues."""
    pass
