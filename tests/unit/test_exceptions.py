"""
Unit tests for custom exceptions.

This module contains tests for the custom exception classes used
throughout the gama-gymnasium package.
"""

import pytest
import sys
import os

# Import the exception classes to test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gama_gymnasium.exceptions import (
    GamaEnvironmentError,
    GamaConnectionError,
    GamaCommandError,
    SpaceConversionError
)


class TestGamaEnvironmentError:
    """Test the base GamaEnvironmentError exception."""

    def test_basic_exception_creation(self):
        """Test creating a basic GamaEnvironmentError."""
        error = GamaEnvironmentError("Test error message")
        
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
        assert isinstance(error, GamaEnvironmentError)

    def test_exception_inheritance(self):
        """Test that GamaEnvironmentError inherits from Exception."""
        error = GamaEnvironmentError("Test")
        
        assert isinstance(error, Exception)
        assert issubclass(GamaEnvironmentError, Exception)

    def test_exception_with_empty_message(self):
        """Test creating exception with empty message."""
        error = GamaEnvironmentError("")
        
        assert str(error) == ""

    def test_exception_with_none_message(self):
        """Test creating exception with None message."""
        error = GamaEnvironmentError(None)
        
        assert str(error) == "None"

    def test_exception_raising(self):
        """Test raising and catching GamaEnvironmentError."""
        with pytest.raises(GamaEnvironmentError) as exc_info:
            raise GamaEnvironmentError("Test exception")
        
        assert str(exc_info.value) == "Test exception"
        assert exc_info.type == GamaEnvironmentError

    def test_exception_with_formatting(self):
        """Test exception with formatted message."""
        port = 9000
        host = "localhost"
        error = GamaEnvironmentError(f"Failed to connect to {host}:{port}")
        
        assert str(error) == "Failed to connect to localhost:9000"


class TestGamaConnectionError:
    """Test the GamaConnectionError exception."""

    def test_basic_connection_error(self):
        """Test creating a basic GamaConnectionError."""
        error = GamaConnectionError("Connection failed")
        
        assert str(error) == "Connection failed"
        assert isinstance(error, GamaConnectionError)
        assert isinstance(error, GamaEnvironmentError)
        assert isinstance(error, Exception)

    def test_connection_error_inheritance(self):
        """Test that GamaConnectionError inherits from GamaEnvironmentError."""
        error = GamaConnectionError("Connection error")
        
        assert isinstance(error, GamaEnvironmentError)
        assert issubclass(GamaConnectionError, GamaEnvironmentError)

    def test_connection_error_specific_message(self):
        """Test connection error with specific connection details."""
        host = "192.168.1.100"
        port = 8080
        error = GamaConnectionError(f"Could not connect to GAMA server at {host}:{port}")
        
        assert "192.168.1.100:8080" in str(error)
        assert "Could not connect to GAMA server" in str(error)

    def test_connection_error_raising(self):
        """Test raising and catching GamaConnectionError."""
        with pytest.raises(GamaConnectionError) as exc_info:
            raise GamaConnectionError("Network timeout")
        
        assert str(exc_info.value) == "Network timeout"
        assert exc_info.type == GamaConnectionError

    def test_connection_error_caught_as_base(self):
        """Test that GamaConnectionError can be caught as base exception."""
        with pytest.raises(GamaEnvironmentError):
            raise GamaConnectionError("Connection issue")

    def test_connection_error_with_cause(self):
        """Test connection error with underlying cause."""
        try:
            # Simulate an underlying connection error
            raise ConnectionRefusedError("Connection refused")
        except ConnectionRefusedError as e:
            # Wrap in our custom exception
            with pytest.raises(GamaConnectionError) as exc_info:
                raise GamaConnectionError("Failed to connect to GAMA") from e
            
            assert str(exc_info.value) == "Failed to connect to GAMA"
            assert exc_info.value.__cause__ is not None
            assert isinstance(exc_info.value.__cause__, ConnectionRefusedError)


class TestGamaCommandError:
    """Test the GamaCommandError exception."""

    def test_basic_command_error(self):
        """Test creating a basic GamaCommandError."""
        error = GamaCommandError("Command execution failed")
        
        assert str(error) == "Command execution failed"
        assert isinstance(error, GamaCommandError)
        assert isinstance(error, GamaEnvironmentError)

    def test_command_error_inheritance(self):
        """Test that GamaCommandError inherits from GamaEnvironmentError."""
        error = GamaCommandError("Command error")
        
        assert isinstance(error, GamaEnvironmentError)
        assert issubclass(GamaCommandError, GamaEnvironmentError)

    def test_command_error_with_details(self):
        """Test command error with command details."""
        command = "load_experiment"
        error_msg = "Invalid experiment file"
        error = GamaCommandError(f"Command '{command}' failed: {error_msg}")
        
        assert command in str(error)
        assert error_msg in str(error)

    def test_command_error_raising(self):
        """Test raising and catching GamaCommandError."""
        with pytest.raises(GamaCommandError) as exc_info:
            raise GamaCommandError("Invalid GAMA expression")
        
        assert str(exc_info.value) == "Invalid GAMA expression"
        assert exc_info.type == GamaCommandError

    def test_command_error_caught_as_base(self):
        """Test that GamaCommandError can be caught as base exception."""
        with pytest.raises(GamaEnvironmentError):
            raise GamaCommandError("Command failed")

    def test_command_error_with_expression_info(self):
        """Test command error with GAMA expression information."""
        expression = "GymAgent[0].state"
        error = GamaCommandError(f"Failed to execute expression '{expression}': syntax error")
        
        assert expression in str(error)
        assert "syntax error" in str(error)


class TestSpaceConversionError:
    """Test the SpaceConversionError exception."""

    def test_basic_space_conversion_error(self):
        """Test creating a basic SpaceConversionError."""
        error = SpaceConversionError("Invalid space definition")
        
        assert str(error) == "Invalid space definition"
        assert isinstance(error, SpaceConversionError)
        assert isinstance(error, GamaEnvironmentError)

    def test_space_conversion_error_inheritance(self):
        """Test that SpaceConversionError inherits from GamaEnvironmentError."""
        error = SpaceConversionError("Space error")
        
        assert isinstance(error, GamaEnvironmentError)
        assert issubclass(SpaceConversionError, GamaEnvironmentError)

    def test_space_conversion_error_with_space_type(self):
        """Test space conversion error with space type information."""
        space_type = "UnknownSpace"
        error = SpaceConversionError(f"Unknown space type: {space_type}")
        
        assert space_type in str(error)
        assert "Unknown space type" in str(error)

    def test_space_conversion_error_raising(self):
        """Test raising and catching SpaceConversionError."""
        with pytest.raises(SpaceConversionError) as exc_info:
            raise SpaceConversionError("Failed to convert space")
        
        assert str(exc_info.value) == "Failed to convert space"
        assert exc_info.type == SpaceConversionError

    def test_space_conversion_error_caught_as_base(self):
        """Test that SpaceConversionError can be caught as base exception."""
        with pytest.raises(GamaEnvironmentError):
            raise SpaceConversionError("Space conversion failed")

    def test_space_conversion_error_with_details(self):
        """Test space conversion error with detailed information."""
        space_definition = {"type": "Box", "low": "invalid", "high": 1.0}
        error = SpaceConversionError(
            f"Failed to convert Box space: invalid low value '{space_definition['low']}'"
        )
        
        assert "Box space" in str(error)
        assert "invalid low value" in str(error)
        assert "invalid" in str(error)


class TestExceptionHierarchy:
    """Test the exception hierarchy and inheritance relationships."""

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from GamaEnvironmentError."""
        exceptions = [
            GamaConnectionError("test"),
            GamaCommandError("test"),
            SpaceConversionError("test")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, GamaEnvironmentError)
            assert isinstance(exc, Exception)

    def test_exception_hierarchy_structure(self):
        """Test the complete exception hierarchy."""
        # Test inheritance chain
        assert issubclass(GamaConnectionError, GamaEnvironmentError)
        assert issubclass(GamaCommandError, GamaEnvironmentError)
        assert issubclass(SpaceConversionError, GamaEnvironmentError)
        assert issubclass(GamaEnvironmentError, Exception)
        
        # Test that specific exceptions are not subclasses of each other
        assert not issubclass(GamaConnectionError, GamaCommandError)
        assert not issubclass(GamaConnectionError, SpaceConversionError)
        assert not issubclass(GamaCommandError, SpaceConversionError)

    def test_catching_specific_vs_general(self):
        """Test catching specific exceptions vs general ones."""
        # Test that we can catch specific exceptions
        with pytest.raises(GamaConnectionError):
            raise GamaConnectionError("Connection failed")
        
        # Test that we can catch them as general exceptions
        with pytest.raises(GamaEnvironmentError):
            raise GamaConnectionError("Connection failed")
        
        with pytest.raises(Exception):
            raise GamaConnectionError("Connection failed")

    def test_multiple_exception_types(self):
        """Test handling multiple exception types."""
        def raise_different_errors(error_type):
            if error_type == "connection":
                raise GamaConnectionError("Connection error")
            elif error_type == "command":
                raise GamaCommandError("Command error")
            elif error_type == "space":
                raise SpaceConversionError("Space error")
            else:
                raise GamaEnvironmentError("General error")
        
        # Test that we can catch all with base exception
        error_types = ["connection", "command", "space", "general"]
        
        for error_type in error_types:
            with pytest.raises(GamaEnvironmentError):
                raise_different_errors(error_type)

    def test_exception_chaining(self):
        """Test exception chaining with custom exceptions."""
        try:
            # Simulate a low-level error
            raise ValueError("Invalid value")
        except ValueError as e:
            # Chain with our custom exception
            with pytest.raises(SpaceConversionError) as exc_info:
                raise SpaceConversionError("Space conversion failed") from e
            
            # Verify chaining
            assert exc_info.value.__cause__ is not None
            assert isinstance(exc_info.value.__cause__, ValueError)
            assert str(exc_info.value.__cause__) == "Invalid value"


class TestExceptionUsagePatterns:
    """Test common usage patterns for the custom exceptions."""

    def test_exception_in_context_manager(self):
        """Test using exceptions in context managers."""
        class MockResource:
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is GamaConnectionError:
                    # Handle connection errors gracefully
                    return True  # Suppress the exception
                return False
        
        # Test that connection errors are handled
        with MockResource():
            raise GamaConnectionError("Connection lost")
        
        # Test that other errors are not suppressed
        with pytest.raises(GamaCommandError):
            with MockResource():
                raise GamaCommandError("Command failed")

    def test_exception_with_logging_context(self):
        """Test exceptions with logging information."""
        import logging
        
        # Configure a simple logger for testing
        logger = logging.getLogger("test_logger")
        
        def problematic_function():
            try:
                # Simulate some operation that fails
                raise ConnectionError("Network is down")
            except ConnectionError as e:
                logger.error(f"Connection failed: {e}")
                raise GamaConnectionError("Failed to connect to GAMA server") from e
        
        with pytest.raises(GamaConnectionError) as exc_info:
            problematic_function()
        
        # Verify exception chaining
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ConnectionError)

    def test_exception_error_codes(self):
        """Test exceptions with error codes for different scenarios."""
        class GamaEnvironmentErrorWithCode(GamaEnvironmentError):
            def __init__(self, message, error_code=None):
                super().__init__(message)
                self.error_code = error_code
        
        # Test custom exception with error code
        error = GamaEnvironmentErrorWithCode("Test error", error_code=500)
        
        assert str(error) == "Test error"
        assert error.error_code == 500
        assert isinstance(error, GamaEnvironmentError)


class TestExceptionScenarios:
    """Test exception scenarios based on real usage patterns."""

    def test_connection_timeout_scenario(self):
        """Test connection timeout exception scenario."""
        with pytest.raises(GamaConnectionError) as exc_info:
            raise GamaConnectionError("Connection timeout after 30 seconds")
        
        assert "timeout" in str(exc_info.value).lower()

    def test_invalid_experiment_file_scenario(self):
        """Test invalid experiment file exception scenario."""
        file_path = "/path/to/invalid.gaml"
        with pytest.raises(GamaCommandError) as exc_info:
            raise GamaCommandError(f"Failed to load experiment from {file_path}: file not found")
        
        assert file_path in str(exc_info.value)
        assert "file not found" in str(exc_info.value)

    def test_space_definition_error_scenario(self):
        """Test space definition error exception scenario."""
        invalid_space = {"type": "Box", "low": "invalid", "high": 1.0}
        with pytest.raises(SpaceConversionError) as exc_info:
            raise SpaceConversionError(f"Invalid space definition: {invalid_space}")
        
        assert "Invalid space definition" in str(exc_info.value)

    def test_gama_expression_error_scenario(self):
        """Test GAMA expression error exception scenario."""
        expression = "invalid_agent[0].nonexistent_property"
        with pytest.raises(GamaCommandError) as exc_info:
            raise GamaCommandError(f"Expression '{expression}' failed: undefined reference")
        
        assert expression in str(exc_info.value)
        assert "undefined reference" in str(exc_info.value)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
