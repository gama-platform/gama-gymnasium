# GAMA-Gymnasium Test Suite

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Pytest](https://img.shields.io/badge/pytest-7.0+-green.svg)
![Coverage](https://img.shields.io/badge/coverage-80%25+-brightgreen.svg)

This directory contains a comprehensive test suite for the GAMA-Gymnasium project, ensuring code quality, reliability, and maintainability through automated testing.

## 📁 Project Structure

```
tests/
├── unit/                           # Unit tests for individual components
│   ├── test_exceptions.py          # Exception handling tests
│   ├── test_gama_client_wrapper.py # GAMA client communication tests
│   ├── test_gama_env.py            # Environment management tests
│   └── test_space_converter.py     # Space conversion utility tests
├── test_manager.py                 # Main test management script
├── test_runner.ps1                 # PowerShell interface (Windows)
├── test_runner.bat                 # Batch file interface (Windows)
├── Makefile                        # Make interface (Unix/Linux/WSL)
├── conftest.py                     # Pytest configuration and fixtures
├── pytest.ini                      # Pytest settings
└── README.md                       # This documentation
```

## 🚀 Quick Start

### Test Manager (Recommended)

The test manager provides the most comprehensive functionality with colored output and detailed reporting:

```bash
# Get system overview and test information
python tests/test_manager.py --info

# List all available tests
python tests/test_manager.py --list

# Quick smoke test (verify basic functionality)
python tests/test_manager.py --quick

# Run all unit tests
python tests/test_manager.py --unit

# Run all tests with detailed output
python tests/test_manager.py --all

# Generate comprehensive coverage report
python tests/test_manager.py --coverage

# Run specific test file
python tests/test_manager.py --file test_exceptions.py

# Run tests matching a pattern
python tests/test_manager.py --pattern "space"
```

### Alternative Interfaces

#### PowerShell (Windows)

```powershell
# Run unit tests with colored output
.\tests\test_runner.ps1 -Action unit

# Generate coverage report
.\tests\test_runner.ps1 -Action coverage

# Quick functionality check
.\tests\test_runner.ps1 -Action quick

# Show help
.\tests\test_runner.ps1 -Action help
```

#### Batch File (Windows)

```cmd
# Run all tests
tests\test_runner.bat all

# Run unit tests only
tests\test_runner.bat unit

# Generate coverage report
tests\test_runner.bat coverage

# Quick check
tests\test_runner.bat quick
```

#### Make (Unix/Linux/WSL)

```bash
# Run tests
make test

# Coverage report with HTML output
make coverage

# Lint code
make lint

# Format code
make format

# Clean build artifacts
make clean
```

### Direct Pytest Usage

For advanced users who need fine-grained control:

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run unit tests only
python -m pytest tests/unit/ -v

# Run with coverage reporting
python -m pytest tests/unit/ --cov=src --cov-report=html -v

# Run specific test function
python -m pytest tests/unit/test_exceptions.py::TestGamaEnvironmentError::test_base_exception -v

# Run tests matching pattern
python -m pytest tests/ -k "space" -v

# Stop on first failure
python -m pytest tests/ -x -v

# Show the slowest 10 tests
python -m pytest tests/ --durations=10
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)

Unit tests focus on testing individual components in isolation using mocking to avoid dependencies on external services:

- **`test_exceptions.py`**: Tests custom exception hierarchy and error handling
- **`test_space_converter.py`**: Tests space conversion between GAMA and Gymnasium formats  
- **`test_gama_client_wrapper.py`**: Tests GAMA server communication wrapper
- **`test_gama_env.py`**: Tests main Gymnasium environment implementation

### Test Features

- **🎭 Mocking**: Uses `unittest.mock` for component isolation
- **🔧 Fixtures**: Pytest fixtures for test setup and teardown
- **📊 Parametrization**: Multiple test scenarios per function
- **📈 Coverage**: Comprehensive code coverage reporting
- **⚡ Performance**: Fast execution with parallel test support
- **🌐 Cross-platform**: Works on Windows, macOS, and Linux

## 📊 Coverage Reporting

Coverage reports are generated in multiple formats for different use cases:

- **📟 Terminal**: Real-time summary statistics during test runs
- **🌐 HTML**: Interactive web-based report (`htmlcov/index.html`)
- **📄 XML**: Machine-readable format for CI/CD integration (`coverage.xml`)

```bash
# Generate all coverage report formats
python tests/test_manager.py --coverage

# Open HTML report (cross-platform)
# Windows
start htmlcov/index.html
# macOS  
open htmlcov/index.html
# Linux
xdg-open htmlcov/index.html
```

## ⚙️ Configuration

### Pytest Configuration (`pytest.ini`)

```ini
[pytest]
# Test discovery settings
testpaths = tests/unit
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Output and reporting options
addopts = -v --tb=short --strict-markers --strict-config --color=yes --durations=10

# Test categorization markers
markers =
    unit: Unit tests that test individual components in isolation
    slow: Tests that take a long time to run
    gama: Tests that require GAMA simulation platform
    asyncio: Async tests that use asyncio

# Minimum version requirements
minversion = 7.0

# Asyncio configuration
asyncio_mode = auto
```

### Test Fixtures (`conftest.py`)

Common test fixtures and utilities include:

- **🗂️ Project paths**: Access to project directories
- **📋 Sample data**: Space definitions, GAMA responses, test actions
- **🎭 Mock objects**: Pre-configured mocks for GAMA client and space converter
- **🔧 Utility functions**: Test helpers and assertion utilities
- **⏱️ Performance timers**: For performance and benchmark testing

## 📦 Dependencies

### Required Packages

```bash
# Core testing framework
pytest>=7.0.0              # Main testing framework
pytest-cov>=4.0.0          # Coverage reporting plugin
pytest-mock>=3.10.0        # Enhanced mocking capabilities
pytest-asyncio>=0.21.0     # Async test support

# Project dependencies  
numpy>=1.21.0              # Numerical computations
gymnasium>=0.29.0          # RL environment framework

# Optional performance testing
pytest-xdist>=3.0.0        # Parallel test execution
pytest-benchmark>=4.0.0    # Performance benchmarking
```

### Installation

```bash
# Install using the test manager (recommended)
python tests/test_manager.py --install-deps

# Install manually with pip
pip install pytest pytest-cov pytest-mock pytest-asyncio numpy gymnasium

# Install from requirements file (if available)
pip install -r tests/requirements-test.txt
```

## 🎯 Writing New Tests

### Test Structure Template

```python
"""Tests for new module functionality."""

import pytest
from unittest.mock import MagicMock, patch, Mock
import numpy as np
from pathlib import Path
import sys

# Add source path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from your_module import YourClass


class TestYourClass:
    """Test suite for YourClass functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.instance = YourClass()
        self.mock_dependency = MagicMock()
    
    def test_basic_functionality(self):
        """Test basic functionality works as expected."""
        result = self.instance.some_method()
        assert result is not None
        assert isinstance(result, expected_type)
    
    @patch('your_module.external_dependency')
    def test_with_external_mocking(self, mock_dependency):
        """Test behavior with external dependencies mocked."""
        # Arrange
        mock_dependency.return_value = "expected_value"
        
        # Act
        result = self.instance.method_using_dependency()
        
        # Assert
        assert result == "expected_value"
        mock_dependency.assert_called_once()
    
    @pytest.mark.parametrize("input_value,expected_output", [
        (1, 2),
        (2, 4), 
        (3, 6),
        (0, 0),
        (-1, -2),
    ])
    def test_parametrized_scenarios(self, input_value, expected_output):
        """Test multiple scenarios using parametrization."""
        result = self.instance.double(input_value)
        assert result == expected_output
    
    @pytest.mark.unit
    def test_error_handling(self):
        """Test proper error handling and exceptions."""
        with pytest.raises(ValueError, match="Invalid input"):
            self.instance.method_that_should_fail("invalid_input")
    
    @pytest.mark.slow
    def test_performance_critical_operation(self):
        """Test performance-critical operations."""
        import time
        start_time = time.time()
        result = self.instance.slow_operation()
        execution_time = time.time() - start_time
        
        assert result is not None
        assert execution_time < 5.0  # Should complete within 5 seconds
```

### Test Markers

Categorize tests using pytest markers:

```python
@pytest.mark.unit
def test_isolated_component():
    """Fast unit test that runs in isolation."""
    pass

@pytest.mark.slow  
def test_long_running_operation():
    """Test that takes significant time to complete."""
    pass

@pytest.mark.gama
def test_requiring_gama_server():
    """Test that requires GAMA simulation platform."""
    pass

@pytest.mark.asyncio
async def test_async_functionality():
    """Test async/await functionality."""
    result = await async_function()
    assert result is not None
```

## 🔧 Best Practices

### Test Quality Guidelines

- **⚡ Fast**: Unit tests should run in < 1 second each
- **🔒 Isolated**: No dependencies on external services or file system
- **🎯 Deterministic**: Same input always produces same output
- **📝 Clear**: Test names clearly describe what is being tested
- **🎭 Mocked**: External dependencies are properly mocked

### Development Workflow

1. **🔍 Quick Check**: Run `--quick` first to verify basic functionality
2. **🧪 Unit Tests**: Use `--unit` for focused development testing  
3. **📊 Coverage**: Run `--coverage` before committing changes
4. **🔍 Pattern Testing**: Use `--pattern` to test specific functionality
5. **🚀 Full Suite**: Run `--all` before major releases

### Code Coverage Standards

- **🎯 Target**: Aim for 80%+ code coverage on unit tests
- **✅ New Code**: All new code should include corresponding tests
- **🚨 Critical Paths**: Critical functionality should have 95%+ coverage
- **📈 Trending**: Coverage should not decrease over time

## 🔍 Debugging and Troubleshooting

### Debug Mode Options

```bash
# Verbose output with detailed test information
python tests/test_manager.py --unit --verbose

# Stop on first failure for immediate debugging
python tests/test_manager.py --unit --fail-fast

# Don't capture output (shows print statements)
python -m pytest tests/unit/test_exceptions.py -s

# Show local variables in tracebacks
python -m pytest tests/unit/test_exceptions.py --tb=long

# Drop into debugger on failures
python -m pytest tests/unit/test_exceptions.py --pdb
```

### Common Issues and Solutions

#### Import Errors
```bash
# Verify project structure and Python path
python tests/test_manager.py --info

# Check current working directory
pwd  # Should be in project root
cd /path/to/gama-gymnasium
```

#### Missing Dependencies
```bash
# Install all required test dependencies
python tests/test_manager.py --install-deps

# Verify installation
python -c "import pytest, numpy, gymnasium; print('All dependencies available')"
```

#### Unicode Errors (Windows)
- Test manager uses ASCII-compatible output for Windows
- Ensure terminal supports UTF-8 encoding
- Use Windows Terminal or PowerShell 7+ for best experience

#### Path Issues
- Always run tests from project root directory
- Use forward slashes in paths for cross-platform compatibility
- Verify `src/` directory structure is correct

## 🚀 Continuous Integration

### GitHub Actions Example

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python tests/test_manager.py --install-deps
    
    - name: Run quick check
      run: python tests/test_manager.py --quick
    
    - name: Run test suite with coverage
      run: python tests/test_manager.py --coverage
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

### Pre-commit Hooks

```bash
# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "Running pre-commit tests..."
python tests/test_manager.py --quick
if [ $? -ne 0 ]; then
    echo "❌ Tests failed! Commit aborted."
    exit 1
fi
echo "✅ All tests passed!"
EOF

chmod +x .git/hooks/pre-commit
```

## 📈 Performance Optimization

### Test Execution Times

| Test Type | Expected Duration | Optimization Tips |
|-----------|------------------|-------------------|
| Quick check | 3-5 seconds | Verify basic imports and simple functionality |
| Unit tests | 10-30 seconds | Run during development for rapid feedback |
| Full suite | 30-60 seconds | Run before commits and releases |
| Coverage report | +10-20 seconds | Generate before major releases |

### Parallel Execution

```bash
# Install pytest-xdist for parallel execution
pip install pytest-xdist

# Run tests in parallel (auto-detect CPU cores)
python -m pytest tests/ -n auto

# Run tests with specific number of workers
python -m pytest tests/ -n 4
```

## 🆘 Getting Help

### Quick Commands
```bash
# View all available commands and options
python tests/test_manager.py --help

# Check test environment status
python tests/test_manager.py --info

# List all discoverable tests
python tests/test_manager.py --list

# Run basic functionality verification
python tests/test_manager.py --quick
```

### Additional Resources

- 📚 **Pytest Documentation**: [https://docs.pytest.org/](https://docs.pytest.org/)
- 🐍 **Python Testing Guide**: [https://docs.python.org/3/library/unittest.html](https://docs.python.org/3/library/unittest.html)
- 🏃 **Gymnasium Documentation**: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)
- 🎮 **GAMA Platform**: [https://gama-platform.org/](https://gama-platform.org/)

---

## 📝 Contributing

When contributing to this project:

1. **📝 Write Tests**: Include tests for all new functionality
2. **🧪 Run Suite**: Ensure all tests pass before submitting PRs
3. **📊 Maintain Coverage**: Don't let coverage decrease
4. **📖 Update Docs**: Update this README for new test patterns
5. **🔍 Review**: Code review should include test review

## 🔗 Project Navigation

- **[🏠 Main Project](../README.md)**: Overall GAMA-Gymnasium documentation
- **[📦 Source Code](../src/README.md)**: Package structure and development guide
- **[🎯 Basic Example](../examples/basic_example/README.md)**: Start here for practical usage
- **[🧠 CartPole DQN](../examples/cartpole%20DQN/README.md)**: Advanced reinforcement learning example
- **[🔧 Direct GAMA Test](../examples/basic_example/README_basic_test.md)**: Low-level API demonstration

---

**Last Updated**: July 2025  
**Python Versions**: 3.8+  
**Pytest Version**: 7.0+  
**Maintained by**: GAMA-Gymnasium Team

This comprehensive test suite ensures the reliability and quality of the GAMA-Gymnasium project. When adding new features, always include corresponding tests to maintain code quality and prevent regressions.
