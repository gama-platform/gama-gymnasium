# GAMA-Gymnasium Source Code

This directory contains the main source code for the GAMA-Gymnasium package, which provides Gymnasium environment integration with GAMA simulations.

## 📦 Package Structure

```
src/
├── pyproject.toml          # Python package configuration
├── LICENSE                 # Package license
├── README.md              # This documentation
└── gama_gymnasium/        # Main Python module
    ├── __init__.py        # Package initialization
    ├── exceptions.py      # Custom exception classes
    ├── gama_client_wrapper.py  # GAMA communication wrapper
    ├── gama_env.py       # Main Gymnasium environment
    └── space_converter.py # Gymnasium/GAMA space conversion
```

## 🧩 Module Components

### `gama_env.py`
Main Gymnasium environment class that interfaces with GAMA simulations:
- Implements standard Gymnasium environment API
- Handles GAMA server communication
- Manages environment lifecycle (reset, step, close)

### `gama_client_wrapper.py`
Wrapper around gama-client for robust communication:
- Manages GAMA server connections
- Handles error recovery and reconnection
- Provides simplified API for environment operations

### `space_converter.py`
Converts between GAMA and Gymnasium space formats:
- Supports all standard Gymnasium spaces (Box, Discrete, MultiDiscrete, etc.)
- Handles data type conversions
- Validates space definitions

### `exceptions.py`
Custom exception hierarchy for clear error handling:
- `GamaEnvironmentError`: Base exception class
- `GamaConnectionError`: Connection-related errors
- `GamaSpaceError`: Space definition/conversion errors

## 🔗 Related Documentation

- **[Main Project README](../README.md)**: Overall project documentation
- **[Testing Guide](../tests/README.md)**: How to test the package
- **[Examples](../examples/)**: Practical usage examples
- **[Basic Example](../examples/basic_example/README.md)**: Step-by-step tutorial

## 🛠️ Development

### Package Configuration
The `pyproject.toml` file contains:
- Package metadata and dependencies
- Build system configuration
- Entry points and optional dependencies

### Installation for Development
```bash
# From the repository root
pip install -e src/
```

### Running Tests
```bash
# Run the comprehensive test suite
python tests/test_manager.py --all

# Test specific components
python tests/test_manager.py --pattern "space"
```

## 📚 API Reference

For detailed API documentation, see the docstrings in each module or generate documentation with:

```bash
python -c "import gama_gymnasium; help(gama_gymnasium)"
```
