# Makefile for GAMA-Gymnasium
# Compatible with Windows (using Python scripts)

# Variables
PYTHON = python
PIP = pip
SRC_DIR = src
TEST_DIR = tests
COVERAGE_DIR = htmlcov

# Help target
.PHONY: help
help:
	@echo "GAMA-Gymnasium Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install-dev    Install development dependencies"
	@echo "  install        Install package in development mode"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-performance  Run performance tests only"
	@echo "  test-fast      Run fast tests (exclude slow tests)"
	@echo "  coverage       Generate coverage report"
	@echo "  benchmark      Run benchmark tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           Run all linting tools"
	@echo "  format         Format code with black and isort"
	@echo "  check-format   Check code formatting"
	@echo "  type-check     Run mypy type checking"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean          Remove build artifacts and cache files"
	@echo "  clean-coverage Remove coverage reports"
	@echo ""
	@echo "Build:"
	@echo "  build          Build package"
	@echo "  docs           Build documentation"

# Installation targets
.PHONY: install-dev
install-dev:
	$(PIP) install -e ".[dev,docs,examples]"

.PHONY: install
install:
	$(PIP) install -e .

# Testing targets
.PHONY: test
test:
	$(PYTHON) run_tests.py all

.PHONY: test-unit
test-unit:
	$(PYTHON) run_tests.py unit

.PHONY: test-integration
test-integration:
	$(PYTHON) run_tests.py integration

.PHONY: test-performance
test-performance:
	$(PYTHON) run_tests.py performance

.PHONY: test-fast
test-fast:
	$(PYTHON) run_tests.py fast

.PHONY: coverage
coverage:
	$(PYTHON) run_tests.py coverage

.PHONY: benchmark
benchmark:
	$(PYTHON) run_tests.py benchmark

# Code quality targets
.PHONY: lint
lint:
	$(PYTHON) run_tests.py lint

.PHONY: format
format:
	$(PYTHON) run_tests.py format

.PHONY: check-format
check-format:
	$(PYTHON) -m black --check $(SRC_DIR) $(TEST_DIR)
	$(PYTHON) -m isort --check-only $(SRC_DIR) $(TEST_DIR)

.PHONY: type-check
type-check:
	$(PYTHON) -m mypy $(SRC_DIR)/gama_gymnasium

# Dependencies check
.PHONY: deps
deps:
	$(PYTHON) run_tests.py deps

# Cleanup targets
.PHONY: clean
clean:
	@echo "Cleaning build artifacts and cache files..."
	@if exist build rmdir /s /q build
	@if exist dist rmdir /s /q dist
	@if exist *.egg-info rmdir /s /q *.egg-info
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@for /r . %%f in (*.pyc *.pyo) do @if exist "%%f" del /q "%%f"
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist .coverage del /q .coverage
	@if exist .mypy_cache rmdir /s /q .mypy_cache
	@echo "Clean completed!"

.PHONY: clean-coverage
clean-coverage:
	@echo "Cleaning coverage reports..."
	@if exist $(COVERAGE_DIR) rmdir /s /q $(COVERAGE_DIR)
	@if exist coverage.xml del /q coverage.xml
	@if exist .coverage del /q .coverage
	@echo "Coverage clean completed!"

# Build targets
.PHONY: build
build: clean
	$(PYTHON) -m build

.PHONY: docs
docs:
	@echo "Building documentation..."
	@if not exist docs mkdir docs
	$(PYTHON) -m mkdocs build

# Development workflow targets
.PHONY: dev-setup
dev-setup: install-dev deps
	@echo "Development environment setup complete!"

.PHONY: pre-commit
pre-commit: format lint test-fast
	@echo "Pre-commit checks passed!"

.PHONY: ci
ci: deps lint test coverage
	@echo "CI pipeline completed!"

# Package targets
.PHONY: check-package
check-package: build
	$(PYTHON) -m twine check dist/*

.PHONY: upload-test
upload-test: check-package
	$(PYTHON) -m twine upload --repository testpypi dist/*

.PHONY: upload
upload: check-package
	$(PYTHON) -m twine upload dist/*

# Utility targets
.PHONY: requirements
requirements:
	$(PIP) freeze > requirements-dev.txt

.PHONY: security
security:
	$(PYTHON) -m safety check
	$(PYTHON) -m bandit -r $(SRC_DIR)

.PHONY: outdated
outdated:
	$(PIP) list --outdated

# Default target
.DEFAULT_GOAL := help
