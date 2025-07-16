# Makefile for GAMA-Gymnasium tests
# Usage: make [target]

.PHONY: help test unit all quick list info install-deps coverage clean

# Default target
help:
	@echo "🧪 GAMA-Gymnasium Test Makefile"
	@echo "==============================="
	@echo ""
	@echo "Available targets:"
	@echo "  help        - Show this help message"
	@echo "  test        - Run unit tests (default)"
	@echo "  unit        - Run unit tests"
	@echo "  all         - Run all tests"
	@echo "  quick       - Run quick smoke test"
	@echo "  list        - List available tests"
	@echo "  info        - Show test environment info"
	@echo "  install-deps - Install test dependencies"
	@echo "  coverage    - Run tests with coverage report"
	@echo "  clean       - Clean test artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make test"
	@echo "  make coverage"
	@echo "  make quick"

# Default test target (unit tests)
test: unit

# Run unit tests
unit:
	@echo "🔬 Running unit tests..."
	@python tests/test_manager.py --unit

# Run all tests
all:
	@echo "🚀 Running all tests..."
	@python tests/test_manager.py --all

# Quick smoke test
quick:
	@echo "⚡ Running quick smoke test..."
	@python tests/test_manager.py --quick

# List available tests
list:
	@echo "📋 Listing available tests..."
	@python tests/test_manager.py --list

# Show test environment info
info:
	@echo "ℹ️ Showing test environment info..."
	@python tests/test_manager.py --info

# Install test dependencies
install-deps:
	@echo "📦 Installing test dependencies..."
	@python tests/test_manager.py --install-deps

# Run tests with coverage
coverage:
	@echo "📊 Running tests with coverage..."
	@python tests/test_manager.py --coverage

# Clean test artifacts
clean:
	@echo "🧹 Cleaning test artifacts..."
	@rm -rf htmlcov/
	@rm -rf .coverage
	@rm -rf coverage.xml
	@rm -rf .pytest_cache/
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Test artifacts cleaned"

# Advanced targets for CI/CD
ci-test:
	@echo "🔄 Running CI tests..."
	@python tests/test_manager.py --all --coverage

# Development targets
dev-install: install-deps
	@echo "🛠 Installing development dependencies..."
	@pip install black isort mypy flake8

format:
	@echo "🎨 Formatting code..."
	@black src/ tests/ --line-length 88
	@isort src/ tests/ --profile black

lint:
	@echo "🔍 Linting code..."
	@flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	@mypy src/ --ignore-missing-imports

# Help for specific test files
test-space:
	@python tests/test_manager.py --file test_space_converter.py

test-env:
	@python tests/test_manager.py --file test_gama_env.py

test-client:
	@python tests/test_manager.py --file test_gama_client_wrapper.py

test-exceptions:
	@python tests/test_manager.py --file test_exceptions.py
