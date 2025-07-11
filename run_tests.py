#!/usr/bin/env python3
"""
Test runner script for GAMA-Gymnasium.

This script provides convenient commands to run different types of tests.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🏃 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def run_unit_tests():
    """Run unit tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/",
        "-v",
        "--cov=gama_gymnasium",
        "--cov-report=term-missing"
    ]
    return run_command(cmd, "Running unit tests")


def run_integration_tests():
    """Run integration tests."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/integration/",
        "-v",
        "-m", "integration"
    ]
    return run_command(cmd, "Running integration tests")


def run_performance_tests():
    """Run performance tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/performance/",
        "-v", 
        "-m", "performance"
    ]
    return run_command(cmd, "Running performance tests")


def run_all_tests():
    """Run all tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--cov=gama_gymnasium",
        "--cov-report=term-missing",
        "--cov-report=html"
    ]
    return run_command(cmd, "Running all tests")


def run_fast_tests():
    """Run fast tests only (exclude slow tests)."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "-m", "not slow and not performance",
        "--cov=gama_gymnasium",
        "--cov-report=term-missing"
    ]
    return run_command(cmd, "Running fast tests")


def run_lint():
    """Run linting tools."""
    tools = [
        ([sys.executable, "-m", "black", "--check", "src/", "tests/"], "Black formatting check"),
        ([sys.executable, "-m", "isort", "--check-only", "src/", "tests/"], "Import sorting check"),
        ([sys.executable, "-m", "flake8", "src/", "tests/"], "Flake8 linting"),
        ([sys.executable, "-m", "mypy", "src/gama_gymnasium/"], "MyPy type checking"),
    ]
    
    all_passed = True
    for cmd, description in tools:
        if not run_command(cmd, description):
            all_passed = False
    
    return all_passed


def run_format():
    """Run code formatting."""
    tools = [
        ([sys.executable, "-m", "black", "src/", "tests/"], "Black formatting"),
        ([sys.executable, "-m", "isort", "src/", "tests/"], "Import sorting"),
    ]
    
    all_passed = True
    for cmd, description in tools:
        if not run_command(cmd, description):
            all_passed = False
    
    return all_passed


def run_benchmark():
    """Run benchmark tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/performance/",
        "-v",
        "-m", "benchmark",
        "--benchmark-only",
        "--benchmark-sort=min",
        "--benchmark-columns=min,max,mean,stddev",
        "--benchmark-histogram"
    ]
    return run_command(cmd, "Running benchmark tests")


def generate_coverage_report():
    """Generate detailed coverage report."""
    # First run tests with coverage
    cmd1 = [
        sys.executable, "-m", "pytest",
        "tests/unit/",
        "--cov=gama_gymnasium",
        "--cov-report=html",
        "--cov-report=xml",
        "--cov-report=term-missing"
    ]
    
    if run_command(cmd1, "Generating coverage report"):
        print("\n📊 Coverage report generated:")
        print("  - HTML report: htmlcov/index.html")
        print("  - XML report: coverage.xml")
        return True
    
    return False


def check_dependencies():
    """Check if all dependencies are installed."""
    dependencies = [
        "pytest",
        "pytest-cov", 
        "pytest-asyncio",
        "pytest-benchmark",
        "black",
        "isort",
        "flake8",
        "mypy",
        "gymnasium",
        "numpy"
    ]
    
    print("🔍 Checking dependencies...")
    missing = []
    
    for dep in dependencies:
        try:
            subprocess.run([sys.executable, "-c", f"import {dep.replace('-', '_')}"], 
                         check=True, capture_output=True)
            print(f"  ✅ {dep}")
        except subprocess.CalledProcessError:
            print(f"  ❌ {dep}")
            missing.append(dep)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Install them with: pip install " + " ".join(missing))
        return False
    
    print("\n✅ All dependencies are installed!")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test runner for GAMA-Gymnasium")
    
    parser.add_argument("command", choices=[
        "unit", "integration", "performance", "all", "fast",
        "lint", "format", "benchmark", "coverage", "deps"
    ], help="Test command to run")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent
    import os
    os.chdir(project_root)
    
    print(f"🧪 GAMA-Gymnasium Test Runner")
    print(f"Project root: {project_root}")
    
    # Map commands to functions
    commands = {
        "unit": run_unit_tests,
        "integration": run_integration_tests, 
        "performance": run_performance_tests,
        "all": run_all_tests,
        "fast": run_fast_tests,
        "lint": run_lint,
        "format": run_format,
        "benchmark": run_benchmark,
        "coverage": generate_coverage_report,
        "deps": check_dependencies
    }
    
    # Run the command
    success = commands[args.command]()
    
    if success:
        print(f"\n✅ {args.command} completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ {args.command} failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
