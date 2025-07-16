#!/usr/bin/env python3
"""
Test Manager for GAMA-Gymnasium

This script provides a comprehensive test management system for the GAMA-Gymnasium project.
It allows you to run different types of tests, generate reports, and manage test execution.

Usage:
    python test_manager.py [options]
    
Examples:
    python test_manager.py --all                    # Run all tests
    python test_manager.py --unit                   # Run unit tests only
    python test_manager.py --coverage               # Run with coverage report
    python test_manager.py --file test_space_converter.py  # Run specific file
    python test_manager.py --pattern "space"        # Run tests matching pattern
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


class TestManager:
    """Main test management class."""
    
    def __init__(self):
        # Get the absolute path to the project root from the script location
        self.script_path = Path(__file__).resolve()
        self.tests_dir = self.script_path.parent
        self.project_root = self.tests_dir.parent
        self.src_dir = self.project_root / "src"
        
        # Verify we're in the right directory by checking for pytest.ini
        pytest_ini = self.project_root / "pytest.ini"
        if not pytest_ini.exists():
            raise RuntimeError(f"pytest.ini not found at {pytest_ini}. Please run from project root or check project structure.")
    
    def print_colored(self, message: str, color: str = Colors.WHITE, bold: bool = False) -> None:
        """Print colored message to console."""
        prefix = Colors.BOLD if bold else ""
        print(f"{prefix}{color}{message}{Colors.END}")
    
    def print_header(self, title: str) -> None:
        """Print a formatted header."""
        separator = "=" * 60
        self.print_colored(separator, Colors.GREEN)
        self.print_colored(f"🧪 {title}", Colors.GREEN, bold=True)
        self.print_colored(separator, Colors.GREEN)
    
    def print_section(self, title: str) -> None:
        """Print a formatted section header."""
        self.print_colored(f"\n📋 {title}", Colors.CYAN, bold=True)
        self.print_colored("-" * 40, Colors.CYAN)
    
    def run_command(self, cmd: List[str], description: str) -> Tuple[bool, str, str]:
        """
        Run a command and return success status and output.
        
        Args:
            cmd: Command to run as list of strings
            description: Description of what the command does
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        self.print_colored(f"\n🚀 {description}", Colors.YELLOW)
        self.print_colored(f"Command: {' '.join(cmd)}", Colors.WHITE)
        print()
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=self.project_root
            )
            
            elapsed = time.time() - start_time
            success = result.returncode == 0
            
            # Print output
            if result.stdout:
                print(result.stdout)
            
            if result.stderr:
                self.print_colored("STDERR:", Colors.YELLOW)
                print(result.stderr)
            
            # Print result
            status = "✅ PASSED" if success else "❌ FAILED"
            color = Colors.GREEN if success else Colors.RED
            self.print_colored(f"\n{status} - {description} (took {elapsed:.2f}s)", color, bold=True)
            
            return success, result.stdout, result.stderr
            
        except Exception as e:
            self.print_colored(f"❌ ERROR: {str(e)}", Colors.RED, bold=True)
            return False, "", str(e)
    
    def get_python_command(self) -> str:
        """Get the appropriate Python command to use."""
        # Check if we're in a conda environment
        if os.environ.get('CONDA_DEFAULT_ENV'):
            conda_prefix = os.environ.get('CONDA_PREFIX', '')
            if conda_prefix and 'gama-gymnasium' in conda_prefix:
                return "python"
        
        # Try different Python commands
        for python_cmd in ["python", "python3", "py"]:
            try:
                result = subprocess.run([python_cmd, "--version"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return python_cmd
            except FileNotFoundError:
                continue
        
        raise RuntimeError("Python interpreter not found")
    
    def install_dependencies(self) -> bool:
        """Install test dependencies."""
        self.print_section("Installing Test Dependencies")
        
        python_cmd = self.get_python_command()
        
        dependencies = [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-mock>=3.10",
            "numpy>=1.21",
            "gymnasium>=0.29"
        ]
        
        for dep in dependencies:
            self.print_colored(f"Installing {dep}...", Colors.WHITE)
            success, _, _ = self.run_command(
                [python_cmd, "-m", "pip", "install", dep],
                f"Install {dep}"
            )
            if not success:
                self.print_colored(f"Failed to install {dep}", Colors.RED)
                return False
        
        self.print_colored("✅ All dependencies installed successfully", Colors.GREEN)
        return True
    
    def discover_tests(self) -> Dict[str, List[str]]:
        """Discover available test files."""
        test_files = {
            "unit": [],
            "custom": [],
            "all": []
        }
        
        # Unit tests
        unit_dir = self.tests_dir / "unit"
        if unit_dir.exists():
            for test_file in unit_dir.glob("test_*.py"):
                test_files["unit"].append(str(test_file.relative_to(self.project_root)))
                test_files["all"].append(str(test_file.relative_to(self.project_root)))
        
        # Custom tests
        custom_dir = self.tests_dir / "tests_custom"
        if custom_dir.exists():
            for test_file in custom_dir.glob("test_*.py"):
                test_files["custom"].append(str(test_file.relative_to(self.project_root)))
                test_files["all"].append(str(test_file.relative_to(self.project_root)))
        
        # Root level tests
        for test_file in self.tests_dir.glob("test_*.py"):
            test_files["all"].append(str(test_file.relative_to(self.project_root)))
        
        return test_files
    
    def run_unit_tests(self, verbose: bool = False, fail_fast: bool = False) -> bool:
        """Run unit tests."""
        python_cmd = self.get_python_command()
        
        cmd = [python_cmd, "-m", "pytest", "tests/unit/"]
        
        if verbose:
            cmd.append("-vv")
        else:
            cmd.append("-v")
        
        if fail_fast:
            cmd.append("-x")
        
        cmd.extend(["--tb=short", "-m", "unit"])
        
        success, _, _ = self.run_command(cmd, "Unit Tests")
        return success
    
    def run_all_tests(self, verbose: bool = False, fail_fast: bool = False) -> bool:
        """Run all available tests."""
        python_cmd = self.get_python_command()
        
        cmd = [python_cmd, "-m", "pytest", "tests/"]
        
        if verbose:
            cmd.append("-vv")
        else:
            cmd.append("-v")
        
        if fail_fast:
            cmd.append("-x")
        
        cmd.append("--tb=short")
        
        success, _, _ = self.run_command(cmd, "All Tests")
        return success
    
    def run_specific_file(self, test_file: str, verbose: bool = False) -> bool:
        """Run tests from a specific file."""
        python_cmd = self.get_python_command()
        
        # Construct full path
        if not test_file.startswith("tests/"):
            test_file = f"tests/{test_file}"
        
        if not test_file.endswith(".py"):
            test_file = f"{test_file}.py"
        
        test_path = self.project_root / test_file
        if not test_path.exists():
            self.print_colored(f"❌ Test file not found: {test_file}", Colors.RED)
            return False
        
        cmd = [python_cmd, "-m", "pytest", test_file]
        
        if verbose:
            cmd.append("-vv")
        else:
            cmd.append("-v")
        
        cmd.append("--tb=short")
        
        success, _, _ = self.run_command(cmd, f"Specific Test File: {test_file}")
        return success
    
    def run_pattern_tests(self, pattern: str, verbose: bool = False) -> bool:
        """Run tests matching a pattern."""
        python_cmd = self.get_python_command()
        
        cmd = [python_cmd, "-m", "pytest", "tests/", "-k", pattern]
        
        if verbose:
            cmd.append("-vv")
        else:
            cmd.append("-v")
        
        cmd.append("--tb=short")
        
        success, _, _ = self.run_command(cmd, f"Tests matching pattern: {pattern}")
        return success
    
    def run_with_coverage(self, test_scope: str = "unit", verbose: bool = False) -> bool:
        """Run tests with coverage reporting."""
        python_cmd = self.get_python_command()
        
        if test_scope == "unit":
            test_path = "tests/unit/"
        else:
            test_path = "tests/"
        
        cmd = [
            python_cmd, "-m", "pytest", test_path,
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-fail-under=80"
        ]
        
        if verbose:
            cmd.append("-vv")
        else:
            cmd.append("-v")
        
        cmd.append("--tb=short")
        
        success, _, _ = self.run_command(cmd, f"Coverage Tests ({test_scope})")
        
        if success:
            self.print_colored("\n📊 Coverage reports generated:", Colors.GREEN)
            self.print_colored("  HTML: htmlcov/index.html", Colors.WHITE)
            self.print_colored("  XML: coverage.xml", Colors.WHITE)
        
        return success
    
    def run_quick_check(self) -> bool:
        """Run a quick smoke test to verify basic functionality."""
        python_cmd = self.get_python_command()
        
        # Test import of main modules
        modules_to_test = [
            "gama_gymnasium.space_converter",
            "gama_gymnasium.exceptions", 
            "gama_gymnasium.gama_client_wrapper",
            "gama_gymnasium.gama_env"
        ]
        
        # Add the src directory to Python path (use forward slashes for cross-platform)
        src_path = str(self.src_dir).replace('\\', '/')
        
        for module in modules_to_test:
            import_cmd = f"import sys; sys.path.insert(0, r'{self.src_dir}'); import {module}; print('OK {module}')"
            cmd = [python_cmd, "-c", import_cmd]
            success, stdout, stderr = self.run_command(cmd, f"Import {module}")
            if not success:
                return False
        
        # Run a quick test if imports succeed
        cmd = [python_cmd, "-m", "pytest", "tests/unit/test_exceptions.py::TestGamaEnvironmentError::test_basic_exception", "-q"]
        success, _, _ = self.run_command(cmd, "Quick Exception Test")
        
        return success
    
    def list_available_tests(self) -> None:
        """List all available tests."""
        self.print_section("Available Tests")
        
        test_files = self.discover_tests()
        
        for category, files in test_files.items():
            if files:
                self.print_colored(f"\n{category.upper()} Tests:", Colors.CYAN)
                for file in sorted(files):
                    self.print_colored(f"  • {file}", Colors.WHITE)
    
    def show_test_info(self) -> None:
        """Show information about the test setup."""
        self.print_section("Test Environment Information")
        
        python_cmd = self.get_python_command()
        
        # Python version
        result = subprocess.run([python_cmd, "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            self.print_colored(f"Python: {result.stdout.strip()}", Colors.WHITE)
        
        # Pytest version
        result = subprocess.run([python_cmd, "-m", "pytest", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pytest_info = result.stdout.strip().split('\n')[0]
            self.print_colored(f"Pytest: {pytest_info}", Colors.WHITE)
        
        # Test files count
        test_files = self.discover_tests()
        total_tests = len(test_files["all"])
        unit_tests = len(test_files["unit"])
        custom_tests = len(test_files["custom"])
        
        self.print_colored(f"Total test files: {total_tests}", Colors.WHITE)
        self.print_colored(f"  Unit tests: {unit_tests}", Colors.WHITE)
        self.print_colored(f"  Custom tests: {custom_tests}", Colors.WHITE)
        
        # Project structure
        self.print_colored(f"Project root: {self.project_root}", Colors.WHITE)
        self.print_colored(f"Tests directory: {self.tests_dir}", Colors.WHITE)
        self.print_colored(f"Source directory: {self.src_dir}", Colors.WHITE)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Manager for GAMA-Gymnasium",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_manager.py --all                           # Run all tests
  python test_manager.py --unit                          # Run unit tests only
  python test_manager.py --coverage                      # Run with coverage
  python test_manager.py --file test_space_converter.py  # Run specific file
  python test_manager.py --pattern "space"               # Run tests matching pattern
  python test_manager.py --quick                         # Quick smoke test
  python test_manager.py --list                          # List available tests
  python test_manager.py --install-deps                  # Install dependencies
        """
    )
    
    # Test execution options
    parser.add_argument("--all", action="store_true", 
                       help="Run all tests")
    parser.add_argument("--unit", action="store_true", 
                       help="Run unit tests only")
    parser.add_argument("--file", type=str, metavar="FILENAME",
                       help="Run tests from specific file")
    parser.add_argument("--pattern", type=str, metavar="PATTERN",
                       help="Run tests matching pattern")
    parser.add_argument("--coverage", action="store_true",
                       help="Run tests with coverage reporting")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick smoke test")
    
    # Configuration options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--fail-fast", "-x", action="store_true",
                       help="Stop on first failure")
    parser.add_argument("--install-deps", action="store_true",
                       help="Install test dependencies")
    
    # Information options
    parser.add_argument("--list", action="store_true",
                       help="List available tests")
    parser.add_argument("--info", action="store_true",
                       help="Show test environment information")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    try:
        manager = TestManager()
        
        # Print header
        manager.print_header("GAMA-Gymnasium Test Manager")
        
        results = []
        
        # Handle information commands first
        if args.info:
            manager.show_test_info()
            return 0
        
        if args.list:
            manager.list_available_tests()
            return 0
        
        # Install dependencies if requested
        if args.install_deps:
            success = manager.install_dependencies()
            if not success:
                return 1
        
        # Handle test execution commands
        if args.quick:
            results.append(("Quick Check", manager.run_quick_check()))
        
        if args.unit:
            results.append(("Unit Tests", manager.run_unit_tests(args.verbose, args.fail_fast)))
        
        if args.all:
            results.append(("All Tests", manager.run_all_tests(args.verbose, args.fail_fast)))
        
        if args.file:
            results.append((f"File: {args.file}", manager.run_specific_file(args.file, args.verbose)))
        
        if args.pattern:
            results.append((f"Pattern: {args.pattern}", manager.run_pattern_tests(args.pattern, args.verbose)))
        
        if args.coverage:
            scope = "unit" if args.unit else "all"
            results.append(("Coverage Tests", manager.run_with_coverage(scope, args.verbose)))
        
        # If no test commands specified, run unit tests by default
        if not any([args.quick, args.unit, args.all, args.file, args.pattern, args.coverage]):
            results.append(("Unit Tests (default)", manager.run_unit_tests(args.verbose, args.fail_fast)))
        
        # Print summary
        if results:
            manager.print_section("Test Summary")
            
            total_tests = len(results)
            passed_tests = sum(1 for _, success in results if success)
            failed_tests = total_tests - passed_tests
            
            for test_name, success in results:
                status = "✅ PASSED" if success else "❌ FAILED"
                color = Colors.GREEN if success else Colors.RED
                manager.print_colored(f"{status} {test_name}", color)
            
            manager.print_colored(f"\nTotal: {total_tests}, Passed: {passed_tests}, Failed: {failed_tests}", 
                                Colors.WHITE, bold=True)
            
            if failed_tests > 0:
                manager.print_colored("\n❌ Some tests failed!", Colors.RED, bold=True)
                return 1
            else:
                manager.print_colored("\n🎉 All tests passed!", Colors.GREEN, bold=True)
                return 0
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Tests interrupted by user{Colors.END}")
        return 130
    except Exception as e:
        print(f"{Colors.RED}❌ Error: {str(e)}{Colors.END}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
