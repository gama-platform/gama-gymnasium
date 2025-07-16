@echo off
REM Batch script to run GAMA-Gymnasium tests
REM Usage: test_runner.bat [action] [options]

setlocal enabledelayedexpansion

REM Default values
set ACTION=%1
if "%ACTION%"=="" set ACTION=help

REM Colors (limited in batch)
set GREEN=[92m
set YELLOW=[93m
set RED=[91m
set CYAN=[96m
set WHITE=[97m
set RESET=[0m

echo %GREEN%🧪 GAMA-Gymnasium Batch Test Runner%RESET%
echo %GREEN%=====================================%RESET%

REM Check if we're in the right directory
if not exist "pytest.ini" (
    echo %RED%❌ Error: pytest.ini not found. Please run from project root directory.%RESET%
    exit /b 1
)

REM Check if Python test manager exists
if not exist "tests\test_manager.py" (
    echo %RED%❌ Error: test_manager.py not found in tests directory.%RESET%
    exit /b 1
)

REM Handle different actions
if /i "%ACTION%"=="help" goto :help
if /i "%ACTION%"=="unit" goto :unit
if /i "%ACTION%"=="all" goto :all
if /i "%ACTION%"=="quick" goto :quick
if /i "%ACTION%"=="list" goto :list
if /i "%ACTION%"=="info" goto :info
if /i "%ACTION%"=="install" goto :install

REM Default action
goto :help

:help
echo %CYAN%📋 Available Actions:%RESET%
echo   help        - Show this help
echo   unit        - Run unit tests
echo   all         - Run all tests
echo   quick       - Quick smoke test
echo   list        - List available tests
echo   info        - Show test environment info
echo   install     - Install test dependencies
echo.
echo %CYAN%📋 Examples:%RESET%
echo   test_runner.bat unit
echo   test_runner.bat all
echo   test_runner.bat quick
echo   test_runner.bat list
goto :end

:unit
echo %CYAN%🔬 Running unit tests...%RESET%
python tests\test_manager.py --unit
goto :end

:all
echo %CYAN%🚀 Running all tests...%RESET%
python tests\test_manager.py --all
goto :end

:quick
echo %CYAN%⚡ Running quick smoke test...%RESET%
python tests\test_manager.py --quick
goto :end

:list
echo %CYAN%📋 Listing available tests...%RESET%
python tests\test_manager.py --list
goto :end

:info
echo %CYAN%ℹ️ Showing test environment info...%RESET%
python tests\test_manager.py --info
goto :end

:install
echo %CYAN%📦 Installing test dependencies...%RESET%
python tests\test_manager.py --install-deps
goto :end

:end
echo.
echo %CYAN%💡 Quick Tips:%RESET%
echo   - Use 'list' to see all available tests
echo   - Use 'quick' for a fast verification
echo   - Use 'info' to check your test environment
echo   - For advanced options, use: python tests\test_manager.py --help

endlocal
