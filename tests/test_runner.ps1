# PowerShell Test Runner for GAMA-Gymnasium
# Simple wrapper around the Python test manager

param(
    [string]$Action = "help",
    [string]$File = "",
    [string]$Pattern = "",
    [switch]$Verbose = $false,
    [switch]$Coverage = $false,
    [switch]$FailFast = $false
)

# Colors for output
$Green = "Green"
$Yellow = "Yellow" 
$Red = "Red"
$Cyan = "Cyan"
$White = "White"

# Function to print colored text
function Write-ColorText {
    param(
        [string]$Text,
        [string]$Color = "White"
    )
    Write-Host $Text -ForegroundColor $Color
}

# Check if we're in the right directory
if (-not (Test-Path "pytest.ini")) {
    Write-ColorText "❌ Error: pytest.ini not found. Please run from project root directory." $Red
    exit 1
}

# Check if Python test manager exists
if (-not (Test-Path "tests\test_manager.py")) {
    Write-ColorText "❌ Error: test_manager.py not found in tests directory." $Red
    exit 1
}

Write-ColorText "🧪 GAMA-Gymnasium PowerShell Test Runner" $Green
Write-ColorText "==========================================" $Green

# Determine Python command to use
$pythonCmd = "python"
if ($env:CONDA_DEFAULT_ENV) {
    Write-ColorText "📦 Using conda environment: $env:CONDA_DEFAULT_ENV" $Yellow
}

# Build command based on action
$testCmd = @($pythonCmd, "tests\test_manager.py")

switch ($Action.ToLower()) {
    "help" {
        Write-ColorText "📋 Available Actions:" $Cyan
        Write-ColorText "  help        - Show this help" $White
        Write-ColorText "  unit        - Run unit tests" $White
        Write-ColorText "  all         - Run all tests" $White
        Write-ColorText "  quick       - Quick smoke test" $White
        Write-ColorText "  list        - List available tests" $White
        Write-ColorText "  info        - Show test environment info" $White
        Write-ColorText "  install     - Install test dependencies" $White
        Write-ColorText "" $White
        Write-ColorText "📋 Additional Options:" $Cyan
        Write-ColorText "  -File       - Run specific test file" $White
        Write-ColorText "  -Pattern    - Run tests matching pattern" $White
        Write-ColorText "  -Coverage   - Generate coverage report" $White
        Write-ColorText "  -Verbose    - Verbose output" $White
        Write-ColorText "  -FailFast   - Stop on first failure" $White
        Write-ColorText "" $White
        Write-ColorText "📋 Examples:" $Cyan
        Write-ColorText "  .\test_runner.ps1 unit" $White
        Write-ColorText "  .\test_runner.ps1 all -Coverage" $White
        Write-ColorText "  .\test_runner.ps1 -File test_space_converter.py" $White
        Write-ColorText "  .\test_runner.ps1 -Pattern space -Verbose" $White
        exit 0
    }
    "unit" {
        $testCmd += "--unit"
        Write-ColorText "🔬 Running unit tests..." $Cyan
    }
    "all" {
        $testCmd += "--all"
        Write-ColorText "🚀 Running all tests..." $Cyan
    }
    "quick" {
        $testCmd += "--quick"
        Write-ColorText "⚡ Running quick smoke test..." $Cyan
    }
    "list" {
        $testCmd += "--list"
        Write-ColorText "📋 Listing available tests..." $Cyan
    }
    "info" {
        $testCmd += "--info"
        Write-ColorText "ℹ️ Showing test environment info..." $Cyan
    }
    "install" {
        $testCmd += "--install-deps"
        Write-ColorText "📦 Installing test dependencies..." $Cyan
    }
    default {
        Write-ColorText "❌ Unknown action: $Action" $Red
        Write-ColorText "Use 'help' to see available actions." $Yellow
        exit 1
    }
}

# Add file-specific option
if ($File -ne "") {
    $testCmd += "--file"
    $testCmd += $File
    Write-ColorText "🎯 Running tests from file: $File" $Yellow
}

# Add pattern option
if ($Pattern -ne "") {
    $testCmd += "--pattern"
    $testCmd += $Pattern
    Write-ColorText "🔍 Running tests matching pattern: $Pattern" $Yellow
}

# Add coverage option
if ($Coverage) {
    $testCmd += "--coverage"
    Write-ColorText "📊 Coverage reporting enabled" $Yellow
}

# Add verbose option
if ($Verbose) {
    $testCmd += "--verbose"
    Write-ColorText "📝 Verbose output enabled" $Yellow
}

# Add fail-fast option
if ($FailFast) {
    $testCmd += "--fail-fast"
    Write-ColorText "⚡ Fail-fast mode enabled" $Yellow
}

Write-ColorText "" $White
Write-ColorText "🚀 Executing: $($testCmd -join ' ')" $White
Write-ColorText "" $White

# Execute the command
try {
    & $testCmd[0] $testCmd[1..($testCmd.Length-1)]
    $exitCode = $LASTEXITCODE
} catch {
    Write-ColorText "❌ Error executing command: $_" $Red
    exit 1
}

Write-ColorText "" $White
if ($exitCode -eq 0) {
    Write-ColorText "✅ Command completed successfully!" $Green
} else {
    Write-ColorText "❌ Command failed with exit code: $exitCode" $Red
}

Write-ColorText "" $White
Write-ColorText "💡 Quick Tips:" $Cyan
Write-ColorText "  - Use 'list' to see all available tests" $White
Write-ColorText "  - Use 'quick' for a fast verification" $White
Write-ColorText "  - Use 'info' to check your test environment" $White
Write-ColorText "  - Add -Verbose for detailed output" $White

exit $exitCode
