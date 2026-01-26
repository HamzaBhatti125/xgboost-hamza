#!/bin/bash

# Setup and Test Live Trading System
# ===================================
# This script sets up a virtual environment and tests the live system

set -e  # Exit on error

echo "=================================="
echo "  Live Trading System Setup"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo "ℹ️  $1"
}

# Check Python version
echo "Step 1: Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
print_info "Python version: $python_version"

# Extract major and minor version
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
    print_error "Python 3.8+ required, found $python_version"
    exit 1
else
    print_success "Python version OK"
fi

# Check if venv exists
echo ""
echo "Step 2: Setting up virtual environment..."

if [ -d "venv" ]; then
    print_warning "Virtual environment already exists"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing old venv..."
        rm -rf venv
    fi
fi

if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_info "Using existing virtual environment"
fi

# Activate venv
echo ""
echo "Step 3: Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
echo ""
echo "Step 4: Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

# Install requirements
echo ""
echo "Step 5: Installing dependencies..."
print_info "This may take a few minutes..."

if pip install -r requirements.txt; then
    print_success "Dependencies installed"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# List installed packages
echo ""
echo "Installed packages:"
pip list | grep -E "(polars|xgboost|web3|aiohttp|numpy|pandas|scikit)"

# Run tests
echo ""
echo "Step 6: Running system tests..."
echo ""

if python test_live_system.py; then
    print_success "All tests passed!"
    
    echo ""
    echo "=================================="
    echo "  Setup Complete!"
    echo "=================================="
    echo ""
    echo "Your live trading system is ready to use."
    echo ""
    echo "Next Steps:"
    echo ""
    echo "1. Train model (if not done already):"
    echo "   python run_pipeline.py"
    echo ""
    echo "2. Start live signal generator:"
    echo "   python live_signal_generator.py"
    echo ""
    echo "3. Monitor signals:"
    echo "   tail -f signals_live.csv"
    echo ""
    echo "4. Test Envio streaming:"
    echo "   python envio_hypersync.py"
    echo ""
    echo "=================================="
else
    print_error "Some tests failed"
    echo ""
    echo "Please review the test output above and fix any issues."
    echo ""
    exit 1
fi
