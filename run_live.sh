#!/bin/bash
# QUICK START: Run Live Signal Generator with Regime Detection

echo "🚀 Live Trading Signal Generator with Regime Detection"
echo "=================================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

echo "✅ Environment ready"
echo ""

# Check required files
echo "🔍 Checking required files..."
FILES=(
    "xgb_model.json"
    "xgb_calibrator.pkl"
    "regime_detector.pkl"
)

MISSING=0
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (MISSING)"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "❌ Missing required files!"
    echo "Make sure to train models first:"
    echo "  python model_training.py"
    echo "  python calibrate_model.py"
    echo "  python quickstart_regime_detector.py train"
    exit 1
fi

echo ""
echo "🎯 Starting live signal generator..."
echo "=================================================="
echo ""

# Run the live signal generator
python live_signal_generator.py

echo ""
echo "✅ Done!"
