#!/bin/bash
# Generate new signals and track them automatically

set -e

echo "=================================================================================="
echo "🚀 GENERATE AND TRACK SIGNALS"
echo "=================================================================================="

# Activate venv and set API token
source venv/bin/activate
export ENVIO_API_TOKEN="1ae7c8f0-5cdf-4316-81f6-0fa3cb84aaa8"

echo ""
echo "📊 Step 1: Generating new signals..."
echo "=================================================================================="
python3 quick_backfill_and_signals.py

echo ""
echo "📂 Step 2: Finding latest signal file..."
echo "=================================================================================="

# Find the most recent timestamped signal file
LATEST_SIGNAL=$(ls -t signals_2026-*.csv 2>/dev/null | head -1)

if [ -z "$LATEST_SIGNAL" ]; then
    echo "❌ No signal file found! Check signals_live.csv instead."
    LATEST_SIGNAL="signals_live.csv"
fi

echo "✅ Found: $LATEST_SIGNAL"

echo ""
echo "📈 Step 3: Tracking signals..."
echo "=================================================================================="
python3 track_signals_cache.py "$LATEST_SIGNAL"

echo ""
echo "✅ Complete! Results saved to: ${LATEST_SIGNAL%.csv}_tracked_cache.csv"
echo "=================================================================================="
