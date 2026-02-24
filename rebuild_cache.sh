#!/bin/bash
# Rebuild Cache Script
# This will run the live signal generator briefly to rebuild the cache

echo "🔄 Rebuilding cache..."
echo "This will take about 1-2 minutes..."
echo ""

# Run live signal generator for 90 seconds to build cache
timeout 90 .venv/bin/python live_signal_generator.py || true

echo ""
echo "✅ Cache rebuilt!"
echo ""

# Check if cache exists now
if [ -f "live_candles_cache.pkl" ]; then
    SIZE=$(du -h live_candles_cache.pkl | cut -f1)
    echo "📊 Cache size: $SIZE"
    echo "✅ Ready to run regime analyzer"
else
    echo "❌ Cache file not created. Try running live_signal_generator.py manually."
fi
