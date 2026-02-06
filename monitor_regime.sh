#!/bin/bash
# Monitor for regime detector activation

echo "🔍 Monitoring live signal generator for regime detector activation..."
echo "   Log file: signal_gen_with_regime.log"
echo "   Waiting for next 15-minute signal generation cycle..."
echo ""
echo "Looking for:"
echo "  ✓ Model loading"
echo "  ✓ Regime detector loading"
echo "  ✓ Filter activation"
echo "  ✓ Crash detection and blocking"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo "=" | tr '=' '=' | head -c 80
echo ""

# Monitor the log for key events
tail -f signal_gen_with_regime.log | grep --line-buffered -E "(Loaded model|MarketRegimeDetector|APPLYING ADVANCED REGIME|CRASH RISK|REGIME FILTER RESULTS|Generating signals|SIGNALS GENERATED)"
