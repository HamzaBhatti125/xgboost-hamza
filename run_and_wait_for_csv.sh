#!/bin/bash
# Script to run live_signal_generator and wait for CSV generation

cd "/home/user-008/Desktop/Genesis/XGBoost All/xgboost-hamza"

# Ensure ENVIO_API_TOKEN is available
if [ -z "$ENVIO_API_TOKEN" ]; then
    echo "❌ ENVIO_API_TOKEN not set"
    exit 1
fi

export ENVIO_API_TOKEN

echo "🚀 Starting live_signal_generator.py..."
echo "📊 Monitoring for CSV file generation..."
echo ""

# Run the script in background
python3 live_signal_generator.py > /tmp/signal_gen_monitor.log 2>&1 &
PID=$!

echo "Process started with PID: $PID"
echo ""

# Monitor for CSV files
MAX_WAIT=1800  # 30 minutes max
ELAPSED=0
INTERVAL=15    # Check every 15 seconds

while [ $ELAPSED -lt $MAX_WAIT ]; do
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    
    # Check if process is still running
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  Process exited. Checking logs..."
        tail -30 /tmp/signal_gen_monitor.log
        exit 1
    fi
    
    # Check for CSV files
    CSV_COUNT=$(find . -maxdepth 1 -name "signals*.csv" -type f 2>/dev/null | wc -l)
    
    if [ $CSV_COUNT -gt 0 ]; then
        echo ""
        echo "✅ CSV file(s) generated!"
        echo ""
        ls -lth signals*.csv 2>/dev/null | head -5
        echo ""
        
        # Check if regime column exists
        FIRST_CSV=$(ls -t signals*.csv 2>/dev/null | head -1)
        if [ -n "$FIRST_CSV" ]; then
            echo "📋 Checking first CSV file: $FIRST_CSV"
            echo ""
            head -2 "$FIRST_CSV" | cut -d',' -f1-15
            echo ""
            if head -1 "$FIRST_CSV" | grep -q "regime"; then
                echo "✅ Regime column found in CSV!"
            else
                echo "⚠️  Regime column NOT found in CSV"
            fi
        fi
        
        # Kill the process
        kill $PID 2>/dev/null
        wait $PID 2>/dev/null
        exit 0
    fi
    
    # Show progress every minute
    if [ $((ELAPSED % 60)) -eq 0 ]; then
        MINUTES=$((ELAPSED / 60))
        echo "[$MINUTES min] Still waiting for CSV generation... (checking every 15s)"
        # Show last few log lines
        tail -3 /tmp/signal_gen_monitor.log 2>/dev/null | sed 's/^/  /'
    fi
done

echo ""
echo "⏱️  Timeout reached after $MAX_WAIT seconds"
echo "Last log entries:"
tail -20 /tmp/signal_gen_monitor.log

# Kill the process
kill $PID 2>/dev/null
wait $PID 2>/dev/null

exit 1
