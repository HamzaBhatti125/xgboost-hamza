# Where Signals Are Saved

## Main Signal File

**File**: `signals_live.csv`  
**Location**: `/home/user-008/Desktop/Genesis/XGBoost All/xgboost-hamza/signals_live.csv`

This is the **main file** where all signals are saved (both OHLCV and BB signals).

## Timestamped Signal Files

**Format**: `signals_YYYY-MM-DD_HH-MM-SS.csv`  
**Example**: `signals_2026-01-30_14-31-37.csv`

These are **snapshot files** created each time signals are generated, for easy comparison.

## Current Status

**File exists**: ❌ Not yet (signals haven't been generated)

**Why?**
- Process is still accumulating data (need 20+ candles per pair)
- Currently: Max 16 candles per pair (need 20+)
- Update interval: Every 15 minutes
- First signals expected: Within 1-2 hours

## Signal File Format

The CSV file contains these columns:

### Common Columns (Both Signal Types)
- `signal_type`: "OHLCV" or "BB" (to distinguish)
- `pair_address`: Trading pair address
- `close`: Current price
- `take_profit_pct`: 2.0%
- `take_profit_price`: Target price
- `stop_loss_pct`: 1.0%
- `stop_loss_price`: Stop loss price
- `holding_period_days`: 1 day
- `generated_at`: Timestamp when signal was generated

### OHLCV Signal Columns
- `pred_proba`: XGBoost prediction probability (0-1)
- `signal`: 1 (active signal)
- Plus all feature columns (return_1c, ema_cross_signal, etc.)

### BB Signal Columns
- `pred_proba`: Signal strength (0-1, based on BB position)
- `bb_signal_reason`: "bb_lower_touch", "bb_upper_touch", or "bb_squeeze"
- `bb_position`: Position within bands (0-1)
- `bb_signal_strength`: Calculated strength
- `bb_middle`, `bb_upper`, `bb_lower`, `bb_width`: BB values

## Sample File

I've created a sample file to show the format:
- **File**: `signals_live_SAMPLE.csv`
- This shows what the actual output will look like

## How to Check

```bash
# Check if signals file exists
ls -lh signals_live.csv

# View signals (when generated)
head signals_live.csv
cat signals_live.csv

# Filter by signal type
grep "OHLCV" signals_live.csv
grep "BB" signals_live.csv

# Count signals by type
grep -c "OHLCV" signals_live.csv
grep -c "BB" signals_live.csv
```

## When Will Signals Be Generated?

1. ✅ Process is running (check: `ps aux | grep live_signal_generator`)
2. ⏳ Accumulating candles (currently 16 max, need 20+)
3. ⏳ Waiting for 15-minute update interval
4. ⏳ Once 20+ candles per pair: Signals will be generated

**Expected**: Within 1-2 hours from now

## Monitoring

```bash
# Check process status
ps aux | grep live_signal_generator

# View logs
tail -f signal_generator.log

# Check for signal files
ls -lh signals*.csv

# Watch for new signals
watch -n 10 'ls -lh signals*.csv 2>/dev/null || echo "No signals yet"'
```
