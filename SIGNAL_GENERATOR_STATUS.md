# Signal Generator Status

## Current Status

The live signal generator is **running in the background** and will generate both OHLCV and Bollinger Band signals.

### Process Status
- **Running**: Yes (check with `ps aux | grep live_signal_generator`)
- **Log File**: `signal_generator.log`
- **Output File**: `signals_live.csv` (will be created when signals are generated)

### Requirements for Signal Generation

1. **OHLCV Signals (XGBoost Model)**:
   - Need 20+ candles per pair (currently accumulating)
   - Prediction probability >= 0.32 (calibrated threshold)
   - Generated every 15 minutes

2. **Bollinger Band Signals**:
   - Need candles with BB data (from live streaming)
   - Triggers on:
     - Price touching lower band (oversold = buy)
     - Price touching upper band (overbought = sell)
     - BB Squeeze (narrow bands = breakout potential)

### Current Data Status

- **Total Pairs**: ~3,500 pairs tracked
- **Total Candles**: ~9,000 candles
- **Candles per Pair**: Most pairs have 15 candles (need 20+)
- **BB Data**: Will be available from live streaming (not in cached data)

### What's Happening Now

1. ✅ Backfilling 25 hours of historical data
2. ✅ Streaming live swaps with Bollinger Bands enabled
3. ⏳ Accumulating candles (need 20+ per pair)
4. ⏳ Waiting for update interval (15 minutes)

### Expected Timeline

- **First Signals**: Within 1-2 hours (once 20+ candles accumulated per pair)
- **Update Frequency**: Every 15 minutes
- **Signal Types**: Both OHLCV and BB signals will be saved to `signals_live.csv`

### Output Format

Signals will be saved to `signals_live.csv` with columns:
- `signal_type`: "OHLCV" or "BB" (to distinguish signal types)
- `pair_address`: Trading pair address
- `pred_proba`: Prediction probability (for OHLCV) or signal strength (for BB)
- `close`: Current price
- `take_profit_price`: Target price (+2%)
- `stop_loss_price`: Stop loss price (-1%)
- `take_profit_pct`: 2.0%
- `stop_loss_pct`: 1.0%
- `holding_period_days`: 1 day
- `generated_at`: Timestamp
- Plus additional BB-specific fields for BB signals

### Monitoring

```bash
# Check if process is running
ps aux | grep live_signal_generator

# View recent logs
tail -f signal_generator.log

# Check for generated signals
ls -lh signals*.csv

# View signals (if generated)
head signals_live.csv
```

### Next Steps

1. Wait for the process to accumulate 20+ candles per pair
2. Check `signal_generator.log` periodically for progress
3. Once signals are generated, they'll be in `signals_live.csv`
4. Compare OHLCV vs BB signals using the `signal_type` column
