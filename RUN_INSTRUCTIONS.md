# 🚀 HOW TO RUN: Live Signal Generator with Regime Detection

## Quick Start (3 commands)

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run the live system
python live_signal_generator.py

# 3. Signals will be generated every 15 minutes
# Output: signals_YYYY-MM-DD_HH-MM-SS.csv
```

That's it! ✅

---

## What Happens When You Run

```
1. STARTUP (first 10-20 seconds)
   ├─ Load XGBoost model
   ├─ Load calibrator (probability calibration)
   ├─ Load regime detector (crash detection)
   └─ Load/build candle cache

2. BACKFILL (30-60 seconds, on first run)
   ├─ Fetch 25 hours of historical candles
   ├─ Build technical features
   └─ Store in cache

3. STREAMING (continuous)
   ├─ Stream live swaps from Hypersync
   ├─ Build 15-minute candles per pair
   ├─ Every 15 minutes: Generate signals
   ├─ Every 300 seconds: Save cache
   └─ On Ctrl+C: Save cache & exit gracefully

4. SIGNAL GENERATION (every 15 minutes)
   ├─ Compute 11 technical features
   ├─ Run XGBoost predictions
   ├─ Calibrate probabilities
   ├─ Filter with regime detector ⭐ (NEW)
   ├─ Apply position sizing
   └─ Output: signals_YYYY-MM-DD_HH-MM-SS.csv
```

---

## Output Files

### Main: `signals_YYYY-MM-DD_HH-MM-SS.csv`

CSV with columns:
```
pair_address          : Token pair address
timestamp             : Candle timestamp
pred_proba            : XGBoost prediction (0.0-1.0)
position_scalar       : Regime multiplier (0.0-1.0)  ⭐ NEW
adjusted_confidence   : pred_proba × position_scalar ⭐ NEW
regime                : HMM state (0=normal, 1=mixed, 2=crash) ⭐ NEW
is_emergency          : Anomaly detected? ⭐ NEW
selected              : Pass portfolio filter?
adjusted_position_size: Final position size (%)
expected_value        : Expected return %
```

### Supporting: `live_candles_cache.pkl`
- Cached candle data (rebuilt when interrupted)
- Contains 50+ recent candles per pair
- Atomic writes prevent corruption

### Supporting: `features.json`
- Technical features for debugging
- One row per signal

---

## Regime Detector Integration

The regime detector acts as a **filtering layer** after XGBoost predictions:

```
XGBoost Signal (e.g., pred_proba=0.40)
    ↓
Regime Detector
├─ Input: Pair's 50 recent candles
├─ Models:
│  ├─ HMM: Identifies market regime (Normal/Mixed/Crash)
│  └─ Isolation Forest: Detects anomalies
├─ Output:
│  ├─ position_scalar: 0.0 (crash) to 1.0 (normal)
│  ├─ crash_state_prob: 0.0 to 1.0
│  └─ is_emergency: True/False
    ↓
Adjusted Confidence = 0.40 × position_scalar
    ↓
Filter: Keep if adjusted_confidence ≥ 0.5
```

**Example:**
- Signal: pred_proba=0.40 ✓
- Regime: Crash state (position_scalar=0.0)
- Result: adjusted_confidence=0.0 ✗ BLOCKED

---

## Configuration

All settings in `live_signal_generator.py` → `LiveConfig` class:

### Core
```python
MODEL_PATH = "xgb_model.json"              # XGBoost model
CALIBRATOR_PATH = "xgb_calibrator.pkl"     # Probability calibrator
USE_CALIBRATION = True                      # Apply calibration

SIGNAL_THRESHOLD = 0.32                     # Min prediction threshold
UPDATE_INTERVAL_SECONDS = 900               # 15 minutes (matches candles)
MIN_HISTORY_CANDLES = 20                    # Min candles required
```

### Regime Detector (NEW)
```python
USE_REGIME_DETECTOR = True                  # Enable crash detection
REGIME_DETECTOR_PATH = "regime_detector.pkl" # Trained model
REGIME_DETECTOR_MIN_CONFIDENCE = 0.5        # Min adjusted confidence
```

### Position Sizing
```python
USE_POSITION_SIZING = True
KELLY_FRACTION = 0.25                       # 1/4 Kelly
MAX_POSITIONS = 10                          # Max 10 concurrent
MAX_CAPITAL_DEPLOYED = 0.50                 # Max 50% capital
```

---

## Logs & Monitoring

### Key Log Messages

**Startup:**
```
✓ Loaded model from xgb_model.json
✓ Loaded calibrator from xgb_calibrator.pkl
✓ Loaded MarketRegimeDetector from regime_detector.pkl
   Crash State: State 2
```

**Signal Generation (every 15 min):**
```
Processing 57191 candles for 10481 pairs (avg: 5.5 candles/pair)
✅ Computing predictions for 192 valid rows (from 192 total)
📊 Calibration: raw mean=0.069 → calibrated mean=0.073
📊 Prediction stats: min=0.056, max=0.508, mean=0.074
📈 Confidence levels: >0.5: 3, >0.7: 0, >0.8: 0
📊 Position Sizing: 10/192 signals selected, 49.7% capital deployed

🎯 APPLYING ADVANCED REGIME DETECTOR FILTER
🔴 CRASH RISK (100%): 0x00d43aad... - scalar=0.00
🔴 CRASH RISK (100%): 0x026dabaf... - scalar=0.00
...

📊 REGIME FILTER RESULTS:
   Input Signals: 192
   🚨 Emergency Blocks: 2
   ⚠️  Low Confidence: 190
   ✅ Passed Filter: 0
   📉 Filter Rate: 100.0%
   📊 Avg Position Scalar: 0.00

📈 CUMULATIVE STATS:
   Total Processed: 192
   Blocked: 2 (1.0%)
   Reduced: 190 (99.1%)
```

---

## Troubleshooting

### "EOFError: Ran out of input" (cache corrupted)
```bash
# Delete corrupted cache
rm -f live_candles_cache.pkl

# Restart - will rebuild automatically
python live_signal_generator.py
```

### "Model not found: xgb_model.json"
```bash
# Train model first
python model_training.py
python calibrate_model.py
```

### "Regime detector not found"
```bash
# Train detector first
python quickstart_regime_detector.py train

# Or use the dedicated trainer
python retrain_regime_detector.py --data-source cache
```

### No signals being generated
```bash
# Need 20+ candles per pair first
# Wait 1-2 backfill cycles (5-10 minutes)
# OR restart with --skip-backfill=false
```

---

## Advanced: Disable Regime Detection (Temporarily)

Edit `live_signal_generator.py`:
```python
class LiveConfig:
    USE_REGIME_DETECTOR = False  # Disable regime filtering
```

Then restart. Signals will pass through without regime adjustment.

---

## Advanced: Adjust Regime Sensitivity

Edit `live_signal_generator.py`:
```python
class LiveConfig:
    REGIME_DETECTOR_MIN_CONFIDENCE = 0.3  # Lower = more signals pass (more risk)
    REGIME_DETECTOR_MIN_CONFIDENCE = 0.7  # Higher = fewer signals (more caution)
```

Default 0.5 = medium caution (blocks ~50% during crashes, ~10% during normal)

---

## Performance Notes

- **First run:** 30-60 seconds (backfill)
- **Subsequent runs:** 5-10 seconds (load from cache)
- **Signal generation:** ~10-15 seconds every 15 minutes
- **Memory:** 500MB-1GB (depends on cache size)
- **Disk I/O:** ~1MB per signal generation cycle

---

## What Gets Logged

In terminal (real-time):
- Data streaming progress
- Signal generation stats
- Regime detection results
- Position sizing decisions

In `signals_*.csv`:
- Actual trading signals with regime info
- Position sizes and confidence levels
- Ready for backtesting/execution

In `live_candles_cache.pkl`:
- Cached candle data (binary)
- Automatically rebuilt on crash
- ~10MB per 10k pairs

---

## Next Steps

1. ✅ Run: `python live_signal_generator.py`
2. ⏱️ Wait for first 15-minute signal cycle
3. 📊 Check `signals_*.csv` for results
4. 🔍 Monitor regime filter statistics
5. 📈 Adjust sensitivity if needed

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python live_signal_generator.py` | Run live system |
| `./run_live.sh` | Run with checks |
| `python model_training.py` | Train XGBoost model |
| `python calibrate_model.py` | Calibrate probabilities |
| `python quickstart_regime_detector.py train` | Train regime detector |
| `python quickstart_regime_detector.py test` | Test regime detector |
| `rm live_candles_cache.pkl` | Reset cache |

---

## Support

If signals aren't generating:
1. Check logs for errors
2. Verify model files exist
3. Clear cache: `rm live_candles_cache.pkl`
4. Restart: `python live_signal_generator.py`

Expected: Signals every 15 minutes after ~5 min warmup ✅
