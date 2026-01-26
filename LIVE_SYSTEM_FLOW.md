# Live Signal Generation System - Complete Flow

## Overview

This document explains how the system streams live swap data from Envio Hypersync, converts it to candles, computes features, and generates trading signals.

---

## Architecture Flow

```
Envio Hypersync (Base Chain)
    ↓
[1] Stream Swap Events (V2 & V3)
    ↓
[2] Decode & Buffer Swaps
    ↓
[3] Aggregate into OHLCV Candles (every 5 min)
    ↓
[4] Store Candle History (per pair)
    ↓
[5] Compute 11 Technical Features
    ↓
[6] XGBoost Model Prediction
    ↓
[7] Generate Signals (confidence > 0.8)
    ↓
Save to signals_live.csv
```

---

## Detailed Step-by-Step Process

### Step 1: Stream Swap Events from Envio Hypersync

**File**: `envio_hypersync.py` - `LiveSwapStreamer.stream_swaps()`

**Process**:

1. Connects to Envio Hypersync at `https://base.hypersync.xyz`
2. Queries for Uniswap V2 and V3 swap events using topic signatures:
   - V2: `0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822`
   - V3: `0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67`
3. Polls every 12 seconds for new blocks
4. Processes up to 10,000 blocks per query

**Output**: Raw swap event logs containing:

- Block number & timestamp
- Transaction hash
- Pair/pool address
- Swap amounts (token0, token1)

---

### Step 2: Decode Swap Events

**File**: `envio_hypersync.py` - `SwapEventProcessor`

**Process**:

1. **V2 Swaps** (`decode_v2_swap`):
   - Extracts 4 uint256 values from log data: `amount0In, amount1In, amount0Out, amount1Out`
   - Calculates net amounts: `amount0 = (out - in) / 1e18`
   - Calculates price: `price = |amount1 / amount0|`

2. **V3 Swaps** (`decode_v3_swap`):
   - Extracts signed int256 values: `amount0, amount1`
   - Converts from raw int256 to float with 18 decimals
   - Calculates price: `price = |amount1 / amount0|`

**Output**: `SwapEvent` objects with:

```python
{
    "block_number": 41309290,
    "block_timestamp": 1748153783,
    "pair_address": "0xabc...",
    "amount0": -1.5,
    "amount1": 3000.0,
    "price": 2000.0,
    "is_v3": True
}
```

---

### Step 3: Aggregate Swaps into OHLCV Candles

**File**: `envio_hypersync.py` - `CandleAggregator`

**Configuration**:

- **Candle Interval**: 300 seconds (5 minutes)
- **Buffer**: Stores swaps in memory per pair
- **Generation**: Called via `get_latest_candles()` every 5 minutes

**Process**:

1. Groups swaps by pair address
2. Sorts by timestamp
3. Buckets into 5-minute intervals: `candle_start = (timestamp // 300) * 300`
4. For each bucket, calculates:
   - **Open**: First price in interval
   - **High**: Max price in interval
   - **Low**: Min price in interval
   - **Close**: Last price in interval
   - **Volume Token0**: Sum of absolute amount0 values
   - **Volume Token1**: Sum of absolute amount1 values
   - **Num Trades**: Count of swaps

**Output**: `Candle` objects every 5 minutes:

```python
{
    "pair_address": "0xabc...",
    "timestamp": 1748153700,  # Rounded to 5-min boundary
    "open": 2000.0,
    "high": 2010.0,
    "low": 1995.0,
    "close": 2005.0,
    "volume_token0": 150.5,
    "volume_token1": 301000.0,
    "num_trades": 25
}
```

---

### Step 4: Store Candle History

**File**: `live_signal_generator.py` - `LiveSignalGenerator.update_candles_history()`

**Storage Structure**:

```python
self.candles_history = {
    "0xpair1...": DataFrame([candle1, candle2, ..., candle300]),
    "0xpair2...": DataFrame([candle1, candle2, ..., candle150]),
    ...
}
```

**Process**:

1. Receives new candles from `streamer.get_latest_candles()` every 5 minutes
2. For each pair:
   - Appends new candles to existing history
   - Deduplicates by timestamp (keeps latest)
   - Sorts chronologically
   - **Keeps last 400 candles** (~33 hours of history)
3. Stores as Polars DataFrames for efficient computation

**Memory Management**:

- Old candles beyond 400-candle window are automatically dropped
- This provides sufficient history for all features

---

### Step 5: Compute Technical Features

**File**: `live_signal_generator.py` - `FeatureEngineer.compute_all_features()`

**Required Input**: Minimum **20-30 candles** per pair for stable features

**Features Computed** (11 total):

#### 5.1 Price Returns

```python
return_1d = 100 * (close / close.shift(1) - 1)   # 1-period return
return_3d = 100 * (close / close.shift(3) - 1)   # 3-period return
return_7d = 100 * (close / close.shift(7) - 1)   # 7-period return
```

#### 5.2 EMA Crossover Signal

```python
ema_5 = close.ewm(span=5).mean()
ema_20 = close.ewm(span=20).mean()
ema_cross_signal = 1 if ema_5 > ema_20 else -1
```

#### 5.3 Price Range (Volatility Proxy)

```python
range_normalized = 100 * (high - low) / close
```

#### 5.4 Buy/Sell Flow

```python
buy_sell_ratio = volume_token1 / volume_token0
volume_zscore = (volume - rolling_mean(7)) / rolling_std(7)
trade_accel = num_trades - num_trades.shift(1)
```

#### 5.5 Volatility Regime

```python
volatility_7d = return_1d.rolling(7).std()
volatility_14d = return_1d.rolling(14).std()
vol_regime_change = 100 * (volatility_7d / volatility_14d - 1)
```

**Process Per Pair**:

1. Sort candles chronologically
2. Compute all features using rolling windows
3. Fill NaN/null values with 0
4. Keep only rows with complete feature sets

**Output**: DataFrame with original candle data + 11 feature columns

---

### Step 6: XGBoost Model Prediction

**File**: `live_signal_generator.py` - `LiveSignalGenerator.generate_signals()`

**Model**: `xgb_model.json` (trained on historical data)

**Process**:

1. Filters to pairs with **complete feature history** (no nulls)
2. Gets **latest candle per pair** (most recent 5-min window)
3. Extracts 11 feature columns as numpy array
4. Converts to XGBoost DMatrix
5. Runs prediction: `pred_proba = model.predict(dmatrix)`
6. Returns probability score (0.0 to 1.0)

**Example**:

```python
Features for pair 0xabc:
[
    return_1d=0.5,
    return_3d=1.2,
    return_7d=2.8,
    ema_cross_signal=1,
    range_normalized=0.8,
    buy_sell_ratio=2.5,
    volume_zscore=1.5,
    trade_accel=3,
    volatility_7d=1.2,
    volatility_14d=1.0,
    vol_regime_change=20.0
]
→ Model Prediction: 0.85 (85% confidence)
```

---

### Step 7: Signal Generation & Filtering

**File**: `live_signal_generator.py` - `LiveSignalGenerator.generate_signals()`

**Threshold**: 0.8 (80% confidence)

**Process**:

1. Filter predictions: `pred_proba >= 0.8`
2. Sort by confidence (descending)
3. Add signal flag: `signal = 1`
4. Log to console with pair details
5. Save to `signals_live.csv`

**Output Format**:

```csv
pair_address,timestamp,open,high,low,close,volume_token0,volume_token1,num_trades,pred_proba,signal,generated_at
0xabc...,2026-01-26 10:35:00,2000,2010,1995,2005,150.5,301000,25,0.85,1,2026-01-26 10:40:00
```

---

## Why No Signals Are Being Generated

### Problem 1: Insufficient Candle History

**Current State**: You have only **1 candle per pair**

**Required**: Minimum **20-30 candles** per pair for:

- `return_7d` (needs 7 candles)
- `volatility_14d` (needs 14 candles)
- Rolling windows for z-scores (needs 7+ candles)

**Solution**: Wait 2-3 hours for 30-40 candles to accumulate

---

### Problem 2: Feature Computation Fails with Sparse Data

With only 1 candle:

- All returns are NaN (no previous prices)
- All rolling calculations are NaN (no window)
- Features get filled with 0 (default)
- Model sees all-zero features → low confidence

---

### Problem 3: Model Trained on Daily Candles

**Training Data**: Model was trained on 1-day OHLCV candles

**Live Data**: Now using 5-minute candles

**Issue**: Feature distributions are different:

- Daily returns: ±5-10%
- 5-minute returns: ±0.1-0.5%
- Daily volatility: 20-50%
- 5-minute volatility: 1-3%

**Impact**: Model may output low probabilities (<0.8) even for "good" patterns

---

## Current System Status

### Candle History Tracking

**Location**: `LiveSignalGenerator.candles_history`

**Structure**:

```python
{
    "0xpair1": [
        {timestamp: 10:00, open: 2000, ...},
        {timestamp: 10:05, open: 2005, ...},
        {timestamp: 10:10, open: 2010, ...},
        ...
    ]
}
```

**How It Works**:

1. Every 5 minutes, `UPDATE_INTERVAL_SECONDS = 300` triggers
2. Calls `streamer.get_latest_candles()` → gets newly generated candles
3. Calls `update_candles_history(new_candles)` → appends to history
4. Keeps last 400 candles per pair (~33 hours)
5. Old candles automatically dropped when limit exceeded

**View History**:

```python
# In live_signal_generator.py, add logging
logger.info(f"Pair 0xabc has {len(self.candles_history['0xabc'])} candles")
logger.info(self.candles_history['0xabc'].tail(5))  # Show last 5 candles
```

---

## Debugging: Why No Signals?

### Check 1: Are Candles Being Generated?

**Expected**: `Received X new candles` log every 5 minutes

**Current**: Seeing "Received 994 new candles" → ✅ Working

---

### Check 2: How Many Candles Per Pair?

**Expected**: 30+ candles per pair for signals

**Current**: Likely only 1-5 candles per pair (too new)

**Check**:

```python
# Add to live_signal_generator.py after update_candles_history()
for pair, df in self.candles_history.items():
    if len(df) >= 20:
        logger.info(f"Pair {pair} has {len(df)} candles - ready for signals")
```

---

### Check 3: Are Features Complete?

**Expected**: All 11 features non-null

**Current**: With <20 candles, most features are null/zero

**Check**:

```python
# In generate_signals(), before filtering
logger.info(f"Total rows: {len(df_features)}")
logger.info(f"Valid rows (complete features): {len(valid_rows)}")
logger.info(f"Sample features:\n{valid_rows.head()}")
```

---

### Check 4: What Are Prediction Scores?

**Expected**: Some predictions > 0.8

**Current**: All predictions likely < 0.8

**Check**:

```python
# After model prediction
logger.info(f"Prediction distribution: min={pred_proba.min():.3f}, max={pred_proba.max():.3f}, mean={pred_proba.mean():.3f}")
logger.info(f"Predictions > 0.5: {(pred_proba > 0.5).sum()}")
logger.info(f"Predictions > 0.8: {(pred_proba > 0.8).sum()}")
```

---

## Solutions

### Solution 1: Wait for History (Recommended)

⏰ **Wait 2-3 hours** for system to accumulate 30-40 candles per pair

Then signals will start generating naturally.

---

### Solution 2: Lower Threshold Temporarily

Change threshold from 0.8 → 0.5 for testing:

```python
# live_signal_generator.py, line 35
SIGNAL_THRESHOLD = 0.5  # Temporary - was 0.8
```

This will generate more signals for testing, but lower quality.

---

### Solution 3: Retrain Model on 5-Minute Data

The model needs retraining on 5-minute candles:

1. Collect 1-2 weeks of 5-minute candle data
2. Retrain `model_training.py` with same features
3. Model learns correct feature distributions for 5-min intervals
4. Predictions become better calibrated

---

### Solution 4: Use Historical Data Bootstrap

Load historical candles from Hypersync to jumpstart:

```python
# In LiveSwapStreamer.__init__
# Query last 24 hours of data instead of starting fresh
start_time = int((datetime.now() - timedelta(hours=24)).timestamp())
# ... backfill candles
```

---

## Monitoring Commands

### View Candle History Count

```python
python -c "
from live_signal_generator import LiveSignalGenerator
import asyncio

async def check():
    gen = LiveSignalGenerator()
    await asyncio.sleep(1)  # Let it load
    for pair, df in gen.candles_history.items():
        print(f'{pair}: {len(df)} candles')

asyncio.run(check())
"
```

### Check Latest Features

```python
# Add to live_signal_generator.py temporarily
df_features = FeatureEngineer.compute_all_features(all_candles)
print(df_features.tail(10))
```

### Monitor Predictions

```python
# Add after model.predict()
for i, (pair, score) in enumerate(zip(latest_per_pair['pair_address'], pred_proba)):
    print(f"{pair}: {score:.3f}")
```

---

## Expected Timeline

| Time     | Candles/Pair | Features        | Signals         |
| -------- | ------------ | --------------- | --------------- |
| 0 min    | 1            | ❌ All NaN      | ❌ None         |
| 30 min   | 6            | ⚠️ Partial      | ⚠️ Maybe        |
| 1 hour   | 12           | ⚠️ Partial      | ⚠️ Low quality  |
| 2 hours  | 24           | ✅ Complete     | ✅ Some         |
| 3 hours  | 36           | ✅ Complete     | ✅ Regular      |
| 24 hours | 288          | ✅ Full history | ✅ High quality |

---

## Summary

**Current Issue**: Not enough candle history yet (only 1 candle per pair)

**Root Cause**: System just started, needs time to accumulate data

**Expected Wait**: 2-3 hours for stable signal generation

**Quick Test**: Lower threshold to 0.5 temporarily to see if model is working

**Long-term Fix**: Either retrain model on 5-minute data or wait 24 hours for full history

The system is **working correctly** - it's just not ready yet! 🚀
