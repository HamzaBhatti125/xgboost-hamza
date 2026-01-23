# Signal Generation Guide

## Overview

The `generate_signals.py` script allows you to view historical signals, generate predictions on latest data, and analyze model performance.

## Usage

### 1. View Historical Signals (from test set)

```bash
python generate_signals.py --mode history
```

**What it does:**

- Shows all signals generated on the test set (Oct 2024 - Dec 2025)
- Displays signal date, pair ID, probability, and actual outcome (WIN/LOSS)
- Calculates win rate and performance statistics
- **Saves signals to `signals_history.csv`**

**Output includes:**

- 39 signals found (0.18% signal rate)
- Win rate: 25.6% (10 wins out of 39 signals)
- Each signal with timestamp, pair ID, confidence probability

### 2. Generate Latest Signals (for live trading)

```bash
python generate_signals.py --mode latest
```

**What it does:**

- Analyzes the most recent data point for each pair
- Shows top predictions by confidence
- Identifies active BUY signals (probability ≥ 0.8 threshold)
- **Saves top 50 predictions to `signals_latest.csv`**

**Use case:**

- Run this daily/weekly to see if any pairs trigger BUY signals
- Top predictions show pairs closest to triggering
- Currently no active signals above 0.8 threshold (highest is 0.696)

### 3. Analyze Model Predictions

```bash
python generate_signals.py --mode analyze
```

**What it does:**

- Shows prediction probability distribution across test set
- Calculates win rate at different thresholds (0.5, 0.6, 0.7, 0.8, 0.9)
- Analyzes characteristics of high-confidence signals
- Helps optimize threshold selection

**Key findings:**

- Threshold 0.8: 25.6% win rate, 0.18% signal rate (39 signals)
- Threshold 0.7: 18.7% win rate, 1.30% signal rate (283 signals)
- Threshold 0.6: 20.4% win rate, 9.44% signal rate (2,047 signals)

### Adjusting the Threshold

You can customize the probability threshold:

```bash
# More conservative (fewer, higher quality signals)
python generate_signals.py --mode history --threshold 0.9

# More aggressive (more signals, lower quality)
python generate_signals.py --mode history --threshold 0.7
```

**Threshold guide:**

- **0.8** (default): Very selective, 0.18% signal rate, ~26% win rate
- **0.7**: Moderate, 1.30% signal rate, ~19% win rate
- **0.6**: Aggressive, 9.44% signal rate, ~20% win rate
- **0.5**: Very aggressive, 37% signal rate, ~20% win rate

## Understanding the Output

### Signal Display Format

```
2025-10-26 00:00:00  |  Pair: 5185342  |  Prob: 0.822  |  ❌ LOSS  |  Ret1d: +0.0%  Vol: 0.0%
```

- **Date**: When the signal was generated
- **Pair ID**: Trading pair identifier
- **Prob**: Model's confidence (0-1)
- **Outcome**: ✅ WIN (hit +8% before -3%) or ❌ LOSS
- **Ret1d**: 1-day return at signal time
- **Vol**: 7-day volatility

### CSV Files

**signals_history.csv** contains:

- `timestamp`: Signal date
- `pair_id`: Trading pair
- `pred_proba`: Prediction probability
- `signal`: 1 if above threshold, 0 otherwise
- `signal_label`: Actual outcome (1=WIN, 0=LOSS)
- `return_1d/3d/7d`: Historical returns
- `volatility_7d`: Recent volatility

**signals_latest.csv** contains:

- Same columns but for the most recent data
- Use `pred_proba` to rank pairs by conviction
- Pairs with `signal=1` are active BUY signals

## Predicting on New Data

To use the trained model on completely new data:

1. **Prepare your data** with these columns:

   ```python
   required_features = [
       "return_1d", "return_3d", "return_7d",
       "ema_cross_signal", "range_normalized",
       "buy_sell_ratio", "volume_zscore", "trade_accel",
       "volatility_7d", "volatility_14d", "vol_regime_change"
   ]
   ```

2. **Load model and predict**:

   ```python
   from generate_signals import SignalGenerator

   generator = SignalGenerator()
   generator.load_model()

   # Your new data as polars DataFrame
   predictions = generator.predict(your_df)

   # Filter to signals only
   buy_signals = predictions.filter(pl.col("signal") == 1)
   ```

3. **Or integrate into your pipeline**:
   - Run `onchain_signal_system.py` with new candle data
   - It will generate `processed_data.parquet` with features
   - Use `generate_signals.py --mode latest` to see signals

## Real-World Usage

**Daily routine:**

```bash
# 1. Update data (if you have new candles)
python onchain_signal_system.py

# 2. Check for new signals
python generate_signals.py --mode latest

# 3. If signals found, review details in signals_latest.csv
```

**Backtesting different thresholds:**

```bash
# Test multiple thresholds
for threshold in 0.7 0.75 0.8 0.85 0.9; do
    python generate_signals.py --mode history --threshold $threshold
done
```

## Notes

- **Signal Rate**: 0.18% at threshold 0.8 means ~1 signal per 550 samples
- **Win Rate**: 25.6% is below the 57.1% seen in backtesting because threshold 0.8 was optimal in validation, test set may differ
- **No Recent Signals**: Currently no pairs above 0.8 threshold (highest 0.696), which is normal for rare-event strategy
- **CSV Files**: Saved after each run, can be imported into Excel/pandas for further analysis

## Feature Engineering

The model uses these 11 features:

1. **return_1d/3d/7d**: Price momentum
2. **ema_cross_signal**: EMA(7) vs EMA(21) crossover
3. **range_normalized**: High-low range (volatility proxy)
4. **buy_sell_ratio**: Buy vs sell volume
5. **volume_zscore**: Volume spike detection
6. **trade_accel**: Trade count acceleration
7. **volatility_7d/14d**: Rolling volatility
8. **vol_regime_change**: Volatility regime shifts

All features are calculated in `onchain_signal_system.py` during data preparation.
