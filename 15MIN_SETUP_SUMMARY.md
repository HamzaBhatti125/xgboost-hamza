# 15-Minute Candle Setup Summary

## Overview
The system has been configured to use **15-minute candles** for live trading on Base chain via Envio Hypersync.

## Changes Made

### 1. Model Training ✅
- **File**: `train_15min_model.py`
- **Data Source**: `Files/candles-15m.parquet` (370M rows)
- **Model Output**: `xgboost_15min_model.json`
- **Performance**:
  - Test RMSE: 0.02081
  - Direction Accuracy: 56.78%
  - Top features: momentum_16, volume_vs_ma, momentum_4

### 2. Envio Hypersync Configuration ✅
- **File**: `envio_hypersync.py`
- **Changes**:
  ```python
  CANDLE_INTERVAL_SECONDS = 900   # 15 minutes
  CANDLE_INTERVAL_MINUTES = 15    # 15 minutes
  ```
- **Impact**: All swap events will be aggregated into 15-minute OHLCV candles

### 3. Live Signal Generator ✅
- **File**: `live_signal_generator.py`
- **Changes**:
  ```python
  MODEL_PATH = "xgboost_15min_model.json"  # Updated model
  MIN_HISTORY_CANDLES = 200                 # 50 hours of history
  UPDATE_INTERVAL_SECONDS = 900             # Signal updates every 15 min
  ```

## System Architecture

```
Envio Hypersync (Base Chain)
    ↓
  Swap Events
    ↓
15-Min Candle Aggregation
    ↓
Feature Engineering
    ↓
XGBoost Model (15min)
    ↓
Trading Signals
```

## Feature Engineering (15-Min Timeframe)

### Moving Averages
- **Short**: 4 candles = 1 hour
- **Medium**: 16 candles = 4 hours  
- **Long**: 96 candles = 24 hours

### Key Features
1. momentum_16 (4-hour momentum)
2. volume_vs_ma
3. momentum_4 (1-hour momentum)
4. Price lags (1, 4, 16 candles)
5. Volume lags and ratios

## Files Structure

### Model Files
- `xgboost_15min_model.json` - Trained 15-minute model
- `xgb_model.json` - Original daily model (kept for reference)

### Data Files
- `Files/candles-15m.parquet` - Historical 15-min candles (370M rows)
- `Files/candles-1d.parquet` - Daily candles (for analysis)

### Code Files
- `train_15min_model.py` - Model training script
- `envio_hypersync.py` - Real-time data streaming
- `live_signal_generator.py` - Live trading signals

## Usage

### Train the Model
```bash
source .venv/bin/activate
python train_15min_model.py
```

### Run Live Signal Generation
```bash
source .venv/bin/activate
python live_signal_generator.py
```

### Test the System
```bash
source .venv/bin/activate
python test_live_system.py
```

## Next Steps

1. **Test Live Stream**
   ```bash
   python test_live_system.py
   ```

2. **Monitor Performance**
   - Check signal quality
   - Verify candle generation
   - Monitor latency

3. **Production Deployment**
   - Set up monitoring
   - Configure alerts
   - Implement risk management

## Notes

- **Historical Analysis**: `onchain_signal_system.py` still uses daily candles for long-term analysis
- **Live Trading**: Uses 15-minute candles for real-time signals
- **Data Volume**: 15-min candles generate 4x more data than 1-hour, 96x more than daily
- **Memory**: Each 15-min candle batch ~40MB, plan accordingly

## Configuration Reference

| Component | Setting | Value |
|-----------|---------|-------|
| Candle Interval | CANDLE_INTERVAL_SECONDS | 900 (15 min) |
| Update Frequency | UPDATE_INTERVAL_SECONDS | 900 (15 min) |
| History Required | MIN_HISTORY_CANDLES | 200 (~50 hours) |
| Model Path | MODEL_PATH | xgboost_15min_model.json |
| Data Source | DATA_PATH | Files/candles-15m.parquet |

