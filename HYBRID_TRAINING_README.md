# Hybrid Training: Historical Data + Envio HyperSync

This system trains XGBoost models on a combination of historical data and real-time streaming data from Envio HyperSync.

## Overview

The hybrid training approach:
1. **Loads historical data** from processed parquet files (if available)
2. **Streams new data** from Envio HyperSync API
3. **Processes both datasets** through the same feature engineering pipeline
4. **Combines datasets** for comprehensive training
5. **Trains XGBoost model** on the combined dataset

## Usage

### Basic Usage

```bash
python3 hybrid_training.py
```

This will:
- Load historical data from `processed_data.parquet` (if exists)
- Stream data from Envio HyperSync for 5 minutes
- Process and combine both datasets
- Train XGBoost model
- Save model to `xgb_model.json`

### Configuration

Edit `hybrid_training.py` to customize:

```python
class TrainingConfig:
    # Historical data
    PROCESSED_DATA_PATH = "processed_data.parquet"
    
    # Envio HyperSync
    ENVIO_API_TOKEN = "your-token-here"
    STREAM_DURATION_SECONDS = 300  # 5 minutes
    
    # Model output
    MODEL_OUTPUT_PATH = "xgb_model.json"
```

## Workflow

```
┌─────────────────────┐
│ Historical Data     │
│ (parquet files)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     ┌─────────────────────┐
│ Feature Engineering │     │ Envio HyperSync     │
│ & Labeling          │◄────│ (Real-time stream)  │
└──────────┬──────────┘     └─────────────────────┘
           │
           ▼
┌─────────────────────┐
│ Combined Dataset    │
│ (Historical + New)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Train/Val/Test      │
│ Split (Time-based)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ XGBoost Training    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Saved Model         │
│ (xgb_model.json)    │
└─────────────────────┘
```

## Benefits

1. **Historical Context**: Learns from long-term patterns in historical data
2. **Current Market**: Incorporates latest market conditions from live stream
3. **Adaptive**: Model stays relevant as market conditions change
4. **Robust**: More data = better generalization

## Prerequisites

1. **Historical Data** (optional):
   - Run `python3 onchain_signal_system.py` to create `processed_data.parquet`
   - Or use existing processed data

2. **Envio HyperSync**:
   - Valid API token
   - Network connectivity
   - Base chain access

3. **Dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

## Output

- **Model**: `xgb_model.json` - Trained XGBoost model
- **Metrics**: Training and validation performance metrics
- **Console**: Detailed training progress and evaluation

## Troubleshooting

### "No historical data found"
- This is OK! Training will use only streamed data
- Or run `python3 onchain_signal_system.py` first to create processed data

### "HyperSync API 403 error"
- Check API token is valid
- Verify network connectivity
- Check Envio API status

### "No training data available"
- Ensure at least one data source (historical or streamed) has data
- Check data processing pipeline completed successfully

## Notes

- **Stream Duration**: Default is 5 minutes. Increase for more data, decrease for faster training
- **Data Quality**: Both historical and streamed data go through the same quality filters
- **Time Split**: Training uses time-based splits to prevent data leakage
- **Class Imbalance**: Automatically handles imbalanced classes with `scale_pos_weight`

## Example Output

```
################################################################################
HYBRID TRAINING: HISTORICAL + ENVIO HYPERSYNC
################################################################################

📂 Loading historical data from processed_data.parquet
✓ Loaded 1,234,567 historical samples

🚀 Starting HyperSync stream...
✓ Streamed 5,432 candles from HyperSync

✓ Combined dataset: 1,240,000 total samples
  Date range: 2020-01-01 to 2024-12-17
  Unique pairs: 12,345
  Positive labels: 62,000 (5.00%)

✓ Model training complete
  Best iteration: 245
  Best score: 0.8234

✓ Saved model to: xgb_model.json
```
