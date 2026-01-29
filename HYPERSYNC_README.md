# Envio HyperSync Real-Time Signal Generation

This system streams real-time DEX swap data from Envio HyperSync and generates trading signals using the trained XGBoost model.

## Overview

Instead of using static parquet files, the system now:
1. **Streams swap data** from Envio HyperSync API in real-time
2. **Converts swaps to candles** (1-minute aggregations)
3. **Engineers features** for the XGBoost model
4. **Generates 5 signals** after 1 minute of data collection

## Files

- `hypersync_streamer.py` - Envio HyperSync client and data streaming
- `realtime_signal_generator.py` - Main script to generate signals from streamed data

## Usage

### Basic Usage

```bash
python3 realtime_signal_generator.py
```

This will:
- Connect to Envio HyperSync (Base chain)
- Stream swap data for 60 seconds
- Generate 5 trading signals
- Save signals to `realtime_signals.json`

### Custom Options

```bash
python3 realtime_signal_generator.py \
    --model xgb_model.json \
    --token YOUR_API_TOKEN \
    --duration 60 \
    --signals 5 \
    --threshold 0.7
```

### Parameters

- `--model`: Path to trained XGBoost model (default: `xgb_model.json`)
- `--token`: Envio API token (default: provided token)
- `--duration`: Data collection duration in seconds (default: 60)
- `--signals`: Number of signals to generate (default: 5)
- `--threshold`: Probability threshold for signals (default: 0.7)

## Prerequisites

1. **Trained Model**: You need a trained XGBoost model. Train it first:
   ```bash
   python3 model_training.py
   ```

2. **API Token**: The system uses the provided Envio API token:
   ```
   ENVIO_API_TOKEN=1ae7c8f0-5cdf-4316-81f6-0fa3cb84aaa8
   ```

3. **Dependencies**: Install required packages:
   ```bash
   pip3 install -r requirements.txt
   ```

## How It Works

1. **HyperSync Connection**: Connects to Envio HyperSync API for Base chain
2. **Block Streaming**: Queries blocks in batches (500 blocks at a time)
3. **Swap Detection**: Identifies Uniswap V2 and V3 swap events
4. **Candle Aggregation**: Converts swaps to 1-minute candles
5. **Feature Engineering**: Creates features compatible with the trained model
6. **Signal Generation**: Uses XGBoost to predict and rank signals
7. **Output**: Displays top 5 signals and saves to JSON

## Output Format

Signals are saved to `realtime_signals.json` with the following structure:

```json
[
  {
    "timestamp": "2024-01-15T10:30:00",
    "pair_id": 12345,
    "pool_address": "0x...",
    "pred_proba": 0.85,
    "signal": 1,
    "return_1d": 2.5,
    "return_7d": 8.3,
    "volatility_7d": 15.2,
    "volume": 1000000,
    "buys": 150,
    "sells": 120
  }
]
```

## Notes

- **Limited History**: Real-time data has limited historical context, so some features (like 7d returns) are simplified
- **Model Requirements**: The model should be trained on similar feature distributions
- **Network**: Requires stable internet connection for HyperSync API
- **Rate Limits**: Be mindful of API rate limits when streaming

## Troubleshooting

### "Model not found"
Train the model first: `python3 model_training.py`

### "No candles collected"
- Check internet connection
- Verify API token is valid
- Check if HyperSync API is accessible

### "No signals generated"
- Lower the threshold: `--threshold 0.5`
- Increase duration: `--duration 120`
- Check if model predictions are reasonable

## Architecture

```
HyperSync API → Swap Events → Candle Aggregation → Feature Engineering → XGBoost → Signals
```

The system buffers swap data for 1 minute, then processes it through the same feature engineering pipeline used during training, ensuring compatibility with the trained model.
