# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Verify Your Setup

```bash
cd /home/hamzabhatti18/Desktop/Genesis-labs/xgboost

# Check Python version (need 3.8+)
python --version

# Verify virtual environment is activated
which python  # Should show .venv/bin/python
```

### 2. Verify Data Files

The system expects these files:

```bash
# Pair universe (directory with parquet files)
/home/hamzabhatti18/Desktop/Genesis-labs/backtesting/pair-universe/*.parquet

# Candles (single parquet file)
/home/hamzabhatti18/Desktop/Genesis-labs/backtesting/candles-1d.parquet
```

Check if they exist:

```bash
ls -lh /home/hamzabhatti18/Desktop/Genesis-labs/backtesting/pair-universe/
ls -lh /home/hamzabhatti18/Desktop/Genesis-labs/backtesting/candles-1d.parquet
```

### 3. Run Quick Data Check

```bash
python check_data.py
```

Expected output:

```
Pair universe path: /home/.../pair-universe
  Exists: True
  Parquet files: 12

Candles path: /home/.../candles-1d.parquet
  Exists: True
  Size: 450.23 MB

LOADING SAMPLE DATA...
  Shape: (5, 24)
  Columns: ['pair_id', 'chain_id', 'buy_tax', ...]
```

If files are missing, update paths in `onchain_signal_system.py`:

```python
class Config:
    PAIR_UNIVERSE_PATH = "/YOUR/PATH/HERE"
    CANDLES_PATH = "/YOUR/PATH/HERE"
```

### 4. Run Full Pipeline

**Option A: One command (recommended)**

```bash
python run_pipeline.py
```

This runs everything automatically:

- Data loading & filtering
- Feature engineering
- Model training
- Backtesting
- Validation

**Option B: Step by step**

```bash
# Step 1: Data preparation (20-60 seconds)
python onchain_signal_system.py

# Step 2: Training & backtesting (2-5 minutes)
python model_training.py
```

### 5. Interpret Results

The final output will show:

**Example A: Profitable Strategy ✅**

```
================================================================================
FINAL CONCLUSION
================================================================================

✅ PROFITABLE STRATEGY IDENTIFIED

💰 Expected Performance:
   • Sharpe Ratio: 1.42
   • Win Rate: 56.2%
   • Avg Return/Trade: +3.8%
   • Max Drawdown: 19.2%

🎯 What makes this work:
   • Strict filtering removes 92% of scam pairs
   • Rare signals (only high conviction)
   • Realistic friction costs included
   • Time-based validation prevents leakage
```

**Example B: No Edge ❌**

```
================================================================================
FINAL CONCLUSION
================================================================================

❌ NO PROFITABLE EDGE DETECTED

📊 Issues:
   • Sharpe Ratio: 0.6 (need ≥ 1.0)
   • Max Drawdown: 31.2% (need ≤ 25%)

🔍 Why:
   • On-chain markets are highly efficient
   • High friction costs (0.5%) erode returns
   • Insufficient predictive power from OHLCV alone

💡 What could help:
   • Incorporate wallet-level data
   • Social sentiment signals
   • Lower friction DEX selection
```

## ⚙️ Configuration

### Key Parameters to Adjust

Edit `onchain_signal_system.py`:

```python
class Config:
    # Change these based on your goals

    # More aggressive filtering (fewer but cleaner pairs)
    MIN_VOLUME_30D = 50000          # Default: 10000
    MIN_AVG_DAILY_TRADES = 10       # Default: 5

    # Different risk profile
    LABEL_UPSIDE_PCT = 10.0         # Default: 8.0 (higher = rarer signals)
    LABEL_DOWNSIDE_PCT = 2.0        # Default: 3.0 (tighter = more stopped out)
    LABEL_HORIZON_DAYS = 5          # Default: 7 (shorter = faster trades)

    # More conservative costs
    FEE_PCT = 0.5                   # Default: 0.3
    SLIPPAGE_PCT = 0.5              # Default: 0.2
```

### Different Chain

To analyze a different chain:

```python
class Config:
    BASE_CHAIN_ID = 1       # Ethereum
    # BASE_CHAIN_ID = 56    # BSC
    # BASE_CHAIN_ID = 8453  # Base (default)
    # BASE_CHAIN_ID = 42161 # Arbitrum
```

**Note**: You must retrain from scratch for each chain.

## 🐛 Troubleshooting

### Issue: "File not found"

```bash
FileNotFoundError: /home/.../pair-universe
```

**Solution**: Update paths in `onchain_signal_system.py`:

```python
PAIR_UNIVERSE_PATH = "/correct/path/to/pair-universe"
CANDLES_PATH = "/correct/path/to/candles-1d.parquet"
```

### Issue: "Out of memory"

```bash
MemoryError: Unable to allocate array
```

**Solution**: Reduce chunk size:

```python
class Config:
    CHUNK_SIZE = 5000  # Default: 10000
```

Or filter more aggressively:

```python
MIN_VOLUME_30D = 50000  # Default: 10000
```

### Issue: "No signals generated"

```bash
WARNING: No trades executed!
```

**Solution**: Your filters are too strict or threshold too high.

1. Relax filtering:

   ```python
   MIN_VOLUME_30D = 5000    # From 10000
   MIN_AVG_DAILY_TRADES = 2  # From 5
   ```

2. Lower signal threshold in `model_training.py`:
   ```python
   SIGNAL_THRESHOLD = 0.6  # From 0.7
   ```

### Issue: "All pairs removed"

```
Remaining: 0 pairs (100% removed)
```

**Solution**:

1. Check if you're using the right chain_id
2. Verify data quality (are there any Base chain pairs?)
3. Temporarily disable filters to see what's removing everything:

```python
# Comment out filters one by one
# df = df.filter(pl.col("age_days") >= Config.MIN_AGE_DAYS)
```

### Issue: "Model predicts all 0s"

```
Signal rate: 0.00%
```

**Solution**: Class imbalance too extreme or labeling too strict.

1. Relax labeling:

   ```python
   LABEL_UPSIDE_PCT = 5.0   # From 8.0
   LABEL_HORIZON_DAYS = 10  # From 7
   ```

2. Check positive class rate in labels (should be 2-5%):
   ```python
   print(df['signal_label'].mean() * 100)
   ```

## 📊 Analyzing Results

### Feature Importance

After training, check which features matter:

```python
importance = model.get_score(importance_type='gain')
for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{feat}: {score:.1f}")
```

**Healthy pattern:**

```
buy_sell_ratio: 1250.3
ema_cross_signal: 980.2
volume_zscore: 720.5
return_7d: 650.1
```

**Warning signs:**

```
feature_X: 5000.0  ← One feature dominates (overfitting?)
range_normalized: 12.1  ← Expected important feature is weak (data issue?)
```

### Trade Analysis

Look at individual trades:

```python
trades_df = backtest_results['trades_df']

# Best trades
print(trades_df.nlargest(10, 'pnl_pct'))

# Worst trades
print(trades_df.nsmallest(10, 'pnl_pct'))

# By holding period
print(trades_df.groupby('days_held')['pnl_pct'].mean())
```

### Walk-Forward Validation

Test on multiple time periods:

```python
# Edit model_training.py
periods = [
    ('2024-Q1', '2024-01-01', '2024-03-31'),
    ('2024-Q2', '2024-04-01', '2024-06-30'),
    ('2024-Q3', '2024-07-01', '2024-09-30'),
]

for name, start, end in periods:
    df_period = df_test.filter(
        (pl.col('timestamp') >= start) &
        (pl.col('timestamp') <= end)
    )
    results = backtest_strategy(df_period, predictions)
    print(f"{name}: Sharpe = {results['sharpe_ratio']:.2f}")
```

## 🎯 Next Steps

### If Profitable ✅

1. **Robustness tests**
   - Test on different chains
   - Test on different time periods
   - Sensitivity analysis on parameters

2. **Improve strategy**
   - Add more sophisticated exit logic
   - Dynamic position sizing
   - Multi-timeframe confirmation

3. **Prepare for live trading**
   - Build real-time data pipeline
   - Connect to DEX aggregator API
   - Implement monitoring & alerting

### If Not Profitable ❌

1. **Try different approaches**
   - Different chains (Ethereum, Arbitrum, etc.)
   - Shorter timeframes (4h, 1h candles)
   - Different labeling (longer horizons, asymmetric barriers)

2. **Add alternative data**
   - Wallet-level flows
   - Social sentiment (Twitter, Telegram)
   - Order book data
   - Cross-chain arbitrage signals

3. **Optimize execution**
   - Find lower-friction DEXs
   - Use limit orders instead of market
   - Optimize timing (avoid high-traffic periods)

## 📚 Further Reading

- **METHODOLOGY.md**: Deep dive into each step
- **README.md**: Complete system documentation
- **XGBoost docs**: https://xgboost.readthedocs.io/
- **Polars docs**: https://pola-rs.github.io/polars-book/

## 🤝 Getting Help

### Check Logs

All output is printed to console. Look for:

- ❌ Error messages
- ⚠️ Warnings
- 📊 Statistics that seem wrong

### Common Questions

**Q: How long should this take?**
A:

- Data prep: 30-120 seconds
- Training: 2-5 minutes
- Total: < 10 minutes

**Q: How much data do I need?**
A: Minimum 6 months of daily candles for training + 3 months for testing

**Q: Can I use this for live trading?**
A: System is research-grade. For live trading, add:

- Real-time data integration
- Order execution logic
- Risk management
- Monitoring & alerts

**Q: Why so much filtering?**
A: On-chain data is 95% garbage. Better to trade 100 clean pairs than 10,000 scams.

## ⚡ Quick Commands Reference

```bash
# Full pipeline
python run_pipeline.py

# Just data prep
python onchain_signal_system.py

# Just training
python model_training.py

# Data check
python check_data.py

# Install packages
pip install polars pyarrow xgboost lightgbm scikit-learn numpy pandas matplotlib seaborn
```

---

**Remember**: Most on-chain datasets won't produce alpha. That's normal and expected. The system will tell you honestly if there's no edge.

**Good luck! 🚀**
