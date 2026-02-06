# Market Regime Detector - Advanced Risk Management System

## 🎯 Overview

The `MarketRegimeDetector` is a sophisticated risk management system that dynamically adjusts position sizing and triggers emergency exits based on real-time market conditions. It uses **Gaussian Hidden Markov Models (HMM)** for regime classification and **Isolation Forest** for anomaly detection.

### Key Features

- **Dynamic Position Sizing**: Scales trade sizes from 0% to 100% based on crash state probability
- **Emergency Stop System**: Instantly blocks trades during anomalous market conditions
- **Liquidity-Aware**: Leverages your new `avg_liquidity` and `vwap` fields for fragility detection
- **High Performance**: Optimized for Polars DataFrames with vectorized operations
- **Production Ready**: Save/load functionality, comprehensive logging, error handling

---

## 📊 How It Works

### Architecture

```
Market Data (Polars DataFrame)
         ↓
   Feature Engineering (5 features)
         ↓
    ┌────────────────────────┐
    │   Standard Scaler      │
    │   (Normalize features) │
    └────────────────────────┘
         ↓
    ┌─────────────────┬──────────────────┐
    │  Gaussian HMM   │ Isolation Forest │
    │  (3 States)     │  (Anomalies)     │
    └─────────────────┴──────────────────┘
         ↓                    ↓
   position_scalar      is_emergency
   (0.0 to 1.0)         (True/False)
         ↓                    ↓
    ┌────────────────────────────┐
    │   Risk-Adjusted Signal     │
    │   final_size = base_size   │
    │     × confidence           │
    │     × position_scalar      │
    └────────────────────────────┘
```

### The Two-Layer Defense System

#### Layer 1: HMM Position Sizing (Gradual Risk Adjustment)
The HMM identifies three hidden market states:
- **State 0**: Low volatility (Bull/Bear calm)
- **State 1**: Medium volatility (Normal trading)
- **State 2**: High volatility (Crash/Crisis) ← **"Crash State"**

**How it works:**
1. Trains on historical data to learn volatility patterns
2. Identifies which state has the highest variance (the "Crash State")
3. At prediction time: `position_scalar = 1.0 - P(Crash State)`
   - If P(Crash) = 0%  → position_scalar = 1.0 (full size)
   - If P(Crash) = 30% → position_scalar = 0.7 (reduce to 70%)
   - If P(Crash) = 100% → position_scalar = 0.0 (block trade)

#### Layer 2: Isolation Forest (Binary Emergency Stop)
The Isolation Forest detects outliers in the feature space:
- Trained to flag the top 1% of anomalous market conditions
- Returns `is_emergency = True` if current data is an outlier
- **Use case**: Sudden liquidity drains, flash crashes, "rug pulls"

**Example Scenarios:**

| Market Condition | HMM State | Position Scalar | Isolation Forest | Final Action |
|------------------|-----------|-----------------|------------------|--------------|
| Normal market | State 1 | 0.9 | Normal | ✅ Trade at 90% size |
| Elevated volatility | State 2 (40% prob) | 0.6 | Normal | ⚠️ Trade at 60% size |
| Crash detected | State 2 (100% prob) | 0.0 | Normal | 🚫 Block trade |
| Liquidity rug pull | State 1 | 0.8 | **ANOMALY** | 🚨 Emergency - Block ALL |

---

## 🔧 Implementation Guide

### Step 1: Install Dependencies

```bash
pip install hmmlearn scikit-learn polars numpy joblib
```

### Step 2: Train the Detector

```python
from market_regime_detector import MarketRegimeDetector
import polars as pl
import pickle

# Load your candle history
with open("live_candles_cache.pkl", 'rb') as f:
    candles_history = pickle.load(f)

# Combine data from multiple pairs
training_data = [df for df in candles_history.values() if len(df) >= 20]
combined_df = pl.concat(training_data)

# Initialize and train
detector = MarketRegimeDetector(lookback_window=100)
detector.fit(combined_df)

# Save for production
detector.save("regime_detector.pkl")
```

**Training Requirements:**
- **Minimum samples**: 100 candles (from one or multiple pairs)
- **Recommended**: 500-1000 candles from 50+ pairs for robust learning
- **Retraining frequency**: Weekly or after major market events

### Step 3: Integrate with Signal Generator

```python
# In your live_signal_generator.py or trading bot

from market_regime_detector import MarketRegimeDetector

# Load trained detector (do this once at startup)
detector = MarketRegimeDetector()
detector.load("regime_detector.pkl")

# When you get a trading signal
pair_address = "0x123..."
base_confidence = 0.85  # From your XGBoost model
pair_candles = candles_history[pair_address]

# Get regime prediction
regime_prediction = detector.predict(pair_candles)

# Apply risk adjustment
if regime_prediction['is_emergency']:
    print("🚨 EMERGENCY - Blocking trade")
    continue  # Skip this trade

position_scalar = regime_prediction['position_scalar']
adjusted_confidence = base_confidence * position_scalar

if adjusted_confidence < 0.5:  # Your minimum threshold
    print(f"⚠️ Adjusted confidence too low ({adjusted_confidence:.2f}) - Skipping")
    continue

# Execute trade with adjusted size
execute_trade(pair_address, size=base_size * adjusted_confidence)
```

### Step 4: Monitor Performance

```python
# Log regime predictions
logger.info(f"Pair: {pair_address[:10]}...")
logger.info(f"  Base Confidence: {base_confidence:.2f}")
logger.info(f"  Position Scalar: {position_scalar:.2f}")
logger.info(f"  Adjusted: {adjusted_confidence:.2f}")
logger.info(f"  Regime: {regime_prediction['current_regime']}")
logger.info(f"  Crash Prob: {regime_prediction['crash_state_prob']:.1%}")

# Track statistics
n_emergency_exits = 0
n_reduced_positions = 0
avg_position_scalar = []

if regime_prediction['is_emergency']:
    n_emergency_exits += 1
elif position_scalar < 1.0:
    n_reduced_positions += 1

avg_position_scalar.append(position_scalar)
```

---

## 🧪 Feature Engineering Details

The detector generates 5 features from your candle data:

### 1. **Log Returns** (Momentum)
```python
log_returns = ln(close / close.shift(1))
```
- Measures price momentum
- Normalized for percentage moves
- Used to detect sudden price swings

### 2. **Parkinson Volatility** (Intrabar Volatility)
```python
parkinson_volatility = (ln(high / low)^2) / (4 * ln(2))
```
- More efficient than close-to-close volatility
- Captures intrabar price range
- High values = high uncertainty

### 3. **Amihud Illiquidity** (Price Impact / Market Fragility) ⭐
```python
amihud_illiquidity = abs(log_returns) / (volume_usd + 1)
```
- **This is the "killer app" feature**
- High value = Low liquidity / Fragile market
- Detects when small volumes cause large price moves
- Combines your price and volume data

**Example:**
- Price moves 5% on $10,000 volume → Illiquidity = 0.05 / 10000 = 0.000005 (healthy)
- Price moves 5% on $100 volume → Illiquidity = 0.05 / 100 = 0.0005 (fragile!)

### 4. **Liquidity Shock** (Rug Pull Detection) ⭐
```python
liquidity_shock = (avg_liquidity - avg_liquidity.shift(1)) / avg_liquidity.shift(1)
```
- Uses your new `avg_liquidity` field
- Sudden negative values = liquidity drain
- Detects potential rug pulls or panic withdrawals

**Example:**
- Liquidity drops from $50k to $30k → Shock = -40% 🚨
- Liquidity stable → Shock ≈ 0% ✅

### 5. **VWAP Deviation** (Price vs Fair Value) ⭐
```python
vwap_deviation = (close - vwap) / vwap
```
- Uses your new `vwap` field
- Positive = Price above fair value (overheated?)
- Negative = Price below fair value (oversold?)
- Large deviations = potential reversal or manipulation

---

## 📈 Expected Behavior

### Normal Market Conditions
```
🟢 Regime: 0 or 1
🟢 Position Scalar: 0.8 - 1.0
🟢 Emergency: False
✅ Result: Trade at 80-100% size
```

### Elevated Volatility
```
🟡 Regime: Varies
🟡 Position Scalar: 0.5 - 0.8
🟡 Emergency: False
⚠️ Result: Trade at 50-80% size (cautious)
```

### Crash Conditions
```
🔴 Regime: 2 (Crash State)
🔴 Position Scalar: 0.0 - 0.3
🔴 Emergency: False
🚫 Result: Block trade or minimal size
```

### Anomaly Detected
```
🚨 Regime: Any
🚨 Position Scalar: Any
🚨 Emergency: True
🛑 Result: BLOCK ALL TRADES immediately
```

---

## 🔄 Retraining Strategy

### When to Retrain

1. **Weekly Schedule**: Retrain every 7 days to adapt to market evolution
2. **After Major Events**: Black swan events, protocol hacks, market structure changes
3. **Performance Degradation**: If emergency stops become too frequent (>5%) or too rare (<0.1%)

### How to Retrain

```python
# Load fresh data
with open("live_candles_cache.pkl", 'rb') as f:
    candles_history = pickle.load(f)

# Combine recent history (last 30 days)
recent_pairs = [df.tail(2880) for df in candles_history.values() if len(df) >= 100]
combined_df = pl.concat(recent_pairs)

# Retrain
detector = MarketRegimeDetector(lookback_window=100)
detector.fit(combined_df)

# Save with timestamp
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d")
detector.save(f"regime_detector_{timestamp}.pkl")

# Deploy
detector.save("regime_detector.pkl")  # Overwrite production model
```

### Monitoring Health

Track these metrics weekly:

```python
# 1. Emergency Stop Rate
emergency_rate = n_emergency_stops / n_total_signals
# Target: 0.5% - 2%
# Too high (>5%): Model too conservative, retrain
# Too low (<0.1%): Model too lenient, increase contamination

# 2. Average Position Scalar
avg_scalar = mean(position_scalars)
# Target: 0.7 - 0.9
# Too low (<0.6): Model sees constant danger
# Too high (>0.95): Model not differentiating risk

# 3. Crash State Distribution
crash_distribution = {
    0: count_state_0,
    1: count_state_1,
    2: count_state_2
}
# Healthy distribution: roughly balanced
# If one state dominates >80%: retrain with more diverse data
```

---

## 🎛️ Configuration Tuning

### HMM Parameters

```python
detector = MarketRegimeDetector(lookback_window=100)  # Lookback for features

# In the HMM initialization (market_regime_detector.py):
self.hmm = GaussianHMM(
    n_components=3,        # Number of states (don't change)
    covariance_type="full", # Full covariance matrix (most flexible)
    n_iter=100,            # Max EM iterations (increase if not converging)
    random_state=42        # Reproducibility
)
```

**Tuning Guide:**
- `n_iter`: Increase to 200 if you see "Model not converging" warnings
- `lookback_window`: 
  - 50-100: Responsive to recent changes (crypto markets)
  - 200-500: More stable, less reactive (traditional markets)

### Isolation Forest Parameters

```python
self.isolation_forest = IsolationForest(
    contamination=0.01,  # Top 1% are anomalies
    n_estimators=100,    # Number of trees
    max_samples='auto',  # Subsample size
    random_state=42
)
```

**Tuning Guide:**
- `contamination`: 
  - 0.005 (0.5%): Very conservative, rare emergencies
  - 0.01 (1%): **Recommended** for crypto
  - 0.02 (2%): More aggressive, frequent stops
- `n_estimators`: 
  - 50: Faster prediction, less accuracy
  - 100: **Recommended** balance
  - 200: More accurate, slower

---

## 🧰 Troubleshooting

### Issue: "Model is not converging"

**Cause:** HMM EM algorithm didn't reach optimal solution

**Solutions:**
1. Increase `n_iter` from 100 to 200
2. Ensure you have at least 100 samples for training
3. Check for constant features (all zeros) - indicates data quality issue

### Issue: Too many emergency stops (>5%)

**Cause:** Isolation Forest too sensitive

**Solutions:**
1. Decrease `contamination` from 0.01 to 0.005
2. Retrain with more diverse data
3. Check if there's a persistent data quality issue (e.g., missing VWAP values)

### Issue: Position scalar always near 1.0

**Cause:** HMM not finding distinct states, or crash state misidentified

**Solutions:**
1. Train on more volatile historical data
2. Verify features have variance (not all zeros)
3. Manually inspect `state_variances` during training

### Issue: "Insufficient data" errors

**Cause:** Not enough candles in cache

**Solutions:**
1. Wait for live system to accumulate data (20+ candles per pair)
2. Use lower thresholds temporarily (5+ candles)
3. Use historical backfill data

---

## 📊 Real-World Example

```python
# Scenario: Market suddenly dumps -10% due to major news

# Pair A: High liquidity, normal VWAP
detector.predict(pair_a_candles)
# Returns:
# {
#   'position_scalar': 0.4,        # Reduced to 40% due to volatility spike
#   'is_emergency': False,          # Not an anomaly (broad market move)
#   'current_regime': 2,            # Crash state
#   'crash_state_prob': 0.6         # 60% crash probability
# }
# Action: Trade at 40% size (still trading, just cautious)

# Pair B: Liquidity dropped 50% + price below VWAP
detector.predict(pair_b_candles)
# Returns:
# {
#   'position_scalar': 0.1,         # Nearly zero (95% crash prob)
#   'is_emergency': True,           # Anomaly detected!
#   'current_regime': 2,            # Crash state
#   'crash_state_prob': 0.95        # 95% crash probability
# }
# Action: BLOCK trade (emergency + near-zero scalar)
```

**Result:** You avoided Pair B (likely a rug pull) while still participating in Pair A (legitimate market move with reduced size).

---

## 🚀 Next Steps

1. **Train your first model**:
   ```bash
   python regime_integration_example.py
   ```

2. **Integrate with live system**: See `live_signal_generator.py` integration section above

3. **Monitor for 7 days**: Collect statistics on emergency stops and position scalars

4. **Tune parameters**: Adjust `contamination` and `lookback_window` based on performance

5. **Set up weekly retraining**: Automate model updates to adapt to market changes

---

## 📚 References

- **Gaussian HMM**: Hidden Markov Model for time series regime detection
- **Isolation Forest**: Anomaly detection in high-dimensional spaces  
- **Amihud Illiquidity**: Market microstructure measure (Amihud, 2002)
- **Parkinson Volatility**: Efficient volatility estimator (Parkinson, 1980)

---

## ⚠️ Important Notes

1. **This is not a signal generator**: It's a risk management layer on top of your XGBoost signals
2. **Position scalar is multiplicative**: `final_confidence = xgb_confidence × position_scalar`
3. **Emergency stops override everything**: When `is_emergency=True`, block ALL trades
4. **Retraining is critical**: Markets evolve, your model must adapt
5. **Test thoroughly**: Backtest on historical data before going live

---

**Status**: ✅ Production Ready  
**Created**: February 5, 2026  
**Version**: 1.0
