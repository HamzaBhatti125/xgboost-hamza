# Understanding the Trading Signals

## What Are These Signals?

### **These are BUY signals for LONG positions (not sell signals)**

When the model generates a signal (`signal = 1`), it means:

> **"BUY this pair now. There's a high probability it will hit +8% profit BEFORE hitting -3% stop loss within the next 7 days"**

## Signal Components

### 1. **Historical Signals (signals_history.csv)**

These are **past predictions** with **known outcomes**:

```
2025-10-26 | Pair: 5185342 | Prob: 0.822 | ❌ LOSS | Ret1d: +0.0% | Vol: 0.0%
2025-07-16 | Pair: 3719989 | Prob: 0.801 | ✅ WIN  | Ret1d: +0.1% | Vol: 0.0%
```

**What this means:**

- **Date**: When the model would have said "BUY this pair"
- **Prob**: Model's confidence (0.822 = 82.2% confident)
- **Outcome**:
  - ✅ **WIN** = Price hit +8% BEFORE hitting -3% stop loss
  - ❌ **LOSS** = Price hit -3% stop loss OR failed to reach +8% in 7 days
- **Purpose**: Evaluate how well the model performed historically

### 2. **Latest Signals (signals_latest.csv)**

These show predictions for the **most recent data point** of each pair:

```
🟢 BUY  | Pair: 6289079 | Prob: 0.696 | Date: 2025-12-08
⚪ HOLD | Pair: 4004679 | Prob: 0.655 | Date: 2025-12-10
```

**What this means:**

- **🟢 BUY** (signal=1): Model says buy this pair (prob ≥ 0.8 threshold)
- **⚪ HOLD** (signal=0): Model says don't buy (prob < 0.8 threshold)
- **Date**: The last date we have data for this pair

## Trading Strategy Explained

### Entry Signal (BUY)

```
IF model_probability >= 0.8:
    → BUY at next day's open price
```

### Exit Rules

```
AFTER BUYING:
1. Set take-profit at +8%
2. Set stop-loss at -3%
3. Maximum holding period: 7 days
4. Exit when FIRST condition hits:
   - Price reaches +8% profit → SELL (WIN) ✅
   - Price drops to -3% loss → SELL (LOSS) ❌
   - 7 days pass → SELL at market price
```

### Example Trade Flow

**Day 0 (Signal Generated):**

```
Date: 2025-07-16
Pair: 3719989
Model says: BUY (prob = 0.801)
```

**Day 1 (Entry):**

```
Buy at open: $1.00
Set TP: $1.08 (+8%)
Set SL: $0.97 (-3%)
```

**Days 2-7 (Monitoring):**

```
Day 2: Price = $1.02 → HOLD (neither TP nor SL hit)
Day 3: Price = $1.05 → HOLD
Day 4: Price = $1.09 → HIT TAKE PROFIT! SELL ✅
Result: WIN (+9% profit)
```

## Your Dataset Situation

### Current Status

- **Data ends**: December 2025
- **Current date**: January 2026
- **All signals shown are HISTORICAL** (from the past)

### What This Means

#### ✅ **For Backtesting & Evaluation:**

Perfect! You can:

- See how the model performed on past data
- Evaluate win rate (25.6% with threshold 0.8)
- Understand which signals worked and which didn't
- Analyze patterns in successful trades

#### ❌ **For Live Trading:**

Not actionable because:

- Signals are from Dec 2025 (already in the past)
- You'd need **fresh daily data** to get live signals
- Example: If today is Jan 23, 2026, you need Jan 23, 2026 candle data

## How to Use This System Live

### Step 1: Get Fresh Data

```bash
# You would need to update your candles data to current date
# This depends on your data source
# Example: Download latest candles from your data provider
```

### Step 2: Run Data Preparation

```bash
python onchain_signal_system.py
# This processes the new data and creates features
```

### Step 3: Generate Fresh Signals

```bash
python generate_signals.py --mode latest
# This shows BUY signals for TODAY
```

### Step 4: Act on Signals

```
IF any pair shows 🟢 BUY (prob ≥ 0.8):
  1. Buy that pair at current/next available price
  2. Set stop-loss at -3%
  3. Set take-profit at +8%
  4. Monitor for 7 days
  5. Exit when either condition hits
```

## Signal Performance Metrics

### From Your Backtest Results:

**At Threshold 0.8 (Very Selective):**

- **Win Rate**: 25.6% (10 wins / 39 signals)
- **Signal Rate**: 0.18% (very rare)
- **Average per Trade**: +6.49% (from backtesting module)
- **Sharpe Ratio**: 4.11 (excellent risk-adjusted returns)

**Why only 25.6% win rate but 6.49% average return?**
Because the barrier-based labeling is conservative:

- ✅ **WIN** = Hit +8% BEFORE -3%
- ❌ **LOSS** = Everything else (even +5% gains count as loss if they didn't hit +8% first)

### Real Trading Returns (from backtest):

```
Total trades: 21
Total return: 136.33%
Avg return per trade: 6.49%
Win rate: 57.1%
Max drawdown: 13.67%
```

This shows the actual trading results were better (57.1% win rate) than the label win rate (25.6%).

## CSV File Columns Explained

### signals_history.csv

| Column          | Meaning                           |
| --------------- | --------------------------------- |
| `timestamp`     | When signal was generated         |
| `pair_id`       | Trading pair identifier           |
| `pred_proba`    | Model confidence (0-1)            |
| `signal`        | 1 = BUY, 0 = HOLD                 |
| `signal_label`  | Actual outcome: 1 = WIN, 0 = LOSS |
| `return_1d`     | Price change 1 day before signal  |
| `return_7d`     | Price change 7 days before signal |
| `volatility_7d` | Recent volatility (risk measure)  |

### signals_latest.csv

Same columns, but:

- `signal_label` is empty (unknown future outcome)
- Shows most recent prediction for each pair
- Use `pred_proba` to rank opportunities

## Summary

**Your signals are:**

1. ✅ **BUY signals** (not sell) for entering long positions
2. ✅ **Historical** backtesting results showing past performance
3. ✅ **Rule-based exit**: +8% take-profit, -3% stop-loss, max 7 days
4. ❌ **Not live** because data ends Dec 2025

**To make them live:**

- Need current/fresh data (Jan 2026 onwards)
- Run pipeline with new data
- Generate signals daily/weekly
- Execute trades when signal=1 appears

**Current use case:**

- Evaluate model quality
- Understand strategy mechanics
- See historical win rate
- Validate approach before going live
