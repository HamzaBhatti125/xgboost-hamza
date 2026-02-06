# 🚀 Regime Detector - Quick Reference

## Commands

```bash
# Check status
python quickstart_regime_detector.py status

# Train model (requires 5+ candles per pair)
python quickstart_regime_detector.py train

# Test on current cache
python quickstart_regime_detector.py test
```

## Integration (3 lines of code)

```python
from regime_aware_signals import RegimeAwareSignalGenerator

# At startup
regime_filter = RegimeAwareSignalGenerator()

# After XGBoost generates signals
risk_adjusted = regime_filter.apply_regime_filter(
    signals_df=xgb_signals,
    candles_history=candles_history,
    min_confidence=0.5
)
```

## Files

| File | Purpose |
|------|---------|
| `market_regime_detector.py` | Core HMM + Isolation Forest module |
| `regime_aware_signals.py` | Production wrapper class |
| `quickstart_regime_detector.py` | CLI tool (train/test/status) |
| `REGIME_DETECTOR_GUIDE.md` | Complete documentation |
| `REGIME_DETECTOR_SUMMARY.md` | Implementation summary |

## Key Outputs

### `predict()` returns:
```python
{
    'position_scalar': 0.6,      # 0.0 to 1.0 (multiply with confidence)
    'is_emergency': False,       # True = Block ALL trades
    'current_regime': 1,         # 0, 1, or 2
    'regime_probs': [0.2, 0.7, 0.1],
    'crash_state_prob': 0.4      # P(crash state)
}
```

### Usage:
```python
if prediction['is_emergency']:
    continue  # Block trade

adjusted_confidence = base_confidence * prediction['position_scalar']

if adjusted_confidence < 0.5:
    continue  # Below threshold
```

## Features Generated

1. **Log Returns**: Price momentum
2. **Parkinson Volatility**: Intrabar risk
3. **Amihud Illiquidity**: Market fragility ⭐
4. **Liquidity Shock**: Rug pull detection ⭐
5. **VWAP Deviation**: Price vs fair value ⭐

## Expected Behavior

| Scenario | Position Scalar | Action |
|----------|----------------|--------|
| Normal market | 0.8 - 1.0 | ✅ Trade at 80-100% |
| High volatility | 0.4 - 0.7 | ⚠️ Trade at 40-70% |
| Crash detected | 0.0 - 0.3 | 🚫 Block or minimal |
| Anomaly (emergency) | Any | 🚨 Block ALL |

## Current Status (Limited Data)

- **Cache**: 3416 pairs, 1-16 candles (avg 2.4)
- **Model**: Trained on 100 pairs with 5+ candles
- **Behavior**: 100% crash state (expected with sparse data)
- **Position Scalars**: 0.0 (protective mode)

**This is correct behavior!** The model doesn't have enough data yet.

## Timeline

| Time | Expected Behavior |
|------|-------------------|
| **Now** | Protective mode (blocks trades) |
| **4 hours** | Some pairs reach 10+ candles, 30-50% position scalars |
| **24 hours** | Most pairs reach 20+ candles, 60-80% position scalars |
| **7 days** | Full history, 70-90% position scalars in normal markets |

## Retraining

```bash
# Retrain weekly or after major events
python quickstart_regime_detector.py train
```

Target metrics after 7 days:
- Emergency rate: 0.5% - 2%
- Avg position scalar: 0.7 - 0.9
- Filter rate: 10% - 30%

## Troubleshooting

### "No pairs with N+ candles"
→ Wait for live system to accumulate data

### "100% crash state probability"
→ Expected with <20 candles per pair, wait 24-48 hours

### "Model not converging"
→ Increase `n_iter` in `market_regime_detector.py` from 100 to 200

### Too many emergency stops (>5%)
→ Decrease `contamination` from 0.01 to 0.005

---

**Quick Start**: Run `python quickstart_regime_detector.py status` to check health

**Full Docs**: See `REGIME_DETECTOR_GUIDE.md` for complete documentation

**Version**: 1.0 | **Date**: February 5, 2026
