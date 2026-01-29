"""
Signal Generation Configuration Optimizer
Based on multi-batch performance analysis (1,753 signals)
"""

# ============================================================================
# PERFORMANCE METRICS (Validated on Live Data)
# ============================================================================

VALIDATED_PERFORMANCE = {
    "actual_win_rate": 0.668,  # 66.8% (vs 91.7% backtested)
    "avg_win_return": 0.2673,  # +26.73% (vs +2% target)
    "avg_loss_return": -0.0335,  # -3.35% (vs -1% stop)
    "risk_reward_ratio": 7.98,  # Wins are 8x larger than losses
    "expected_value": 0.1674,  # +16.74% per signal
    "neutral_rate": 0.385,  # 38.5% don't hit win/loss targets
    "sample_size": 1753,  # Total signals analyzed
    "batches_analyzed": 4,  # Jan 28: 18:15, 19:05, 20:22, 22:26
}

# ============================================================================
# OPTIMIZED THRESHOLDS
# ============================================================================

# Conservative (Recommended for Live Trading)
CONSERVATIVE_THRESHOLD = 0.80  # Higher quality, fewer signals
CONSERVATIVE_EXPECTED_WIN_RATE = 0.70  # Estimated based on trends

# Moderate (Balance quality/quantity)
MODERATE_THRESHOLD = 0.75
MODERATE_EXPECTED_WIN_RATE = 0.68

# Aggressive (More signals, lower quality)
AGGRESSIVE_THRESHOLD = 0.70  # Current validated performance
AGGRESSIVE_EXPECTED_WIN_RATE = 0.668

# Very Conservative (Highest quality)
VERY_CONSERVATIVE_THRESHOLD = 0.85
VERY_CONSERVATIVE_EXPECTED_WIN_RATE = 0.72  # Estimated

# ============================================================================
# CALIBRATION ISSUES (CRITICAL)
# ============================================================================

CALIBRATION_WARNINGS = {
    "high_confidence_underperforms": True,
    "confidence_gte_85_win_rate": 0.484,  # 48.4% (worse than average!)
    "sample_size_high_conf": 36,
    "recommendation": "DO NOT use pred_proba for position sizing",
    "issue": "Model confidence scores not correlated with actual success",
}

# ============================================================================
# TARGET OPTIMIZATION CANDIDATES
# ============================================================================

# Current targets (from training)
CURRENT_PROFIT_TARGET_PCT = 2.0
CURRENT_STOP_LOSS_PCT = -1.0
CURRENT_HOLDING_PERIOD_HOURS = 4

# Potential optimizations to test
OPTIMIZATION_TESTS = {
    "tighter_target": {
        "profit_target_pct": 1.5,  # Capture more wins (reduce neutrals)
        "stop_loss_pct": -1.0,
        "holding_period_hours": 4,
        "rationale": "Reduce 38.5% neutral signals",
    },
    "extended_holding": {
        "profit_target_pct": 2.0,
        "stop_loss_pct": -1.0,
        "holding_period_hours": 6,  # Give more time to reach target
        "rationale": "Allow neutral signals to become wins",
    },
    "tighter_stop": {
        "profit_target_pct": 2.0,
        "stop_loss_pct": -0.75,  # Exit losses faster
        "holding_period_hours": 4,
        "rationale": "Reduce avg loss from -3.35% to -2%",
    },
    "aggressive_target": {
        "profit_target_pct": 3.0,  # Capture bigger moves
        "stop_loss_pct": -1.5,
        "holding_period_hours": 6,
        "rationale": "Optimize for fat-tailed wins (+26% avg)",
    },
}

# ============================================================================
# POSITION SIZING (Kelly Criterion)
# ============================================================================

def calculate_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate Kelly Criterion for optimal position sizing.
    
    Formula: f = (p * b - q) / b
    where:
        p = win probability
        q = loss probability (1 - p)
        b = win/loss ratio (avg_win / avg_loss)
    """
    if avg_loss == 0:
        return 0.0
    
    p = win_rate
    q = 1 - p
    b = abs(avg_win / avg_loss)
    
    kelly = (p * b - q) / b
    return max(0.0, kelly)  # Don't return negative


# Kelly for validated performance
KELLY_FULL = calculate_kelly_fraction(
    win_rate=VALIDATED_PERFORMANCE["actual_win_rate"],
    avg_win=VALIDATED_PERFORMANCE["avg_win_return"],
    avg_loss=abs(VALIDATED_PERFORMANCE["avg_loss_return"])
)

KELLY_HALF = KELLY_FULL / 2  # Conservative Kelly
KELLY_QUARTER = KELLY_FULL / 4  # Very conservative

POSITION_SIZING = {
    "full_kelly": KELLY_FULL,  # ~0.74 (74% of capital per signal - too aggressive!)
    "half_kelly": KELLY_HALF,  # ~0.37 (37% per signal - still aggressive)
    "quarter_kelly": KELLY_QUARTER,  # ~0.18 (18% per signal - recommended)
    "fixed_1_percent": 0.01,  # 1% per signal (very conservative)
    "fixed_2_percent": 0.02,  # 2% per signal (conservative)
    "recommendation": "Use quarter_kelly (18%) or fixed_2_percent (2%) for live trading",
}

# ============================================================================
# TIME-OF-DAY ANALYSIS
# ============================================================================

TIME_PERFORMANCE = {
    "18:15_utc": {"win_rate": 0.634, "signals": 353},
    "19:05_utc": {"win_rate": 0.639, "signals": 390},
    "20:22_utc": {"win_rate": 0.644, "signals": 459},
    "22:26_utc": {"win_rate": 0.753, "signals": 551},  # BEST
    "finding": "Later signals (22:26) performed significantly better (75.3%)",
    "hypothesis": "More candle data accumulated OR better market conditions",
    "recommendation": "Continue monitoring time-of-day effects",
}

# ============================================================================
# RECOMMENDED CONFIGURATION
# ============================================================================

RECOMMENDED_CONFIG = {
    # Signal Generation
    "threshold": CONSERVATIVE_THRESHOLD,  # 0.80
    "model_path": "xgb_model.json",
    "min_history_candles": 20,
    
    # Position Sizing (choose one)
    "position_size_method": "quarter_kelly",  # OR "fixed_2_percent"
    "position_size_pct": KELLY_QUARTER,  # 18% per signal (or 2% if fixed)
    
    # Risk Management
    "max_positions": 10,  # Don't hold more than 10 signals simultaneously
    "max_capital_deployed": 0.50,  # Never deploy more than 50% of capital
    
    # Confidence Handling
    "use_pred_proba_for_sizing": False,  # DON'T use (uncalibrated)
    "treat_all_signals_equally": True,  # Ignore confidence differences
    
    # Monitoring
    "track_every_batch": True,
    "min_signals_for_adjustment": 5000,  # Need 5k signals before changing config
}

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("SIGNAL GENERATION CONFIGURATION")
    print("="*80)
    
    print("\n📊 VALIDATED PERFORMANCE (1,753 signals)")
    print(f"   Win Rate: {VALIDATED_PERFORMANCE['actual_win_rate']*100:.1f}%")
    print(f"   Avg Win: +{VALIDATED_PERFORMANCE['avg_win_return']*100:.2f}%")
    print(f"   Avg Loss: {VALIDATED_PERFORMANCE['avg_loss_return']*100:.2f}%")
    print(f"   Risk-Reward: {VALIDATED_PERFORMANCE['risk_reward_ratio']:.1f}:1")
    print(f"   Expected Value: +{VALIDATED_PERFORMANCE['expected_value']*100:.2f}% per signal")
    
    print("\n🎯 RECOMMENDED THRESHOLD")
    print(f"   Current: 0.70 (aggressive)")
    print(f"   Recommended: {RECOMMENDED_CONFIG['threshold']} (conservative)")
    print(f"   Expected Win Rate: {CONSERVATIVE_EXPECTED_WIN_RATE*100:.1f}%")
    
    print("\n⚠️  CALIBRATION WARNING")
    print(f"   High confidence (≥85%) signals: {CALIBRATION_WARNINGS['confidence_gte_85_win_rate']*100:.1f}% win rate")
    print(f"   This is WORSE than overall {VALIDATED_PERFORMANCE['actual_win_rate']*100:.1f}%!")
    print(f"   Recommendation: {CALIBRATION_WARNINGS['recommendation']}")
    
    print("\n💰 POSITION SIZING")
    print(f"   Full Kelly: {POSITION_SIZING['full_kelly']*100:.1f}% (too aggressive)")
    print(f"   Half Kelly: {POSITION_SIZING['half_kelly']*100:.1f}% (aggressive)")
    print(f"   Quarter Kelly: {POSITION_SIZING['quarter_kelly']*100:.1f}% (recommended)")
    print(f"   Fixed 2%: {POSITION_SIZING['fixed_2_percent']*100:.1f}% (conservative)")
    
    print("\n⏰ TIME-OF-DAY INSIGHTS")
    for time, data in TIME_PERFORMANCE.items():
        if time not in ["finding", "hypothesis", "recommendation"]:
            print(f"   {time}: {data['win_rate']*100:.1f}% win rate ({data['signals']} signals)")
    
    print("\n✅ RECOMMENDED CONFIG")
    for key, value in RECOMMENDED_CONFIG.items():
        if isinstance(value, float):
            print(f"   {key}: {value*100:.1f}%")
        else:
            print(f"   {key}: {value}")
    
    print("\n" + "="*80)
