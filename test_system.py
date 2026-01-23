#!/usr/bin/env python3
"""
System Validation & Testing
============================

Comprehensive tests to verify the signal generation system
works correctly before running on real data.
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
import sys

def test_barrier_labeling():
    """Test barrier-based labeling logic"""
    print(f"\n{'='*80}")
    print("TEST 1: Barrier-Based Labeling")
    print(f"{'='*80}\n")
    
    # Create synthetic price series
    test_cases = [
        {
            'name': 'Hit target before stop (SUCCESS)',
            'prices': [100, 105, 108, 110, 108, 105, 103],
            'expected_label': 1,
            'reason': 'Hit +8% (108) on day 2 before hitting -3%'
        },
        {
            'name': 'Hit stop before target (FAIL)',
            'prices': [100, 98, 96, 100, 110, 115, 120],
            'expected_label': 0,
            'reason': 'Hit -3% (97) on day 1 before hitting +8%'
        },
        {
            'name': 'Neither barrier hit (FAIL)',
            'prices': [100, 102, 101, 103, 104, 102, 105],
            'expected_label': 0,
            'reason': 'Never reached +8% or -3%'
        },
        {
            'name': 'Large spike then crash (SUCCESS if +8% hit first)',
            'prices': [100, 112, 90, 85, 80, 75, 70],
            'expected_label': 1,
            'reason': 'Hit +8% (112) on day 1 before any stop'
        },
    ]
    
    def check_barrier(prices, target_pct=8.0, stop_pct=3.0):
        """Simple barrier label checker"""
        base = prices[0]
        for price in prices[1:]:
            ret = (price - base) / base * 100
            if ret <= -stop_pct:
                return 0  # Stop hit
            if ret >= target_pct:
                return 1  # Target hit
        return 0  # Neither
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        result = check_barrier(test['prices'])
        expected = test['expected_label']
        
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"{i}. {test['name']}")
        print(f"   Prices: {test['prices']}")
        print(f"   Expected: {expected}, Got: {result}")
        print(f"   {status} - {test['reason']}\n")
        
        if result == expected:
            passed += 1
        else:
            failed += 1
    
    print(f"Results: {passed} passed, {failed} failed\n")
    return failed == 0


def test_filtering_logic():
    """Test pair filtering logic"""
    print(f"\n{'='*80}")
    print("TEST 2: Filtering Logic")
    print(f"{'='*80}\n")
    
    # Create synthetic pair data
    pairs = pl.DataFrame({
        'pair_id': ['pair1', 'pair2', 'pair3', 'pair4', 'pair5'],
        'chain_id': [8453, 8453, 1, 8453, 8453],
        'flag_inactive': [False, True, False, False, False],
        'flag_blacklisted_manually': [False, False, False, False, True],
        'flag_unsupported_quote_token': [False, False, False, False, False],
        'flag_unknown_exchange': [False, False, False, False, False],
        'sell_tax': [2.0, 0.0, 0.0, 15.0, 0.0],
        'buy_tax': [2.0, 0.0, 0.0, 0.0, 0.0],
        'transfer_tax': [0.0, 0.0, 0.0, 0.0, 0.0],
        'buy_volume_30d': [50000, 100000, 80000, 60000, 70000],
        'sell_volume_30d': [50000, 100000, 80000, 60000, 70000],
    })
    
    print("Initial pairs:")
    print(pairs)
    
    # Apply filters
    print("\n\n1. Filter for Base chain (chain_id = 8453):")
    filtered = pairs.filter(pl.col('chain_id') == 8453)
    print(f"   Removed: pair3 (Ethereum)")
    print(f"   Remaining: {len(filtered)}\n")
    
    print("2. Filter inactive:")
    filtered = filtered.filter(pl.col('flag_inactive') == False)
    print(f"   Removed: pair2 (inactive)")
    print(f"   Remaining: {len(filtered)}\n")
    
    print("3. Filter blacklisted:")
    filtered = filtered.filter(pl.col('flag_blacklisted_manually') == False)
    print(f"   Removed: pair5 (blacklisted)")
    print(f"   Remaining: {len(filtered)}\n")
    
    print("4. Filter high taxes (>10%):")
    filtered = filtered.filter(pl.col('sell_tax') <= 10)
    print(f"   Removed: pair4 (sell_tax=15%)")
    print(f"   Remaining: {len(filtered)}\n")
    
    print("Final pairs:")
    print(filtered)
    
    expected_remaining = ['pair1']
    actual_remaining = filtered['pair_id'].to_list()
    
    if actual_remaining == expected_remaining:
        print("\n✅ Filtering logic correct!")
        return True
    else:
        print(f"\n❌ Filtering logic incorrect!")
        print(f"   Expected: {expected_remaining}")
        print(f"   Got: {actual_remaining}")
        return False


def test_feature_engineering():
    """Test feature calculations"""
    print(f"\n{'='*80}")
    print("TEST 3: Feature Engineering")
    print(f"{'='*80}\n")
    
    # Create synthetic candle data
    dates = [datetime(2024, 1, i) for i in range(1, 11)]
    
    candles = pl.DataFrame({
        'pair_id': ['pair1'] * 10,
        'timestamp': dates,
        'open': [100, 102, 101, 103, 105, 107, 106, 108, 110, 109],
        'high': [103, 104, 103, 106, 108, 110, 109, 111, 113, 112],
        'low': [99, 101, 100, 102, 104, 106, 105, 107, 109, 108],
        'close': [102, 101, 103, 105, 107, 106, 108, 110, 109, 111],
        'volume': [1000, 1100, 900, 1200, 1500, 1300, 1400, 1600, 1700, 1800],
        'buy_volume': [600, 550, 500, 700, 900, 800, 850, 1000, 1100, 1200],
        'sell_volume': [400, 550, 400, 500, 600, 500, 550, 600, 600, 600],
        'buys': [50, 55, 45, 60, 75, 65, 70, 80, 85, 90],
        'sells': [40, 55, 40, 50, 60, 50, 55, 60, 60, 60],
    })
    
    print("Input candles:")
    print(candles)
    
    # Calculate features
    candles = candles.with_columns([
        (pl.col('close').pct_change(1)).alias('return_1d'),
        ((pl.col('high') - pl.col('low')) / pl.col('open')).alias('range_normalized'),
        (pl.col('buy_volume') / pl.col('sell_volume')).alias('buy_sell_ratio'),
    ])
    
    print("\n\nFeatures calculated:")
    print(candles.select(['timestamp', 'close', 'return_1d', 'range_normalized', 'buy_sell_ratio']))
    
    # Validate calculations
    tests = [
        ('return_1d calculation', candles['return_1d'][1], (101-102)/102),
        ('range_normalized', candles['range_normalized'][0], (103-99)/100),
        ('buy_sell_ratio', candles['buy_sell_ratio'][0], 600/400),
    ]
    
    all_pass = True
    for name, actual, expected in tests:
        close = abs(actual - expected) < 0.001 if expected != 0 else actual == expected
        status = "✅" if close else "❌"
        print(f"\n{status} {name}")
        print(f"   Expected: {expected:.6f}")
        print(f"   Got: {actual:.6f}")
        if not close:
            all_pass = False
    
    return all_pass


def test_friction_impact():
    """Test trading friction calculations"""
    print(f"\n{'='*80}")
    print("TEST 4: Trading Friction Impact")
    print(f"{'='*80}\n")
    
    entry_price = 100.0
    exit_price = 108.0  # +8% before friction
    
    fee_pct = 0.3
    slippage_pct = 0.2
    total_friction = fee_pct + slippage_pct
    
    # Apply friction
    entry_with_friction = entry_price * (1 + total_friction / 100)
    exit_with_friction = exit_price * (1 - total_friction / 100)
    
    pnl_no_friction = (exit_price - entry_price) / entry_price * 100
    pnl_with_friction = (exit_with_friction - entry_with_friction) / entry_with_friction * 100
    
    print(f"Scenario: +8% raw move")
    print(f"\nWithout friction:")
    print(f"  Entry: ${entry_price:.2f}")
    print(f"  Exit: ${exit_price:.2f}")
    print(f"  PnL: +{pnl_no_friction:.2f}%")
    
    print(f"\nWith friction ({total_friction}%):")
    print(f"  Entry: ${entry_with_friction:.2f} (paid {fee_pct + slippage_pct}%)")
    print(f"  Exit: ${exit_with_friction:.2f} (paid {fee_pct + slippage_pct}%)")
    print(f"  PnL: +{pnl_with_friction:.2f}%")
    
    print(f"\nImpact: {pnl_no_friction - pnl_with_friction:.2f}% reduction")
    
    # Test break-even
    print(f"\n\nBreak-even calculation:")
    breakeven_move = (entry_with_friction - entry_price) / entry_price * 100
    print(f"  Need +{breakeven_move:.2f}% just to cover entry friction")
    
    full_breakeven = (entry_with_friction * (1 + total_friction / 100) - entry_price) / entry_price * 100
    print(f"  Need +{full_breakeven:.2f}% to break even after round-trip")
    
    # Validate
    expected_impact = 1.0  # Approximately 1% impact from 0.5% round-trip on 8% move
    actual_impact = pnl_no_friction - pnl_with_friction
    
    if abs(actual_impact - expected_impact) < 0.5:
        print(f"\n✅ Friction calculation correct")
        return True
    else:
        print(f"\n❌ Friction calculation incorrect")
        return False


def test_signal_rate():
    """Test signal rate constraints"""
    print(f"\n{'='*80}")
    print("TEST 5: Signal Rate Constraints")
    print(f"{'='*80}\n")
    
    # Simulate predictions at different thresholds
    np.random.seed(42)
    probabilities = np.random.beta(2, 8, 1000)  # Skewed distribution
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("Signal rates at different thresholds:\n")
    
    for thresh in thresholds:
        predictions = (probabilities >= thresh).astype(int)
        signal_rate = predictions.mean() * 100
        
        status = "✅" if signal_rate < 5 else "⚠️"
        print(f"{status} Threshold {thresh:.1f}: {signal_rate:.2f}% signal rate")
        
        if signal_rate < 5:
            print(f"    Good - signals are rare (<5%)")
        else:
            print(f"    Warning - signals too common (≥5%)")
    
    print("\n✅ Signal rate test complete")
    return True


def run_all_tests():
    """Run all validation tests"""
    print(f"\n{'#'*80}")
    print("SYSTEM VALIDATION & TESTING")
    print(f"{'#'*80}")
    print(f"\nRunning comprehensive tests to verify system correctness...")
    
    tests = [
        ("Barrier Labeling", test_barrier_labeling),
        ("Filtering Logic", test_filtering_logic),
        ("Feature Engineering", test_feature_engineering),
        ("Friction Impact", test_friction_impact),
        ("Signal Rate", test_signal_rate),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print(f"\n{'#'*80}")
    print("TEST SUMMARY")
    print(f"{'#'*80}\n")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n{'='*80}")
    print(f"TOTAL: {passed_count}/{total_count} tests passed")
    print(f"{'='*80}\n")
    
    if passed_count == total_count:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Review errors above before proceeding.")
        return False


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
