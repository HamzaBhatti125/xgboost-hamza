#!/usr/bin/env python3
"""
Real-Time Market Dump Analysis

This script analyzes the current market conditions during the dump
and shows how the regime detector is responding.
"""

import polars as pl
import pickle
from datetime import datetime, timedelta
from market_regime_detector import MarketRegimeDetector
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def analyze_market_dump():
    """Analyze current market conditions during the dump"""
    
    logger.info("=" * 80)
    logger.info("🚨 REAL-TIME MARKET DUMP ANALYSIS")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Load cache
    with open("live_candles_cache.pkl", 'rb') as f:
        candles_history = pickle.load(f)
    
    logger.info(f"\n📊 Cache Status: {len(candles_history)} pairs loaded\n")
    
    # Load detector
    detector = MarketRegimeDetector()
    detector.load("regime_detector.pkl")
    
    # Analyze pairs with sufficient history
    pairs_to_analyze = [
        (addr, df) for addr, df in candles_history.items()
        if len(df) >= 5
    ]
    
    # Sort by most candles
    pairs_to_analyze.sort(key=lambda x: len(x[1]), reverse=True)
    
    logger.info(f"🎯 Analyzing {min(30, len(pairs_to_analyze))} pairs with most history\n")
    
    # Collect statistics
    stats = {
        'total': 0,
        'crash_state': 0,
        'emergency': 0,
        'high_risk': 0,
        'medium_risk': 0,
        'low_risk': 0,
        'position_scalars': [],
        'crash_probs': []
    }
    
    detailed_results = []
    
    for addr, df in pairs_to_analyze[:30]:
        try:
            prediction = detector.predict(df)
            
            stats['total'] += 1
            stats['position_scalars'].append(prediction['position_scalar'])
            stats['crash_probs'].append(prediction['crash_state_prob'])
            
            # Get latest candle data
            latest = df.tail(1)
            close = latest['close'][0]
            volume = latest['volume_token1'][0]
            liquidity = latest['avg_liquidity'][0] if 'avg_liquidity' in latest.columns else 0
            
            # Categorize risk
            crash_prob = prediction['crash_state_prob']
            is_emergency = prediction['is_emergency']
            
            if is_emergency:
                risk_level = "🚨 EMERGENCY"
                stats['emergency'] += 1
            elif crash_prob >= 0.7:
                risk_level = "🔴 CRASH"
                stats['crash_state'] += 1
            elif crash_prob >= 0.4:
                risk_level = "🟡 HIGH"
                stats['high_risk'] += 1
            elif crash_prob >= 0.2:
                risk_level = "🟠 MEDIUM"
                stats['medium_risk'] += 1
            else:
                risk_level = "🟢 LOW"
                stats['low_risk'] += 1
            
            detailed_results.append({
                'address': addr[:10] + '...',
                'candles': len(df),
                'risk_level': risk_level,
                'crash_prob': f"{crash_prob:.0%}",
                'position_scalar': f"{prediction['position_scalar']:.2f}",
                'regime': prediction['current_regime'],
                'close': f"{close:.2e}",
                'volume': f"{volume:.2e}",
                'liquidity': f"{liquidity:.0f}"
            })
        
        except Exception as e:
            logger.error(f"Error analyzing {addr[:10]}...: {e}")
    
    # Display detailed results
    logger.info("┌" + "─" * 78 + "┐")
    logger.info("│" + " " * 25 + "DETAILED ANALYSIS" + " " * 36 + "│")
    logger.info("├" + "─" * 78 + "┤")
    logger.info(f"│ {'Pair':<12} │ {'Candles':<7} │ {'Risk':<13} │ {'Crash':<7} │ {'Pos':<5} │ {'Regime':<6} │")
    logger.info("├" + "─" * 78 + "┤")
    
    for result in detailed_results[:15]:
        logger.info(
            f"│ {result['address']:<12} │ {result['candles']:<7} │ "
            f"{result['risk_level']:<13} │ {result['crash_prob']:<7} │ "
            f"{result['position_scalar']:<5} │ {result['regime']:<6} │"
        )
    
    logger.info("└" + "─" * 78 + "┘")
    
    # Overall statistics
    logger.info("\n" + "=" * 80)
    logger.info("📊 MARKET RISK SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Pairs Analyzed: {stats['total']}")
    logger.info("")
    logger.info("Risk Distribution:")
    logger.info(f"  🚨 EMERGENCY:  {stats['emergency']:3d} ({100*stats['emergency']/stats['total']:5.1f}%)")
    logger.info(f"  🔴 CRASH:      {stats['crash_state']:3d} ({100*stats['crash_state']/stats['total']:5.1f}%)")
    logger.info(f"  🟡 HIGH RISK:  {stats['high_risk']:3d} ({100*stats['high_risk']/stats['total']:5.1f}%)")
    logger.info(f"  🟠 MEDIUM:     {stats['medium_risk']:3d} ({100*stats['medium_risk']/stats['total']:5.1f}%)")
    logger.info(f"  🟢 LOW RISK:   {stats['low_risk']:3d} ({100*stats['low_risk']/stats['total']:5.1f}%)")
    logger.info("")
    
    if stats['position_scalars']:
        avg_scalar = sum(stats['position_scalars']) / len(stats['position_scalars'])
        avg_crash = sum(stats['crash_probs']) / len(stats['crash_probs'])
        
        logger.info("Market Metrics:")
        logger.info(f"  Avg Position Scalar: {avg_scalar:.2f} (0.0 = Block all, 1.0 = Full size)")
        logger.info(f"  Avg Crash Probability: {avg_crash:.1%}")
        logger.info(f"  Min Position Scalar: {min(stats['position_scalars']):.2f}")
        logger.info(f"  Max Position Scalar: {max(stats['position_scalars']):.2f}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("💡 RECOMMENDATION")
    logger.info("=" * 80)
    
    if stats['emergency'] > 0:
        logger.info("🚨 CRITICAL: Anomalies detected - BLOCK ALL TRADING")
    elif stats['crash_state'] >= stats['total'] * 0.8:
        logger.info("🔴 SEVERE: 80%+ crash state - EMERGENCY EXIT MODE")
    elif stats['crash_state'] >= stats['total'] * 0.5:
        logger.info("🟡 HIGH ALERT: 50%+ crash state - REDUCE ALL POSITIONS")
    elif stats['crash_state'] >= stats['total'] * 0.2:
        logger.info("🟠 CAUTION: 20%+ crash state - TRADE WITH REDUCED SIZE")
    else:
        logger.info("🟢 NORMAL: Market conditions acceptable for trading")
    
    logger.info("=" * 80)
    
    # Show recent price action on top pairs
    logger.info("\n" + "=" * 80)
    logger.info("📈 RECENT PRICE ACTION (Top 5 Pairs)")
    logger.info("=" * 80)
    
    for i, (addr, df) in enumerate(pairs_to_analyze[:5], 1):
        if len(df) >= 3:
            recent = df.tail(3)
            closes = recent['close'].to_list()
            times = recent['timestamp'].to_list()
            
            logger.info(f"\n{i}. {addr[:16]}...")
            logger.info(f"   Candles: {len(df)}")
            
            if len(closes) >= 2:
                pct_change = ((closes[-1] - closes[0]) / closes[0]) * 100
                logger.info(f"   Recent Change: {pct_change:+.2f}%")
                
                for j, (t, c) in enumerate(zip(times[-3:], closes[-3:])):
                    logger.info(f"   [{j+1}] {t}: {c:.2e}")
    
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_market_dump()
