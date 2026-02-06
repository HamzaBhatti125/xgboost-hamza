#!/usr/bin/env python3
"""
Quick Start: Market Regime Detector

This script provides a simple CLI to train, test, and monitor your regime detector.

Usage:
    python quickstart_regime_detector.py train    # Train new model
    python quickstart_regime_detector.py test     # Test on current cache
    python quickstart_regime_detector.py status   # Show model health
"""

import sys
import pickle
import polars as pl
from pathlib import Path
from market_regime_detector import MarketRegimeDetector
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_cache():
    """Load the live candles cache"""
    cache_path = Path("live_candles_cache.pkl")
    if not cache_path.exists():
        logger.error("❌ live_candles_cache.pkl not found")
        logger.error("   Run live_signal_generator.py first to generate cache")
        sys.exit(1)
    
    with open(cache_path, 'rb') as f:
        candles_history = pickle.load(f)
    
    logger.info(f"✅ Loaded {len(candles_history)} pairs from cache")
    return candles_history


def train_model(min_candles=10, sample_size=200):
    """
    Train a new regime detector model
    
    Args:
        min_candles: Minimum candles per pair to include in training
        sample_size: Number of pairs to use for training
    """
    logger.info("=" * 70)
    logger.info("🔧 TRAINING NEW REGIME DETECTOR")
    logger.info("=" * 70)
    
    # Load cache
    candles_history = load_cache()
    
    # Filter pairs with sufficient history
    valid_pairs = {
        addr: df for addr, df in candles_history.items()
        if len(df) >= min_candles
    }
    
    if not valid_pairs:
        logger.error(f"❌ No pairs with {min_candles}+ candles")
        logger.error(f"   Wait for more data to accumulate")
        logger.error(f"   Current max: {max(len(df) for df in candles_history.values())} candles")
        sys.exit(1)
    
    logger.info(f"📊 Found {len(valid_pairs)} pairs with {min_candles}+ candles")
    
    # Sample pairs if needed
    if len(valid_pairs) > sample_size:
        # Sort by candle count, take top N
        sorted_pairs = sorted(
            valid_pairs.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:sample_size]
        training_pairs = dict(sorted_pairs)
        logger.info(f"   Using top {sample_size} pairs by history")
    else:
        training_pairs = valid_pairs
        logger.info(f"   Using all {len(training_pairs)} valid pairs")
    
    # Combine training data
    combined_df = pl.concat(list(training_pairs.values()))
    logger.info(f"📈 Training dataset: {len(combined_df)} candles")
    
    # Train detector
    detector = MarketRegimeDetector(lookback_window=100)
    detector.fit(combined_df)
    
    # Save model
    detector.save("regime_detector.pkl")
    logger.info("\n✅ Training complete!")
    logger.info(f"   Model saved to: regime_detector.pkl")
    logger.info(f"   Crash State: State {detector.crash_state_idx}")
    
    return detector


def test_model():
    """Test the regime detector on current cache"""
    logger.info("=" * 70)
    logger.info("🧪 TESTING REGIME DETECTOR")
    logger.info("=" * 70)
    
    # Load model
    model_path = Path("regime_detector.pkl")
    if not model_path.exists():
        logger.error("❌ regime_detector.pkl not found")
        logger.error("   Run 'python quickstart_regime_detector.py train' first")
        sys.exit(1)
    
    detector = MarketRegimeDetector()
    detector.load("regime_detector.pkl")
    logger.info("✅ Model loaded")
    
    # Load cache
    candles_history = load_cache()
    
    # Test on pairs with sufficient history
    test_pairs = [
        (addr, df) for addr, df in candles_history.items()
        if len(df) >= 5
    ][:20]  # Test on 20 pairs
    
    if not test_pairs:
        logger.error("❌ No pairs with 5+ candles for testing")
        sys.exit(1)
    
    logger.info(f"🎯 Testing on {len(test_pairs)} pairs")
    logger.info("")
    
    # Run predictions
    results = {
        'emergency': 0,
        'crash_state': 0,
        'normal': 0,
        'position_scalars': []
    }
    
    for i, (addr, df) in enumerate(test_pairs, 1):
        try:
            prediction = detector.predict(df)
            
            position_scalar = prediction['position_scalar']
            is_emergency = prediction['is_emergency']
            crash_prob = prediction['crash_state_prob']
            regime = prediction['current_regime']
            
            results['position_scalars'].append(position_scalar)
            
            if is_emergency:
                status = "🚨 EMERGENCY"
                results['emergency'] += 1
            elif crash_prob > 0.7:
                status = "🔴 CRASH"
                results['crash_state'] += 1
            elif crash_prob > 0.4:
                status = "🟡 CAUTION"
            else:
                status = "🟢 NORMAL"
                results['normal'] += 1
            
            logger.info(f"{i:2d}. {addr[:10]}... | {status:15s} | Pos: {position_scalar:.2f} | Crash: {crash_prob:.0%} | Regime: {regime}")
        
        except Exception as e:
            logger.error(f"{i:2d}. {addr[:10]}... | ❌ ERROR: {e}")
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total Tested: {len(test_pairs)}")
    logger.info(f"  🟢 Normal: {results['normal']} ({100*results['normal']/len(test_pairs):.0f}%)")
    logger.info(f"  🔴 Crash State: {results['crash_state']} ({100*results['crash_state']/len(test_pairs):.0f}%)")
    logger.info(f"  🚨 Emergency: {results['emergency']} ({100*results['emergency']/len(test_pairs):.0f}%)")
    
    if results['position_scalars']:
        avg_scalar = sum(results['position_scalars']) / len(results['position_scalars'])
        logger.info(f"  📊 Avg Position Scalar: {avg_scalar:.2f}")
    
    logger.info("=" * 70)


def show_status():
    """Show model health and cache statistics"""
    logger.info("=" * 70)
    logger.info("📊 REGIME DETECTOR STATUS")
    logger.info("=" * 70)
    
    # Check model
    model_path = Path("regime_detector.pkl")
    if model_path.exists():
        from datetime import datetime
        mtime = model_path.stat().st_mtime
        model_age = datetime.now().timestamp() - mtime
        hours_old = model_age / 3600
        
        logger.info(f"✅ Model: regime_detector.pkl")
        logger.info(f"   Age: {hours_old:.1f} hours")
        
        if hours_old > 168:  # 7 days
            logger.warning(f"   ⚠️  Model is {hours_old/24:.0f} days old - consider retraining")
        
        # Load and inspect
        detector = MarketRegimeDetector()
        detector.load("regime_detector.pkl")
        logger.info(f"   Crash State: State {detector.crash_state_idx}")
        logger.info(f"   Lookback: {detector.lookback_window} candles")
    else:
        logger.warning("⚠️  No model found (regime_detector.pkl)")
        logger.warning("   Run 'python quickstart_regime_detector.py train'")
    
    logger.info("")
    
    # Check cache
    cache_path = Path("live_candles_cache.pkl")
    if cache_path.exists():
        candles_history = load_cache()
        
        candle_counts = [len(df) for df in candles_history.values()]
        
        logger.info(f"✅ Cache: live_candles_cache.pkl")
        logger.info(f"   Total Pairs: {len(candles_history)}")
        logger.info(f"   Candles: Min={min(candle_counts)}, Max={max(candle_counts)}, Avg={sum(candle_counts)/len(candle_counts):.1f}")
        logger.info(f"   Pairs with 5+ candles: {sum(1 for c in candle_counts if c >= 5)}")
        logger.info(f"   Pairs with 20+ candles: {sum(1 for c in candle_counts if c >= 20)}")
        
        if max(candle_counts) < 20:
            logger.warning(f"   ⚠️  Limited history - wait for 20+ candles per pair")
    else:
        logger.error("❌ No cache found (live_candles_cache.pkl)")
        logger.error("   Run live_signal_generator.py first")
    
    logger.info("=" * 70)


def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python quickstart_regime_detector.py train   # Train new model")
        print("  python quickstart_regime_detector.py test    # Test on current cache")
        print("  python quickstart_regime_detector.py status  # Show model health")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "train":
        train_model()
    elif command == "test":
        test_model()
    elif command == "status":
        show_status()
    else:
        logger.error(f"❌ Unknown command: {command}")
        logger.error("   Valid commands: train, test, status")
        sys.exit(1)


if __name__ == "__main__":
    main()
