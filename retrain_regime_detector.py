#!/usr/bin/env python3
"""
Retrain Market Regime Detector with New Data

Usage:
    python retrain_regime_detector.py --data-source cache --validate
    python retrain_regime_detector.py --historical historical_candles.csv
    
Best Practice:
    - Retrain quarterly or after major market events
    - Include at least 6 months of new data
    - Validate on out-of-sample recent data
    - Compare old vs new model performance before deployment
"""

import argparse
import polars as pl
from pathlib import Path
from market_regime_detector import MarketRegimeDetector
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_training_data(source: str) -> dict:
    """Load candle data from cache or file"""
    
    if source == "cache":
        logger.info("📂 Loading data from candles_cache.parquet...")
        
        cache_path = Path("candles_cache.parquet")
        if not cache_path.exists():
            raise FileNotFoundError("Cache not found. Run live system first.")
        
        df = pl.read_parquet(cache_path)
        
        # Group by pair
        pair_data = {}
        for addr in df['pair_address'].unique():
            pair_df = df.filter(pl.col('pair_address') == addr).sort('timestamp')
            pair_data[addr] = pair_df
        
        logger.info(f"✅ Loaded {len(pair_data)} pairs from cache")
        return pair_data
    
    else:
        logger.info(f"📂 Loading data from {source}...")
        df = pl.read_csv(source)
        
        # Group by pair
        pair_data = {}
        for addr in df['pair_address'].unique():
            pair_df = df.filter(pl.col('pair_address') == addr).sort('timestamp')
            pair_data[addr] = pair_df
        
        logger.info(f"✅ Loaded {len(pair_data)} pairs from file")
        return pair_data


def validate_model(detector: MarketRegimeDetector, validation_data: dict) -> dict:
    """Validate model on recent data"""
    
    logger.info("\n🧪 VALIDATING MODEL ON RECENT DATA")
    logger.info("="*80)
    
    results = {
        'total': 0,
        'crash_detected': 0,
        'normal_detected': 0,
        'scalars': []
    }
    
    for addr, df in list(validation_data.items())[:30]:  # Test on 30 pairs
        if len(df) < 5:
            continue
        
        prediction = detector.predict(df)
        results['total'] += 1
        results['scalars'].append(prediction['position_scalar'])
        
        if prediction['crash_state_prob'] > 0.7:
            results['crash_detected'] += 1
            status = "🔴 CRASH"
        else:
            results['normal_detected'] += 1
            status = "🟢 NORMAL"
        
        logger.info(f"   {addr[:10]}... | {status} | Scalar: {prediction['position_scalar']:.2f}")
    
    # Summary
    avg_scalar = sum(results['scalars']) / len(results['scalars']) if results['scalars'] else 0
    crash_pct = (results['crash_detected'] / results['total'] * 100) if results['total'] > 0 else 0
    
    logger.info("\n📊 VALIDATION SUMMARY:")
    logger.info(f"   Total Pairs: {results['total']}")
    logger.info(f"   Crash Detected: {results['crash_detected']} ({crash_pct:.1f}%)")
    logger.info(f"   Normal Detected: {results['normal_detected']} ({100-crash_pct:.1f}%)")
    logger.info(f"   Avg Position Scalar: {avg_scalar:.2f}")
    
    return results


def retrain_model(data_source: str, validate: bool = True, output: str = None):
    """Retrain the regime detector"""
    
    logger.info("="*80)
    logger.info("🔄 RETRAINING MARKET REGIME DETECTOR")
    logger.info("="*80)
    
    # Load data
    pair_data = load_training_data(data_source)
    
    # Filter pairs with sufficient history
    valid_pairs = {addr: df for addr, df in pair_data.items() if len(df) >= 20}
    logger.info(f"✅ Found {len(valid_pairs)} pairs with 20+ candles")
    
    if len(valid_pairs) < 100:
        logger.warning(f"⚠️  Only {len(valid_pairs)} pairs available. Recommend 100+ for robust training.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Retraining cancelled.")
            return
    
    # Split into train/validation (80/20)
    train_size = int(len(valid_pairs) * 0.8)
    train_data = dict(list(valid_pairs.items())[:train_size])
    val_data = dict(list(valid_pairs.items())[train_size:])
    
    logger.info(f"\n📊 Data Split:")
    logger.info(f"   Training: {len(train_data)} pairs")
    logger.info(f"   Validation: {len(val_data)} pairs")
    
    # Backup old model
    old_model_path = Path("regime_detector.pkl")
    if old_model_path.exists():
        backup_path = f"regime_detector_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        import shutil
        shutil.copy(old_model_path, backup_path)
        logger.info(f"💾 Backed up old model to {backup_path}")
    
    # Train new model
    logger.info("\n🎓 Training new model...")
    detector = MarketRegimeDetector(
        n_states=3,
        lookback_window=500,
        crash_threshold=0.7,
        anomaly_contamination=0.05
    )
    
    detector.train(train_data)
    
    # Validate if requested
    if validate and len(val_data) > 0:
        val_results = validate_model(detector, val_data)
    
    # Save model
    output_path = output or "regime_detector.pkl"
    detector.save(output_path)
    logger.info(f"\n💾 Saved new model to {output_path}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ RETRAINING COMPLETE")
    logger.info("="*80)
    logger.info("\n📋 Next Steps:")
    logger.info("   1. Test the new model with: python quickstart_regime_detector.py test")
    logger.info("   2. Compare with old model performance")
    logger.info("   3. If satisfied, restart live system")
    logger.info("   4. If not satisfied, restore backup and investigate")


def compare_models(old_model_path: str, new_model_path: str, test_data_source: str):
    """Compare old vs new model performance"""
    
    logger.info("="*80)
    logger.info("⚖️  COMPARING MODEL PERFORMANCE")
    logger.info("="*80)
    
    # Load test data
    test_data = load_training_data(test_data_source)
    test_pairs = {addr: df for addr, df in list(test_data.items())[:50] if len(df) >= 20}
    
    # Load models
    logger.info(f"\n📂 Loading models...")
    old_detector = MarketRegimeDetector()
    old_detector.load(old_model_path)
    
    new_detector = MarketRegimeDetector()
    new_detector.load(new_model_path)
    
    # Compare predictions
    logger.info(f"\n🧪 Testing on {len(test_pairs)} pairs...")
    
    old_crash = 0
    new_crash = 0
    agreement = 0
    old_scalars = []
    new_scalars = []
    
    for addr, df in test_pairs.items():
        old_pred = old_detector.predict(df)
        new_pred = new_detector.predict(df)
        
        old_is_crash = old_pred['crash_state_prob'] > 0.7
        new_is_crash = new_pred['crash_state_prob'] > 0.7
        
        if old_is_crash:
            old_crash += 1
        if new_is_crash:
            new_crash += 1
        if old_is_crash == new_is_crash:
            agreement += 1
        
        old_scalars.append(old_pred['position_scalar'])
        new_scalars.append(new_pred['position_scalar'])
    
    # Results
    logger.info("\n📊 COMPARISON RESULTS:")
    logger.info(f"   Old Model Crash Detection: {old_crash}/{len(test_pairs)} ({old_crash/len(test_pairs)*100:.1f}%)")
    logger.info(f"   New Model Crash Detection: {new_crash}/{len(test_pairs)} ({new_crash/len(test_pairs)*100:.1f}%)")
    logger.info(f"   Agreement Rate: {agreement}/{len(test_pairs)} ({agreement/len(test_pairs)*100:.1f}%)")
    logger.info(f"   Old Avg Scalar: {sum(old_scalars)/len(old_scalars):.2f}")
    logger.info(f"   New Avg Scalar: {sum(new_scalars)/len(new_scalars):.2f}")
    
    logger.info("\n💡 Interpretation:")
    if agreement / len(test_pairs) > 0.8:
        logger.info("   ✅ High agreement - models are consistent")
    else:
        logger.info("   ⚠️  Low agreement - significant difference in behavior")
    
    if abs(sum(old_scalars)/len(old_scalars) - sum(new_scalars)/len(new_scalars)) < 0.1:
        logger.info("   ✅ Similar risk levels - safe to deploy")
    else:
        logger.info("   ⚠️  Different risk levels - review carefully before deployment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain Market Regime Detector")
    parser.add_argument(
        "--data-source",
        default="cache",
        help="Data source: 'cache' for candles_cache.parquet or path to CSV file"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate model on held-out data"
    )
    parser.add_argument(
        "--output",
        default="regime_detector.pkl",
        help="Output path for trained model"
    )
    parser.add_argument(
        "--compare",
        help="Compare with existing model at this path"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.compare, args.output, args.data_source)
    else:
        retrain_model(args.data_source, args.validate, args.output)
