"""
Integration Example: Using MarketRegimeDetector with Live Signal Generator

This script demonstrates how to integrate the regime detector into your
trading pipeline for dynamic position sizing and emergency stops.
"""

import polars as pl
import pickle
from pathlib import Path
from market_regime_detector import MarketRegimeDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_live_candles() -> dict:
    """Load the live candles cache"""
    cache_path = Path("live_candles_cache.pkl")
    if not cache_path.exists():
        raise FileNotFoundError("live_candles_cache.pkl not found. Run live_signal_generator.py first.")
    
    with open(cache_path, 'rb') as f:
        candles_history = pickle.load(f)
    
    logger.info(f"✅ Loaded {len(candles_history)} pairs from cache")
    return candles_history


def train_regime_detector(candles_history: dict, sample_pairs: int = 50) -> MarketRegimeDetector:
    """
    Train the regime detector on historical data from multiple pairs
    
    Args:
        candles_history: Dict[pair_address -> pl.DataFrame]
        sample_pairs: Number of pairs to use for training (default: 50)
    
    Returns:
        Trained MarketRegimeDetector
    """
    logger.info(f"🔧 Training regime detector on {sample_pairs} pairs")
    
    # Combine candles from multiple pairs for training
    training_pairs = list(candles_history.keys())[:sample_pairs]
    
    # Concatenate data from multiple pairs
    training_data = []
    for pair_address in training_pairs:
        df = candles_history[pair_address]
        if len(df) >= 5:  # Reduced threshold for fresh cache (was 20)
            training_data.append(df)
    
    if not training_data:
        raise ValueError(f"No pairs with sufficient history for training. Need at least 5 candles per pair.")
    
    # Combine all training data
    combined_df = pl.concat(training_data)
    logger.info(f"📊 Training dataset: {len(combined_df)} candles from {len(training_data)} pairs")
    
    # Initialize and train detector
    detector = MarketRegimeDetector(lookback_window=100)
    detector.fit(combined_df)
    
    return detector


def apply_regime_scaling(signals_df: pl.DataFrame, detector: MarketRegimeDetector, 
                         candles_history: dict) -> pl.DataFrame:
    """
    Apply regime-based position sizing to trading signals
    
    Args:
        signals_df: DataFrame with trading signals (pair_address, confidence, etc.)
        detector: Trained MarketRegimeDetector
        candles_history: Dict of candle data for each pair
    
    Returns:
        signals_df with additional columns:
        - position_scalar: Risk-adjusted multiplier (0.0 to 1.0)
        - is_emergency: Boolean flag for emergency exit
        - regime: Current market regime (0, 1, 2)
    """
    logger.info(f"🎯 Applying regime scaling to {len(signals_df)} signals")
    
    # Collect regime predictions for each pair
    regime_results = []
    
    for row in signals_df.iter_rows(named=True):
        pair_address = row['pair_address']
        
        # Get candle history for this pair
        if pair_address not in candles_history:
            logger.warning(f"⚠️ No candle history for {pair_address[:10]}... - skipping")
            regime_results.append({
                'pair_address': pair_address,
                'position_scalar': 0.0,
                'is_emergency': True,
                'regime': 2  # Assume crash state if no data
            })
            continue
        
        pair_candles = candles_history[pair_address]
        
        # Need at least 5 candles for regime detection (reduced from 20 for fresh cache)
        if len(pair_candles) < 5:
            logger.warning(f"⚠️ Insufficient history for {pair_address[:10]}... ({len(pair_candles)} candles)")
            regime_results.append({
                'pair_address': pair_address,
                'position_scalar': 0.5,  # Conservative sizing
                'is_emergency': False,
                'regime': 1
            })
            continue
        
        # Predict regime for this pair
        try:
            prediction = detector.predict(pair_candles)
            
            regime_results.append({
                'pair_address': pair_address,
                'position_scalar': prediction['position_scalar'],
                'is_emergency': prediction['is_emergency'],
                'regime': prediction['current_regime']
            })
            
            if prediction['is_emergency']:
                logger.warning(f"🚨 EMERGENCY detected for {pair_address[:10]}...")
            elif prediction['crash_state_prob'] > 0.3:
                logger.warning(f"⚠️ High crash risk ({prediction['crash_state_prob']:.1%}) for {pair_address[:10]}...")
        
        except Exception as e:
            logger.error(f"❌ Error predicting regime for {pair_address[:10]}...: {e}")
            regime_results.append({
                'pair_address': pair_address,
                'position_scalar': 0.0,
                'is_emergency': True,
                'regime': 2
            })
    
    # Convert to DataFrame and join with signals
    regime_df = pl.DataFrame(regime_results)
    signals_with_regime = signals_df.join(regime_df, on='pair_address', how='left')
    
    # Calculate adjusted position size
    signals_with_regime = signals_with_regime.with_columns([
        (pl.col('confidence') * pl.col('position_scalar')).alias('adjusted_confidence')
    ])
    
    # Filter out emergency signals
    n_emergency = signals_with_regime.filter(pl.col('is_emergency')).shape[0]
    if n_emergency > 0:
        logger.warning(f"🚨 Filtering {n_emergency} emergency signals")
    
    safe_signals = signals_with_regime.filter(~pl.col('is_emergency'))
    
    logger.info(f"✅ Regime scaling complete: {len(safe_signals)}/{len(signals_df)} signals approved")
    
    return safe_signals


def main():
    """
    Example workflow: Train detector → Load signals → Apply regime scaling
    """
    logger.info("=" * 70)
    logger.info("🚀 Regime-Aware Trading Pipeline")
    logger.info("=" * 70)
    
    # 1. Load candle data
    candles_history = load_live_candles()
    
    # 2. Train regime detector (do this once, then save)
    detector_path = Path("regime_detector.pkl")
    
    if detector_path.exists():
        logger.info(f"📦 Loading existing detector from {detector_path}")
        detector = MarketRegimeDetector()
        detector.load(str(detector_path))
    else:
        logger.info("🔧 Training new detector")
        detector = train_regime_detector(candles_history, sample_pairs=100)
        detector.save(str(detector_path))
    
    # 3. Example: Load latest signals (mock data for demonstration)
    # In production, replace this with actual signals from your signal generator
    logger.info("\n📊 Loading trading signals...")
    
    sample_pairs = list(candles_history.keys())[:10]
    mock_signals = pl.DataFrame({
        'pair_address': sample_pairs,
        'confidence': [0.85, 0.92, 0.78, 0.88, 0.75, 0.91, 0.82, 0.79, 0.87, 0.90],
        'side': ['BUY'] * 10,
        'timestamp': [pl.datetime(2026, 2, 5, 13, 0)] * 10
    })
    
    logger.info(f"📥 Received {len(mock_signals)} trading signals")
    
    # 4. Apply regime-based position sizing
    risk_adjusted_signals = apply_regime_scaling(mock_signals, detector, candles_history)
    
    # 5. Display results
    logger.info("\n" + "=" * 70)
    logger.info("📊 RISK-ADJUSTED SIGNALS")
    logger.info("=" * 70)
    
    if len(risk_adjusted_signals) > 0:
        display_cols = [
            'pair_address', 'confidence', 'position_scalar', 
            'adjusted_confidence', 'regime', 'is_emergency'
        ]
        
        # Truncate pair addresses for display
        display_df = risk_adjusted_signals.with_columns([
            pl.col('pair_address').str.slice(0, 10).alias('pair_short')
        ])
        
        print("\n" + str(display_df.select(['pair_short', 'confidence', 'position_scalar', 
                                              'adjusted_confidence', 'regime']).head(10)))
        
        # Summary statistics
        logger.info("\n" + "=" * 70)
        logger.info("📈 SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Signals Passed: {len(risk_adjusted_signals)}/{len(mock_signals)}")
        logger.info(f"  Avg Position Scalar: {risk_adjusted_signals['position_scalar'].mean():.2f}")
        logger.info(f"  Regimes: 0={risk_adjusted_signals.filter(pl.col('regime')==0).shape[0]}, "
                   f"1={risk_adjusted_signals.filter(pl.col('regime')==1).shape[0]}, "
                   f"2={risk_adjusted_signals.filter(pl.col('regime')==2).shape[0]}")
    else:
        logger.warning("⚠️ No signals passed regime filtering!")
    
    logger.info("\n✅ Pipeline complete")


if __name__ == "__main__":
    main()
