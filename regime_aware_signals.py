"""
Live Signal Generator with Regime-Based Risk Management

This is a drop-in example showing how to integrate the MarketRegimeDetector
into your existing live_signal_generator.py workflow.

Integration Points:
1. Load detector at startup
2. Apply regime filtering after XGBoost predictions
3. Log risk adjustments for monitoring
"""

import polars as pl
from market_regime_detector import MarketRegimeDetector
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RegimeAwareSignalGenerator:
    """
    Enhanced signal generator with regime-based risk management
    
    This wraps around your existing signal generation logic and adds:
    - Dynamic position sizing based on HMM regime detection
    - Emergency stops via Isolation Forest anomaly detection
    - Risk-adjusted confidence scores
    """
    
    def __init__(self, detector_path: str = "regime_detector.pkl"):
        """
        Initialize the regime-aware signal generator
        
        Args:
            detector_path: Path to trained MarketRegimeDetector model
        """
        self.detector_path = Path(detector_path)
        self.detector = None
        self.regime_stats = {
            'total_signals': 0,
            'emergency_blocks': 0,
            'position_reduced': 0,
            'position_scalars': []
        }
        
        self._load_detector()
    
    def _load_detector(self):
        """Load or train the regime detector"""
        if self.detector_path.exists():
            logger.info(f"📦 Loading regime detector from {self.detector_path}")
            self.detector = MarketRegimeDetector()
            self.detector.load(str(self.detector_path))
            logger.info("✅ Regime detector loaded")
        else:
            logger.warning(f"⚠️ No detector found at {self.detector_path}")
            logger.warning("   Run regime_integration_example.py to train one")
            self.detector = None
    
    def apply_regime_filter(
        self,
        signals_df: pl.DataFrame,
        candles_history: dict,
        min_confidence: float = 0.5
    ) -> pl.DataFrame:
        """
        Apply regime-based risk management to trading signals
        
        Args:
            signals_df: DataFrame with columns [pair_address, confidence, ...]
            candles_history: Dict[pair_address -> pl.DataFrame] with candle data
            min_confidence: Minimum adjusted confidence to keep signal (default: 0.5)
        
        Returns:
            Filtered signals with additional columns:
            - position_scalar: Risk multiplier (0.0 to 1.0)
            - adjusted_confidence: confidence × position_scalar
            - regime: Current regime state (0, 1, 2)
            - is_emergency: Emergency stop flag
        """
        if self.detector is None:
            logger.warning("⚠️ Regime detector not loaded - returning unfiltered signals")
            return signals_df.with_columns([
                pl.lit(1.0).alias('position_scalar'),
                pl.col('confidence').alias('adjusted_confidence'),
                pl.lit(1).alias('regime'),
                pl.lit(False).alias('is_emergency')
            ])
        
        logger.info(f"🎯 Applying regime filter to {len(signals_df)} signals")
        
        # Collect regime predictions for each signal
        regime_data = []
        
        for row in signals_df.iter_rows(named=True):
            pair_address = row['pair_address']
            base_confidence = row['confidence']
            
            self.regime_stats['total_signals'] += 1
            
            # Get candle history for this pair
            if pair_address not in candles_history:
                logger.warning(f"⚠️ No candles for {pair_address[:10]}... - skipping")
                regime_data.append({
                    'pair_address': pair_address,
                    'position_scalar': 0.0,
                    'adjusted_confidence': 0.0,
                    'regime': 2,
                    'is_emergency': True
                })
                self.regime_stats['emergency_blocks'] += 1
                continue
            
            pair_candles = candles_history[pair_address]
            
            # Need sufficient history for regime detection
            if len(pair_candles) < 5:
                logger.debug(f"Insufficient history for {pair_address[:10]}... ({len(pair_candles)} candles)")
                # Conservative default: 50% position size
                regime_data.append({
                    'pair_address': pair_address,
                    'position_scalar': 0.5,
                    'adjusted_confidence': base_confidence * 0.5,
                    'regime': 1,
                    'is_emergency': False
                })
                self.regime_stats['position_reduced'] += 1
                self.regime_stats['position_scalars'].append(0.5)
                continue
            
            # Predict regime for this pair
            try:
                prediction = self.detector.predict(pair_candles)
                
                position_scalar = prediction['position_scalar']
                is_emergency = prediction['is_emergency']
                regime = prediction['current_regime']
                crash_prob = prediction['crash_state_prob']
                
                # Log warnings
                if is_emergency:
                    logger.warning(f"🚨 EMERGENCY for {pair_address[:10]}... - BLOCKING")
                    self.regime_stats['emergency_blocks'] += 1
                elif crash_prob > 0.5:
                    logger.warning(f"🔴 HIGH CRASH RISK ({crash_prob:.0%}) for {pair_address}")
                elif position_scalar < 0.8:
                    logger.info(f"⚠️ Reduced position ({position_scalar:.0%}) for {pair_address[:10]}...")
                    self.regime_stats['position_reduced'] += 1
                
                adjusted_confidence = base_confidence * position_scalar
                
                regime_data.append({
                    'pair_address': pair_address,
                    'position_scalar': position_scalar,
                    'adjusted_confidence': adjusted_confidence,
                    'regime': regime,
                    'is_emergency': is_emergency
                })
                
                self.regime_stats['position_scalars'].append(position_scalar)
            
            except Exception as e:
                logger.error(f"❌ Regime prediction failed for {pair_address[:10]}...: {e}")
                # Fail-safe: block on error
                regime_data.append({
                    'pair_address': pair_address,
                    'position_scalar': 0.0,
                    'adjusted_confidence': 0.0,
                    'regime': 2,
                    'is_emergency': True
                })
                self.regime_stats['emergency_blocks'] += 1
        
        # Convert to DataFrame and join with signals
        regime_df = pl.DataFrame(regime_data)
        signals_with_regime = signals_df.join(regime_df, on='pair_address', how='inner')
        
        # Filter by adjusted confidence and emergency status
        filtered_signals = signals_with_regime.filter(
            (~pl.col('is_emergency')) &  # No emergencies
            (pl.col('adjusted_confidence') >= min_confidence)  # Above threshold
        )
        
        # Log summary
        n_total = len(signals_df)
        n_emergency = signals_with_regime.filter(pl.col('is_emergency')).shape[0]
        n_low_conf = signals_with_regime.filter(
            (~pl.col('is_emergency')) & (pl.col('adjusted_confidence') < min_confidence)
        ).shape[0]
        n_passed = len(filtered_signals)
        
        logger.info("=" * 60)
        logger.info("📊 REGIME FILTER RESULTS")
        logger.info("=" * 60)
        logger.info(f"  Total Signals: {n_total}")
        logger.info(f"  🚨 Emergency Blocks: {n_emergency}")
        logger.info(f"  ⚠️  Low Confidence: {n_low_conf}")
        logger.info(f"  ✅ Passed Filter: {n_passed}")
        logger.info(f"  📉 Filter Rate: {100*(n_total-n_passed)/n_total:.1f}%")
        logger.info("=" * 60)
        
        return filtered_signals
    
    def get_regime_stats(self) -> dict:
        """Get statistics about regime filtering performance"""
        if self.regime_stats['total_signals'] == 0:
            return self.regime_stats
        
        stats = self.regime_stats.copy()
        stats['emergency_rate'] = stats['emergency_blocks'] / stats['total_signals']
        stats['reduction_rate'] = stats['position_reduced'] / stats['total_signals']
        
        if stats['position_scalars']:
            stats['avg_position_scalar'] = sum(stats['position_scalars']) / len(stats['position_scalars'])
        else:
            stats['avg_position_scalar'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.regime_stats = {
            'total_signals': 0,
            'emergency_blocks': 0,
            'position_reduced': 0,
            'position_scalars': []
        }


# Example integration into existing live_signal_generator.py
def example_integration():
    """
    Example showing how to integrate into your existing signal generation loop
    """
    
    # At startup (in __init__ or main):
    regime_filter = RegimeAwareSignalGenerator(detector_path="regime_detector.pkl")
    
    # In your signal generation loop (where you currently generate signals):
    # ...existing code to load candles_history and generate xgb_signals...
    
    # Mock example data
    xgb_signals = pl.DataFrame({
        'pair_address': ['0xabc...', '0xdef...', '0x123...'],
        'confidence': [0.85, 0.92, 0.78],
        'side': ['BUY', 'BUY', 'BUY'],
        'timestamp': [pl.datetime(2026, 2, 5)] * 3
    })
    
    candles_history = {}  # Your existing candle cache
    
    # Apply regime filtering AFTER XGBoost predictions
    risk_adjusted_signals = regime_filter.apply_regime_filter(
        signals_df=xgb_signals,
        candles_history=candles_history,
        min_confidence=0.5  # Your threshold
    )
    
    # Now use risk_adjusted_signals for execution
    # The 'adjusted_confidence' column is already risk-adjusted
    for signal in risk_adjusted_signals.iter_rows(named=True):
        pair_address = signal['pair_address']
        adjusted_conf = signal['adjusted_confidence']
        position_scalar = signal['position_scalar']
        
        logger.info(f"📈 Signal: {pair_address}")
        logger.info(f"   Original: {signal['confidence']:.2f}")
        logger.info(f"   Adjusted: {adjusted_conf:.2f} (×{position_scalar:.2f})")
        
        # Execute trade with adjusted confidence
        # execute_trade(pair_address, confidence=adjusted_conf)
    
    # Periodically log statistics
    stats = regime_filter.get_regime_stats()
    logger.info("\n" + "=" * 60)
    logger.info("📊 CUMULATIVE REGIME STATISTICS")
    logger.info("=" * 60)
    logger.info(f"  Total Signals Processed: {stats['total_signals']}")
    logger.info(f"  Emergency Blocks: {stats['emergency_blocks']} ({stats.get('emergency_rate', 0):.1%})")
    logger.info(f"  Positions Reduced: {stats['position_reduced']} ({stats.get('reduction_rate', 0):.1%})")
    logger.info(f"  Avg Position Scalar: {stats.get('avg_position_scalar', 0):.2f}")
    logger.info("=" * 60)


# Standalone test function
def test_with_real_data():
    """
    Test the regime filter with real cache data
    """
    import pickle
    
    logger.info("🧪 Testing regime-aware signal generator")
    
    # Load real candles
    with open("live_candles_cache.pkl", 'rb') as f:
        candles_history = pickle.load(f)
    
    # Create mock signals from pairs with most history
    sorted_pairs = sorted(
        candles_history.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:20]
    
    mock_signals = pl.DataFrame({
        'pair_address': [p[0] for p in sorted_pairs],
        'confidence': [0.7 + i*0.01 for i in range(20)],
        'side': ['BUY'] * 20
    })
    
    # Initialize filter
    regime_filter = RegimeAwareSignalGenerator()
    
    # Apply filtering
    filtered = regime_filter.apply_regime_filter(
        signals_df=mock_signals,
        candles_history=candles_history,
        min_confidence=0.5
    )
    
    # Display results
    if len(filtered) > 0:
        print("\n" + "=" * 80)
        print("RISK-ADJUSTED SIGNALS")
        print("=" * 80)
        print(filtered.select([
            'pair_address',
            'confidence',
            'position_scalar',
            'adjusted_confidence',
            'regime'
        ]).head(10))
    
    # Show stats
    stats = regime_filter.get_regime_stats()
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Emergency Block Rate: {stats.get('emergency_rate', 0):.1%}")
    print(f"Position Reduction Rate: {stats.get('reduction_rate', 0):.1%}")
    print(f"Avg Position Scalar: {stats.get('avg_position_scalar', 0):.2f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_with_real_data()
