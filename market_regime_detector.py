"""
Market Regime Detector - Advanced Risk Management System

Uses Gaussian HMM for regime-based position sizing and Isolation Forest 
for anomaly-based emergency exits. Optimized for Polars DataFrames.

Author: Trading System Risk Manager
Date: February 5, 2026
"""

import polars as pl
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from hmmlearn.hmm import GaussianHMM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Advanced Market Regime Detection and Risk Management System
    
    Architecture:
    - Gaussian HMM (3 states): Identifies market regimes (Bull/Bear, High Vol, Crash)
    - Isolation Forest: Detects anomalous market conditions for emergency exits
    - Polars-optimized feature engineering for high performance
    
    Features:
    - Log Returns: Price momentum
    - Parkinson Volatility: Intrabar volatility
    - Amihud Illiquidity: Price impact / market fragility
    - Liquidity Shock: Sudden liquidity changes (rug pull detection)
    - VWAP Deviation: Price vs fair value
    """
    
    def __init__(self, lookback_window: int = 500):
        """
        Initialize the Market Regime Detector
        
        Args:
            lookback_window: Number of candles to use for rolling features (default: 500)
        """
        self.lookback_window = lookback_window
        
        # Models
        self.hmm = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=100,
            random_state=42,
            verbose=False
        )
        self.isolation_forest = IsolationForest(
            contamination=0.01,  # Top 1% outliers
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        
        # Feature scaling
        self.scaler = StandardScaler()
        
        # State tracking
        self.crash_state_idx = None  # Will be identified during fit
        self.feature_names = [
            'log_returns',
            'parkinson_volatility',
            'amihud_illiquidity',
            'liquidity_shock',
            'vwap_deviation'
        ]
        self.is_fitted = False
        
        logger.info(f"✅ MarketRegimeDetector initialized (lookback: {lookback_window})")
    
    def prepare_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate market regime features using Polars vectorization
        
        Args:
            df: Polars DataFrame with columns:
                - timestamp, open, high, low, close
                - volume_token1 (USD volume)
                - num_trades
                - avg_liquidity
                - vwap
        
        Returns:
            DataFrame with original columns + engineered features
        """
        logger.debug(f"Preparing features for {len(df)} candles")
        
        # 1. Log Returns: ln(close / close.shift(1))
        df = df.with_columns([
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_returns")
        ])
        
        # 2. Parkinson Volatility: (ln(high / low)^2) / (4 * ln(2))
        ln2 = np.log(2)
        df = df.with_columns([
            ((pl.col("high") / pl.col("low")).log().pow(2) / (4 * ln2)).alias("parkinson_volatility")
        ])
        
        # 3. Amihud Illiquidity: abs(log_returns) / (volume_token1 + 1)
        # High value = Low liquidity / Fragile market
        df = df.with_columns([
            (pl.col("log_returns").abs() / (pl.col("volume_token1") + 1)).alias("amihud_illiquidity")
        ])
        
        # 4. Liquidity Shock: (avg_liquidity - avg_liquidity.shift(1)) / avg_liquidity.shift(1)
        # Sudden negative values indicate potential rug pull or liquidity drain
        df = df.with_columns([
            ((pl.col("avg_liquidity") - pl.col("avg_liquidity").shift(1)) / 
             (pl.col("avg_liquidity").shift(1) + 1e-10)).alias("liquidity_shock")
        ])
        
        # 5. VWAP Deviation: (close - vwap) / vwap
        df = df.with_columns([
            ((pl.col("close") - pl.col("vwap")) / (pl.col("vwap") + 1e-10)).alias("vwap_deviation")
        ])
        
        # Handle NaN and Inf values (replace with 0)
        for feature in self.feature_names:
            df = df.with_columns([
                pl.col(feature).fill_nan(0.0).fill_null(0.0).alias(feature)
            ])
            
            # Handle infinities
            df = df.with_columns([
                pl.when(pl.col(feature).is_infinite())
                .then(0.0)
                .otherwise(pl.col(feature))
                .alias(feature)
            ])
        
        logger.debug(f"✅ Generated {len(self.feature_names)} features")
        return df
    
    def fit(self, historical_df: pl.DataFrame):
        """
        Train the HMM and Isolation Forest models on historical data
        
        Args:
            historical_df: Polars DataFrame with market data
        """
        logger.info(f"🔧 Fitting models on {len(historical_df)} historical candles")
        
        # 1. Generate features
        df_with_features = self.prepare_features(historical_df)
        
        # Extract feature matrix (drop NaN rows from initial lags)
        feature_matrix = df_with_features.select(self.feature_names).drop_nulls()
        X = feature_matrix.to_numpy()
        
        if len(X) < 100:
            raise ValueError(f"Insufficient data for training: {len(X)} samples (need at least 100)")
        
        logger.info(f"📊 Training on {len(X)} samples with {X.shape[1]} features")
        
        # 2. Fit StandardScaler
        X_scaled = self.scaler.fit_transform(X)
        logger.info("✅ StandardScaler fitted")
        
        # 3. Fit HMM and identify Crash State (highest variance)
        self.hmm.fit(X_scaled)
        
        # Identify crash state by finding the state with highest variance
        state_variances = []
        for i in range(self.hmm.n_components):
            # Extract covariance for state i (diagonal elements = variances)
            cov_matrix = self.hmm.covars_[i]
            avg_variance = np.mean(np.diag(cov_matrix))
            state_variances.append(avg_variance)
            logger.info(f"  State {i}: avg_variance = {avg_variance:.6f}")
        
        self.crash_state_idx = np.argmax(state_variances)
        logger.info(f"🔴 Crash State identified: State {self.crash_state_idx} (variance: {max(state_variances):.6f})")
        
        # 4. Fit Isolation Forest
        self.isolation_forest.fit(X_scaled)
        logger.info("✅ Isolation Forest fitted")
        
        # Calculate anomaly baseline
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        anomaly_predictions = self.isolation_forest.predict(X_scaled)
        n_anomalies = np.sum(anomaly_predictions == -1)
        logger.info(f"📈 Anomaly baseline: {n_anomalies}/{len(X)} samples ({100*n_anomalies/len(X):.2f}%)")
        logger.info(f"   Score range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
        
        self.is_fitted = True
        logger.info("✅ Model fitting complete")
    
    def predict(self, latest_candles_df: pl.DataFrame) -> Dict:
        """
        Predict current market regime and emergency status
        
        Args:
            latest_candles_df: Recent candles (should include lookback window)
        
        Returns:
            Dictionary with:
            - position_scalar: float (0.0 to 1.0) - multiply against trade size
            - is_emergency: bool - True if anomaly detected
            - current_regime: int (0, 1, or 2) - current HMM state
            - regime_probs: list - [Prob_0, Prob_1, Prob_2]
            - crash_state_prob: float - probability of being in crash state
        """
        if not self.is_fitted:
            raise RuntimeError("Models not fitted. Call fit() first.")
        
        # Generate features
        df_with_features = self.prepare_features(latest_candles_df)
        
        # Extract features for the latest candle (after handling lags)
        feature_matrix = df_with_features.select(self.feature_names).drop_nulls()
        
        if len(feature_matrix) == 0:
            logger.warning("⚠️ No valid features after dropna - returning safe defaults")
            return {
                "position_scalar": 0.0,
                "is_emergency": True,
                "current_regime": self.crash_state_idx,
                "regime_probs": [0.0, 0.0, 1.0] if self.crash_state_idx == 2 else [0.0] * 3,
                "crash_state_prob": 1.0
            }
        
        # Get the most recent sample
        X_latest = feature_matrix.tail(1).to_numpy()
        X_scaled = self.scaler.transform(X_latest)
        
        # HMM Prediction
        # Note: HMM.predict works on sequences, so we use predict_proba for single sample
        regime_probs = self.hmm.predict_proba(X_scaled)[0]  # Shape: (n_components,)
        current_regime = int(np.argmax(regime_probs))
        crash_state_prob = float(regime_probs[self.crash_state_idx])
        
        # Position scalar: 1.0 - P(Crash)
        # Higher crash probability → Lower position size
        position_scalar = float(1.0 - crash_state_prob)
        
        # Isolation Forest Prediction
        anomaly_prediction = self.isolation_forest.predict(X_scaled)[0]
        is_emergency = bool(anomaly_prediction == -1)
        
        # Logging
        if is_emergency:
            logger.warning(f"🚨 EMERGENCY DETECTED! Anomaly in market conditions")
        
        if crash_state_prob > 0.5:
            logger.warning(f"🔴 CRASH STATE ALERT! Probability: {crash_state_prob:.2%}")
        
        logger.debug(f"Regime: {current_regime}, Probs: {regime_probs}, Position: {position_scalar:.2f}")
        
        return {
            "position_scalar": position_scalar,
            "is_emergency": is_emergency,
            "current_regime": current_regime,
            "regime_probs": regime_probs.tolist(),
            "crash_state_prob": crash_state_prob
        }
    
    def save(self, filepath: str):
        """
        Save trained models to disk
        
        Args:
            filepath: Path to save the model bundle
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted models. Call fit() first.")
        
        model_bundle = {
            'hmm': self.hmm,
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'crash_state_idx': self.crash_state_idx,
            'feature_names': self.feature_names,
            'lookback_window': self.lookback_window
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_bundle, filepath)
        logger.info(f"💾 Models saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load trained models from disk
        
        Args:
            filepath: Path to the saved model bundle
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_bundle = joblib.load(filepath)
        
        self.hmm = model_bundle['hmm']
        self.isolation_forest = model_bundle['isolation_forest']
        self.scaler = model_bundle['scaler']
        self.crash_state_idx = model_bundle['crash_state_idx']
        self.feature_names = model_bundle['feature_names']
        self.lookback_window = model_bundle.get('lookback_window', 500)
        self.is_fitted = True
        
        logger.info(f"✅ Models loaded from {filepath}")
        logger.info(f"   Crash State: {self.crash_state_idx}")
        logger.info(f"   Features: {', '.join(self.feature_names)}")


# Utility function for quick testing
def test_detector():
    """
    Test the MarketRegimeDetector with synthetic data
    """
    logger.info("🧪 Running detector test with synthetic data")
    
    # Generate synthetic candle data
    np.random.seed(42)
    n_candles = 1000
    
    synthetic_data = {
        'timestamp': pl.datetime_range(
            start=pl.datetime(2026, 1, 1),
            end=pl.datetime(2026, 2, 5),
            interval='15m',
            eager=True
        )[:n_candles],
        'open': 100 + np.cumsum(np.random.randn(n_candles) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(n_candles) * 0.5) + np.abs(np.random.randn(n_candles)),
        'low': 100 + np.cumsum(np.random.randn(n_candles) * 0.5) - np.abs(np.random.randn(n_candles)),
        'close': 100 + np.cumsum(np.random.randn(n_candles) * 0.5),
        'volume_token1': np.abs(np.random.randn(n_candles) * 10000 + 50000),
        'num_trades': np.random.randint(10, 100, n_candles),
        'avg_liquidity': np.abs(np.random.randn(n_candles) * 5000 + 20000),
        'vwap': 100 + np.cumsum(np.random.randn(n_candles) * 0.5)
    }
    
    df = pl.DataFrame(synthetic_data)
    
    # Initialize and fit detector
    detector = MarketRegimeDetector(lookback_window=100)
    detector.fit(df[:800])  # Train on first 800 candles
    
    # Predict on recent data
    result = detector.predict(df[700:])  # Use 100 candles for context
    
    logger.info("=" * 60)
    logger.info("TEST RESULTS:")
    logger.info(f"  Position Scalar: {result['position_scalar']:.2f}")
    logger.info(f"  Emergency Status: {result['is_emergency']}")
    logger.info(f"  Current Regime: {result['current_regime']}")
    logger.info(f"  Regime Probs: {[f'{p:.2%}' for p in result['regime_probs']]}")
    logger.info(f"  Crash State Prob: {result['crash_state_prob']:.2%}")
    logger.info("=" * 60)
    
    # Test save/load
    test_path = "test_regime_detector.pkl"
    detector.save(test_path)
    
    detector2 = MarketRegimeDetector()
    detector2.load(test_path)
    result2 = detector2.predict(df[700:])
    
    assert result == result2, "Save/load test failed!"
    logger.info("✅ Save/load test passed")
    
    # Cleanup
    Path(test_path).unlink()
    
    return detector, result


if __name__ == "__main__":
    logger.info("🚀 MarketRegimeDetector - Risk Management System")
    test_detector()
