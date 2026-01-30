#!/usr/bin/env python3
"""
Live Trading Signal Generator
==============================

Streams live data from Envio Hypersync, generates candles, computes features,
and runs predictions using the trained XGBoost model to generate trading signals.
"""

import asyncio
import polars as pl
import xgboost as xgb
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import json
import pickle
import time

from envio_hypersync import LiveSwapStreamer, EnvioConfig
from calibrate_model import XGBoostCalibrator
from position_sizing import PositionSizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class LiveConfig:
    """Configuration for live signal generation"""
    
    # Model configuration
    MODEL_PATH = "xgb_model.json"  # Newly trained model
    CALIBRATOR_PATH = "xgb_calibrator.pkl"  # Probability calibrator (isotonic regression)
    USE_CALIBRATION = True  # Apply calibration to fix probability predictions
    SIGNAL_THRESHOLD = 0.32  # CALIBRATED threshold (equivalent to 0.73 raw, ~450-500 signals)
    # NOTE: Calibration correctly maps predictions to actual probabilities
    # Calibrated 0.32 ≈ Raw 0.73 ≈ ~66-75% actual win rate
    # Multi-batch analysis: 66.8% avg win rate, 75.3% best batch, 8:1 risk-reward
    
    # Position sizing configuration
    USE_POSITION_SIZING = True  # Add Kelly Criterion position sizing to signals
    KELLY_FRACTION = 0.25  # Quarter-Kelly (25% of full Kelly)
    MAX_POSITIONS = 10  # Maximum concurrent positions
    MAX_CAPITAL_DEPLOYED = 0.50  # Maximum 50% of capital deployed
    MIN_POSITION_SIZE = 0.01  # 1% minimum position
    MAX_POSITION_SIZE = 0.10  # 10% maximum position
    
    # Feature names (must match training - 15-min candles)
    FEATURE_COLS = [
        "return_1c", "return_4c", "return_16c",
        "ema_cross_signal", "range_normalized",
        "buy_sell_ratio", "volume_zscore", "trade_accel",
        "volatility_4h", "volatility_24h", "vol_regime_change"
    ]
    
    # Data requirements for features
    MIN_HISTORY_CANDLES = 20  # Need ~20 candles (5 hours at 15min intervals) for features
    UPDATE_INTERVAL_SECONDS = 900  # Generate signals every 15 minutes (matching candle interval)
    
    # Output configuration
    SIGNALS_OUTPUT_PATH = "signals_live.csv"
    HISTORICAL_CANDLES_PATH = "live_candles_history.parquet"
    
    # Persistence settings
    CANDLE_CACHE_PATH = "live_candles_cache.pkl"
    SAVE_INTERVAL_SECONDS = 300  # Save every 5 minutes
    
    # Base chain pair universe (if available)
    PAIR_UNIVERSE_PATH = "/home/hamzabhatti18/Desktop/Genesis-labs/backtesting/pair-universe"


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Compute features from candle data (matching training pipeline)"""
    
    @staticmethod
    def compute_returns(df: pl.DataFrame) -> pl.DataFrame:
        """Compute price returns (15-min candles)"""
        return df.with_columns([
            (100 * (pl.col("close") / pl.col("close").shift(1) - 1)).alias("return_1c"),
            (100 * (pl.col("close") / pl.col("close").shift(4) - 1)).alias("return_4c"),
            (100 * (pl.col("close") / pl.col("close").shift(16) - 1)).alias("return_16c"),
        ])
    
    @staticmethod
    def compute_ema_cross(df: pl.DataFrame) -> pl.DataFrame:
        """Compute EMA crossover signal (4 vs 16 candles = 1h vs 4h)"""
        df = df.with_columns([
            pl.col("close").ewm_mean(span=4, adjust=False).alias("ema_4"),
            pl.col("close").ewm_mean(span=16, adjust=False).alias("ema_16"),
        ])
        
        return df.with_columns([
            ((pl.col("ema_4") > pl.col("ema_16")).cast(pl.Int32) -
             (pl.col("ema_4") <= pl.col("ema_16")).cast(pl.Int32)).alias("ema_cross_signal")
        ]).drop(["ema_4", "ema_16"])
    
    @staticmethod
    def compute_range_normalized(df: pl.DataFrame) -> pl.DataFrame:
        """Compute normalized price range"""
        return df.with_columns([
            (100 * (pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_normalized")
        ])
    
    @staticmethod
    def compute_flow_features(df: pl.DataFrame) -> pl.DataFrame:
        """Compute buy/sell flow features"""
        # For now, use volume as proxy (would need actual buy/sell data)
        df = df.with_columns([
            (pl.col("volume_token1") / (pl.col("volume_token0") + 1e-9)).alias("buy_sell_ratio")
        ])
        
        # Volume z-score
        df = df.with_columns([
            ((pl.col("volume_token1") - pl.col("volume_token1").rolling_mean(7, min_samples=3)) /
             (pl.col("volume_token1").rolling_std(7, min_samples=3) + 1e-9)).alias("volume_zscore")
        ])
        
        # Trade acceleration
        df = df.with_columns([
            (pl.col("num_trades") - pl.col("num_trades").shift(1)).alias("trade_accel")
        ])
        
        return df
    
    @staticmethod
    def compute_volatility(df: pl.DataFrame) -> pl.DataFrame:
        """Compute volatility features (4h = 16 candles, 24h = 96 candles)"""
        df = df.with_columns([
            pl.col("return_1c").rolling_std(16, min_samples=8).alias("volatility_4h"),
            pl.col("return_1c").rolling_std(96, min_samples=20).alias("volatility_24h"),
        ])
        
        # Volatility regime change
        df = df.with_columns([
            (100 * (pl.col("volatility_4h") / (pl.col("volatility_24h") + 1e-9) - 1)).alias("vol_regime_change")
        ])
        
        return df
    
    @classmethod
    def compute_all_features(cls, df: pl.DataFrame) -> pl.DataFrame:
        """Compute all features required by the model"""
        
        # Must have required columns
        required_cols = ["pair_address", "timestamp", "open", "high", "low", "close", 
                        "volume_token0", "volume_token1", "num_trades"]
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Sort by pair and time
        df = df.sort(["pair_address", "timestamp"])
        
        # Compute features per pair
        df = df.group_by("pair_address").map_groups(lambda group: (
            cls.compute_returns(group)
            .pipe(cls.compute_ema_cross)
            .pipe(cls.compute_range_normalized)
            .pipe(cls.compute_flow_features)
            .pipe(cls.compute_volatility)
        ))
        
        # Fill NaN with 0
        for col in LiveConfig.FEATURE_COLS:
            if col in df.columns:
                df = df.with_columns([
                    pl.col(col).fill_nan(0).fill_null(0)
                ])

        # --- NEW CODE: Save to JSON ---
        output_file = "features.json"
        df.write_json(output_file)
        print(f"Features saved to {output_file}")
        # ------------------------------
        
        return df


# ============================================================================
# LIVE SIGNAL GENERATOR
# ============================================================================

class LiveSignalGenerator:
    """Generates trading signals from live streamed data"""
    
    def __init__(self, model_path: str = LiveConfig.MODEL_PATH):
        self.model_path = Path(model_path)
        self.model: Optional[xgb.Booster] = None
        self.streamer = LiveSwapStreamer()
        self.candles_history: Dict[str, pl.DataFrame] = {}  # pair_address -> candles
        self.last_signal_time = datetime.now()
        self.last_save_time = time.time()
        self.cache_path = Path(LiveConfig.CANDLE_CACHE_PATH)
        
        # Load cached candles on startup
        self._load_cache()
        
    def _load_cache(self):
        """Load cached candles from disk"""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    self.candles_history = pickle.load(f)
                
                total_candles = sum(len(df) for df in self.candles_history.values())
                ready_pairs = sum(1 for df in self.candles_history.values() if len(df) >= LiveConfig.MIN_HISTORY_CANDLES)
                
                logger.info(f"✅ Loaded cache: {len(self.candles_history)} pairs, {total_candles} candles")
                logger.info(f"✅ {ready_pairs} pairs ready with {LiveConfig.MIN_HISTORY_CANDLES}+ candles")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}. Starting fresh.")
                self.candles_history = {}
        else:
            logger.info("No cache found, starting fresh")
            self.candles_history = {}
    
    def _save_cache(self):
        """Save candles to disk cache"""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.candles_history, f)
            
            total_candles = sum(len(df) for df in self.candles_history.values())
            logger.info(f"💾 Saved cache: {len(self.candles_history)} pairs, {total_candles} candles")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def load_model(self):
        """Load trained XGBoost model and calibrator"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
            
        self.model = xgb.Booster()
        self.model.load_model(str(self.model_path))
        logger.info(f"✓ Loaded model from {self.model_path}")
        
        # Load calibrator if enabled
        if LiveConfig.USE_CALIBRATION:
            calibrator_path = Path(LiveConfig.CALIBRATOR_PATH)
            if calibrator_path.exists():
                self.calibrator = XGBoostCalibrator.load(str(calibrator_path))
                logger.info(f"✓ Loaded calibrator from {calibrator_path}")
            else:
                logger.warning(f"⚠️  Calibrator not found at {calibrator_path}, using raw probabilities")
                self.calibrator = None
        else:
            self.calibrator = None
            logger.info("ℹ️  Calibration disabled, using raw model probabilities")
        
    def update_candles_history(self, new_candles: pl.DataFrame):
        """Add new candles to historical data"""
        if len(new_candles) == 0:
            return
            
        for pair_address in new_candles["pair_address"].unique():
            pair_candles = new_candles.filter(pl.col("pair_address") == pair_address)
            
            if pair_address not in self.candles_history:
                self.candles_history[pair_address] = pair_candles
            else:
                # Append and deduplicate by timestamp
                self.candles_history[pair_address] = (
                    pl.concat([self.candles_history[pair_address], pair_candles])
                    .unique(subset=["timestamp"], keep="last")
                    .sort("timestamp")
                )
                
            # Keep only recent history (keep last N candles)
            max_candles = LiveConfig.MIN_HISTORY_CANDLES + 100  # Keep extra buffer
            if len(self.candles_history[pair_address]) > max_candles:
                self.candles_history[pair_address] = self.candles_history[pair_address].tail(max_candles)
        
        # Log candle counts for debugging
        ready_pairs = [p for p, df in self.candles_history.items() if len(df) >= 20]
        if ready_pairs:
            logger.info(f"✅ {len(ready_pairs)} pairs with 20+ candles (ready for signals)")
        else:
            total_candles = [(p, len(df)) for p, df in self.candles_history.items()]
            max_candles_pair = max(total_candles, key=lambda x: x[1]) if total_candles else (None, 0)
            logger.warning(f"⚠️ No pairs ready yet. Max candles: {max_candles_pair[1]} (need 20+)")
            
    def get_all_candles(self) -> pl.DataFrame:
        """Combine all pair candles into single dataframe"""
        if not self.candles_history:
            return pl.DataFrame()
            
        return pl.concat([df for df in self.candles_history.values()])
        
    def generate_signals(self) -> pl.DataFrame:
        """Generate trading signals from current candle data"""
        
        if self.model is None:
            self.load_model()
            
        # Get all candles
        all_candles = self.get_all_candles()
        
        if len(all_candles) == 0:
            logger.warning("No candles available for prediction")
            return pl.DataFrame()
            
        # Show candle distribution per pair
        candle_counts = [(p, len(df)) for p, df in self.candles_history.items()]
        avg_candles = sum(c for _, c in candle_counts) / len(candle_counts) if candle_counts else 0
        logger.info(f"Processing {len(all_candles)} candles for {len(self.candles_history)} pairs (avg: {avg_candles:.1f} candles/pair)")
        
        # Compute features
        try:
            df_features = FeatureEngineer.compute_all_features(all_candles)
        except Exception as e:
            logger.error(f"Error computing features: {e}", exc_info=True)
            return pl.DataFrame()
            
        # Filter to rows with enough history (non-null features)
        valid_rows = df_features.filter(
            pl.all_horizontal([pl.col(c).is_not_null() for c in LiveConfig.FEATURE_COLS])
        )
        
        if len(valid_rows) == 0:
            logger.warning("⚠️ No valid rows with complete features (need more candle history)")
            return pl.DataFrame()
            
        logger.info(f"✅ Computing predictions for {len(valid_rows)} valid rows (from {len(df_features)} total)")
        
        # Get latest row per pair
        latest_per_pair = valid_rows.group_by("pair_address").agg([
            pl.all().sort_by("timestamp").last()
        ])
        
        # Prepare features for prediction
        X = latest_per_pair.select(LiveConfig.FEATURE_COLS).to_numpy()
        dmatrix = xgb.DMatrix(X, feature_names=LiveConfig.FEATURE_COLS)
        
        # Predict (raw probabilities)
        pred_proba_raw = self.model.predict(dmatrix)
        
        # Apply calibration if available
        if self.calibrator is not None:
            pred_proba = self.calibrator.transform(pred_proba_raw)
            logger.info(f"📊 Calibration: raw mean={pred_proba_raw.mean():.3f} → calibrated mean={pred_proba.mean():.3f}")
        else:
            pred_proba = pred_proba_raw
        
        # Log prediction distribution
        logger.info(f"📊 Prediction stats: min={pred_proba.min():.3f}, max={pred_proba.max():.3f}, mean={pred_proba.mean():.3f}")
        above_50 = (pred_proba >= 0.5).sum()
        above_70 = (pred_proba >= 0.7).sum()
        above_80 = (pred_proba >= 0.8).sum()
        logger.info(f"📈 Confidence levels: >0.5: {above_50}, >0.7: {above_70}, >0.8: {above_80}")
        
        # Add predictions to dataframe
        result = latest_per_pair.with_columns([
            pl.Series("pred_proba", pred_proba),
            pl.Series("signal", (pred_proba >= LiveConfig.SIGNAL_THRESHOLD).astype(int))
        ])
        
        # Filter to signals only
        signals = result.filter(pl.col("signal") == 1).sort("pred_proba", descending=True)
        
        # Add position sizing if enabled
        if LiveConfig.USE_POSITION_SIZING and len(signals) > 0:
            sizer = PositionSizer(
                kelly_fraction=LiveConfig.KELLY_FRACTION,
                min_position_size=LiveConfig.MIN_POSITION_SIZE,
                max_position_size=LiveConfig.MAX_POSITION_SIZE,
                max_positions=LiveConfig.MAX_POSITIONS,
                max_capital_deployed=LiveConfig.MAX_CAPITAL_DEPLOYED,
            )
            
            # Add position sizing columns
            signals = sizer.add_position_sizes(signals, confidence_col="pred_proba")
            
            # Apply portfolio limits and select top signals
            signals = sizer.apply_portfolio_limits(signals, sort_by="expected_value")
            
            # Log selection summary
            n_selected = signals.filter(pl.col("selected") == True).shape[0]
            total_capital = signals.filter(pl.col("selected") == True)["adjusted_position_size"].sum()
            logger.info(f"📊 Position Sizing: {n_selected}/{len(signals)} signals selected, {total_capital:.1%} capital deployed")
        
        return signals
        
    async def run_live(self):
        """Main loop: stream data and generate signals"""
        
        logger.info("="*80)
        logger.info("LIVE TRADING SIGNAL GENERATOR - Starting")
        logger.info("="*80)
        logger.info(f"Base Chain ID: {EnvioConfig.CHAIN_ID}")
        logger.info(f"Signal Threshold: {LiveConfig.SIGNAL_THRESHOLD}")
        logger.info(f"Update Interval: {LiveConfig.UPDATE_INTERVAL_SECONDS}s")
        logger.info("="*80)
        
        # Backfill historical data to populate features properly
        logger.info("\n🔄 Starting historical data backfill...")
        try:
            historical_candles = await self.streamer.backfill_historical_candles(hours=25)
            
            if len(historical_candles) > 0:
                logger.info(f"📦 Received {len(historical_candles)} historical candles")
                self.update_candles_history(historical_candles)
                
                # Check readiness after backfill
                ready_pairs = sum(1 for df in self.candles_history.values() if len(df) >= LiveConfig.MIN_HISTORY_CANDLES)
                total_candles = sum(len(df) for df in self.candles_history.values())
                logger.info(f"✅ Backfill complete: {ready_pairs} pairs ready, {total_candles} total candles")
                
                # Save the backfilled data
                self._save_cache()
            else:
                logger.warning("⚠️ No historical data retrieved, will accumulate from live stream")
                
        except Exception as e:
            logger.warning(f"⚠️ Backfill failed: {e}. Continuing with live data only...")
        
        # Start streaming in background
        stream_task = asyncio.create_task(self._stream_swaps())
        
        # Main signal generation loop
        try:
            while True:
                # Wait for update interval
                await asyncio.sleep(LiveConfig.UPDATE_INTERVAL_SECONDS)
                
                # Get latest candles from streamer
                new_candles = self.streamer.get_latest_candles()
                
                if len(new_candles) > 0:
                    logger.info(f"Received {len(new_candles)} new candles")
                    self.update_candles_history(new_candles)
                    
                    # Periodic save to prevent data loss
                    if time.time() - self.last_save_time > LiveConfig.SAVE_INTERVAL_SECONDS:
                        self._save_cache()
                        self.last_save_time = time.time()
                    
                # Generate signals
                signals = self.generate_signals()
                
                if len(signals) > 0:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"🎯 SIGNALS GENERATED: {len(signals)}")
                    logger.info(f"{'='*80}")
                    
                    for row in signals.iter_rows(named=True):
                        logger.info(
                            f"📊 Pair: {row['pair_address']} | "
                            f"Confidence: {row['pred_proba']:.3f} | "
                            f"Price: ${row['close']:.6f} | "
                            f"Volume: ${row['volume_token1']:.2f}"
                        )
                    
                    # Save signals
                    self._save_signals(signals)
                else:
                    logger.info("No signals generated (threshold not met)")
                    
                # Show stats
                total_pairs = len(self.candles_history)
                total_candles = sum(len(df) for df in self.candles_history.values())
                logger.info(f"Status: {total_pairs} pairs tracked, {total_candles} total candles")
                
        except KeyboardInterrupt:
            logger.info("\nShutting down...")
            logger.info("💾 Saving cache before exit...")
            self._save_cache()
            stream_task.cancel()
            
    async def _stream_swaps(self):
        """Background task to stream swaps"""
        try:
            async for swap in self.streamer.stream_swaps():
                # Swaps are automatically added to aggregator
                pass
        except asyncio.CancelledError:
            logger.info("Swap streaming cancelled")
            
    def _save_signals(self, signals: pl.DataFrame):
        """Save signals to file"""
        if len(signals) == 0:
            return
            
        try:
            # Add timestamp as string
            now = datetime.now()
            signals = signals.with_columns([
                pl.lit(now.isoformat()).alias("generated_at")
            ])
            
            # Save to timestamped file for comparison
            timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            timestamped_path = Path(f"signals_{timestamp_str}.csv")
            signals.write_csv(timestamped_path)
            logger.info(f"💾 Signals saved to {timestamped_path}")
            
            # Also save to main signals_live.csv (append mode)
            output_path = Path(LiveConfig.SIGNALS_OUTPUT_PATH)
            if output_path.exists():
                # Read existing without auto-parsing dates
                existing = pl.read_csv(output_path, try_parse_dates=False)
                
                # Ensure all datetime columns in new signals are cast to string
                datetime_cols = [col for col, dtype in zip(signals.columns, signals.dtypes) 
                                if dtype in [pl.Datetime, pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]]
                if datetime_cols:
                    signals = signals.with_columns([
                        pl.col(col).cast(pl.Utf8) for col in datetime_cols
                    ])
                
                # Cast integer columns to match existing schema (Int64)
                int_cols = [col for col, dtype in zip(signals.columns, signals.dtypes) 
                           if dtype in [pl.Int32, pl.Int64]]
                if int_cols:
                    signals = signals.with_columns([
                        pl.col(col).cast(pl.Int64) for col in int_cols
                    ])
                
                # Cast float columns to match existing schema (Float64)
                float_cols = [col for col, dtype in zip(signals.columns, signals.dtypes) 
                             if dtype in [pl.Float32, pl.Float64]]
                if float_cols:
                    signals = signals.with_columns([
                        pl.col(col).cast(pl.Float64) for col in float_cols
                    ])
                
                combined = pl.concat([existing, signals])
                combined.write_csv(output_path)
                logger.info(f"📊 Appended {len(signals)} signals to {output_path} (total: {len(combined)})")
            else:
                signals.write_csv(output_path)
                logger.info(f"📊 Created {output_path} with {len(signals)} signals")
            
        except Exception as e:
            logger.error(f"Error saving signals: {e}", exc_info=True)
            
    def save_candles_history(self):
        """Save historical candles to disk"""
        try:
            all_candles = self.get_all_candles()
            if len(all_candles) > 0:
                output_path = Path(LiveConfig.HISTORICAL_CANDLES_PATH)
                all_candles.write_parquet(output_path)
                logger.info(f"💾 Saved {len(all_candles)} candles to {output_path}")
        except Exception as e:
            logger.error(f"Error saving candles: {e}", exc_info=True)


# ============================================================================
# CLI
# ============================================================================

async def main():
    """Run live signal generator"""
    
    generator = LiveSignalGenerator()
    
    try:
        await generator.run_live()
    finally:
        # Always save cache on exit, regardless of how we exit
        logger.info("\n💾 Saving cache before exit...")
        generator._save_cache()
        logger.info("✅ Cache saved successfully!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Goodbye!")
