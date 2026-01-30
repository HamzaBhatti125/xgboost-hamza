#!/usr/bin/env python3
"""
Quick Backfill and Signal Generation
====================================

Backfills 3 days of historical data to accumulate enough candles,
then immediately generates signals without waiting.
"""

import asyncio
import polars as pl
import os
from pathlib import Path
import logging

# Set API token
os.environ["ENVIO_API_TOKEN"] = "1ae7c8f0-5cdf-4316-81f6-0fa3cb84aaa8"

from envio_hypersync import LiveSwapStreamer, EnvioConfig
from live_signal_generator import LiveSignalGenerator, LiveConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Backfill data and generate signals immediately"""
    
    logger.info("="*80)
    logger.info("QUICK BACKFILL AND SIGNAL GENERATION")
    logger.info("="*80)
    logger.info("Backfilling 72 hours (3 days) of data to accumulate enough candles...")
    logger.info("="*80)
    
    # Initialize generator
    generator = LiveSignalGenerator()
    
    # Load model
    try:
        generator.load_model()
        logger.info("✅ Model loaded")
    except Exception as e:
        logger.error(f"❌ Could not load model: {e}")
        return
    
    # Large backfill - 72 hours (3 days) to ensure 20+ candles per pair
    logger.info("\n" + "="*80)
    logger.info("STEP 1: BACKFILLING HISTORICAL DATA (72 hours)")
    logger.info("="*80)
    
    try:
        historical_candles = await generator.streamer.backfill_historical_candles(hours=72)
        
        if len(historical_candles) > 0:
            logger.info(f"📦 Received {len(historical_candles)} historical candles")
            generator.update_candles_history(historical_candles)
            
            # Check readiness
            ready_pairs = sum(1 for df in generator.candles_history.values() 
                            if len(df) >= LiveConfig.MIN_HISTORY_CANDLES)
            total_candles = sum(len(df) for df in generator.candles_history.values())
            
            logger.info(f"✅ Backfill complete: {ready_pairs} pairs ready, {total_candles} total candles")
            
            # Show distribution
            candle_counts = [(p, len(df)) for p, df in generator.candles_history.items()]
            candle_counts.sort(key=lambda x: x[1], reverse=True)
            if candle_counts:
                max_candles = candle_counts[0][1]
                avg_candles = sum(c for _, c in candle_counts) / len(candle_counts)
                logger.info(f"   Max candles per pair: {max_candles}")
                logger.info(f"   Avg candles per pair: {avg_candles:.1f}")
                logger.info(f"   Pairs with 20+ candles: {ready_pairs}")
            
            # Save cache
            generator._save_cache()
            logger.info("💾 Cache saved")
        else:
            logger.warning("⚠️ No historical data retrieved")
            
    except Exception as e:
        logger.error(f"❌ Backfill failed: {e}", exc_info=True)
        return
    
    # Generate signals immediately
    logger.info("\n" + "="*80)
    logger.info("STEP 2: GENERATING SIGNALS")
    logger.info("="*80)
    
    # Generate OHLCV signals
    logger.info("\n📊 Generating OHLCV signals (XGBoost)...")
    ohlcv_signals = generator.generate_signals()
    
    if len(ohlcv_signals) > 0:
        logger.info(f"✅ Generated {len(ohlcv_signals)} OHLCV signals")
    else:
        logger.warning("⚠️ No OHLCV signals generated (threshold not met)")
    
    # Generate BB signals
    logger.info("\n📊 Generating Bollinger Band signals...")
    bb_signals = generator.generate_bb_signals()
    
    if len(bb_signals) > 0:
        logger.info(f"✅ Generated {len(bb_signals)} BB signals")
    else:
        logger.warning("⚠️ No BB signals generated")
    
    # Combine and save
    all_signals = []
    if len(ohlcv_signals) > 0:
        all_signals.append(ohlcv_signals)
    if len(bb_signals) > 0:
        all_signals.append(bb_signals)
    
    if all_signals:
        # Normalize types and use diagonal concat (handles different columns automatically)
        try:
            # Cast all float columns to Float64 to avoid type mismatches
            normalized_signals = []
            for sig_df in all_signals:
                # Cast all float columns to Float64
                float_cols = [col for col, dtype in zip(sig_df.columns, sig_df.dtypes) 
                             if dtype in [pl.Float32, pl.Float64]]
                if float_cols:
                    sig_df = sig_df.with_columns([
                        pl.col(col).cast(pl.Float64) for col in float_cols
                    ])
                
                # Cast all int columns to Int64
                int_cols = [col for col, dtype in zip(sig_df.columns, sig_df.dtypes) 
                           if dtype in [pl.Int32, pl.Int64]]
                if int_cols:
                    sig_df = sig_df.with_columns([
                        pl.col(col).cast(pl.Int64) for col in int_cols
                    ])
                
                normalized_signals.append(sig_df)
            
            # Use diagonal concat which handles different column schemas
            combined_signals = pl.concat(normalized_signals, how='diagonal')
            
        except Exception as e:
            logger.warning(f"Concatenation failed: {e}, saving separately...")
            # Fallback: save separately if concat fails
            if len(ohlcv_signals) > 0:
                generator._save_signals(ohlcv_signals)
            if len(bb_signals) > 0:
                generator._save_signals(bb_signals)
            logger.info("Signals saved separately due to schema mismatch")
            return
        
        logger.info("\n" + "="*80)
        logger.info(f"🎯 TOTAL SIGNALS: {len(combined_signals)}")
        logger.info(f"   - OHLCV: {len(ohlcv_signals)}")
        logger.info(f"   - BB: {len(bb_signals)}")
        logger.info("="*80)
        
        # Save signals
        generator._save_signals(combined_signals)
        
        logger.info("\n✅ Signals saved to signals_live.csv")
        logger.info("   You can now view the signals!")
    else:
        logger.warning("\n⚠️ No signals generated. Possible reasons:")
        logger.warning("   - Not enough candles per pair (need 20+)")
        logger.warning("   - Prediction probabilities below threshold (0.32)")
        logger.warning("   - No BB data available")
    
    logger.info("\n" + "="*80)
    logger.info("COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nInterrupted")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
