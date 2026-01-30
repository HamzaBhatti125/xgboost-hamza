#!/usr/bin/env python3
"""
Stream Bollinger Bands from Envio Hypersync
===========================================

Connects to Envio Hypersync, streams swap data, creates OHLCV candles with
Bollinger Bands, and logs the results to a JSON file.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import polars as pl

from envio_hypersync import LiveSwapStreamer, EnvioConfig
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Set API token
ENVIO_API_TOKEN = "1ae7c8f0-5cdf-4316-81f6-0fa3cb84aaa8"
os.environ["ENVIO_API_TOKEN"] = ENVIO_API_TOKEN


def candle_to_dict(candle) -> Dict:
    """Convert candle to dictionary for JSON serialization"""
    return {
        "pair_address": candle.pair_address,
        "timestamp": candle.timestamp,
        "datetime": datetime.fromtimestamp(candle.timestamp).isoformat(),
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume_token0": candle.volume_token0,
        "volume_token1": candle.volume_token1,
        "num_trades": candle.num_trades,
        "bb_middle": candle.bb_middle,
        "bb_upper": candle.bb_upper,
        "bb_lower": candle.bb_lower,
        "bb_width": candle.bb_width,
        "bb_position": candle.bb_position,
    }


async def stream_and_log_bollinger_bands(
    output_file: str = "bollinger_bands.json",
    max_candles: int = 50,
    stream_duration_minutes: int = 5
):
    """
    Stream swap data, generate candles with Bollinger Bands, and log to JSON
    
    Args:
        output_file: Path to output JSON file
        max_candles: Maximum number of candles to collect before saving
        stream_duration_minutes: How long to stream (minutes)
    """
    logger.info("="*80)
    logger.info("BOLLINGER BANDS STREAMER")
    logger.info("="*80)
    logger.info(f"API Token: {ENVIO_API_TOKEN[:20]}...")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Max candles per save: {max_candles}")
    logger.info(f"Stream duration: {stream_duration_minutes} minutes")
    logger.info("="*80)
    
    # Initialize streamer with Bollinger Bands enabled
    streamer = LiveSwapStreamer(calculate_bollinger_bands=True)
    
    # Backfill some historical data first (to have enough for BB calculation)
    logger.info("\n📥 Backfilling historical data (25 hours) for Bollinger Bands...")
    historical_df = await streamer.backfill_historical_candles(hours=25)
    
    # Process historical candles to populate BB calculator
    if len(historical_df) > 0:
        logger.info(f"✅ Backfilled {len(historical_df)} historical candles")
        logger.info(f"   Date range: {historical_df['timestamp'].min()} to {historical_df['timestamp'].max()}")
        
        # Add historical candles to BB calculator
        from envio_hypersync import Candle
        historical_candles = []
        for row in historical_df.iter_rows(named=True):
            candle = Candle(
                pair_address=row['pair_address'],
                timestamp=int(row['timestamp'].timestamp()) if hasattr(row['timestamp'], 'timestamp') else int(row['timestamp']),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume_token0=float(row['volume_token0']),
                volume_token1=float(row['volume_token1']),
                num_trades=int(row['num_trades'])
            )
            historical_candles.append(candle)
        
        # Calculate BB for historical candles
        if streamer.aggregator.bb_calculator:
            logger.info(f"📊 Calculating Bollinger Bands for {len(historical_candles)} historical candles...")
            historical_candles_with_bb = streamer.aggregator.bb_calculator.calculate_bands_batch(historical_candles)
            
            # Count how many have BB calculated
            bb_count = sum(1 for c in historical_candles_with_bb if c.bb_middle is not None)
            logger.info(f"✅ {bb_count} historical candles now have Bollinger Bands")
            
            # Save historical candles with BB to output
            for candle in historical_candles_with_bb:
                if candle.bb_middle is not None:
                    candle_dict = {
                        "pair_address": candle.pair_address,
                        "timestamp": datetime.fromtimestamp(candle.timestamp).isoformat(),
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close,
                        "volume_token0": candle.volume_token0,
                        "volume_token1": candle.volume_token1,
                        "num_trades": candle.num_trades,
                        "bb_middle": candle.bb_middle,
                        "bb_upper": candle.bb_upper,
                        "bb_lower": candle.bb_lower,
                        "bb_width": candle.bb_width,
                        "bb_position": candle.bb_position,
                    }
                    all_candles.append(candle_dict)
    else:
        logger.warning("⚠️  No historical data backfilled")
    
    # Collect candles
    all_candles = []
    swap_count = 0
    candle_count = 0
    start_time = datetime.now()
    end_time = start_time.replace(minute=start_time.minute + stream_duration_minutes)
    
    logger.info(f"\n🔄 Starting live stream (until {end_time.strftime('%H:%M:%S')})...")
    logger.info("   Waiting for swaps to generate candles...\n")
    
    try:
        async for swap in streamer.stream_swaps():
            swap_count += 1
            
            # Check if we should stop
            if datetime.now() >= end_time:
                logger.info(f"\n⏰ Stream duration reached ({stream_duration_minutes} minutes)")
                break
            
            # Every 50 swaps, check for new candles
            if swap_count % 50 == 0:
                candles_df = streamer.get_latest_candles()
                
                if len(candles_df) > candle_count:
                    new_candles = candles_df[candle_count:]
                    candle_count = len(candles_df)
                    
                    # Convert to list of dicts
                    for row in new_candles.iter_rows(named=True):
                        # Check if Bollinger Bands are available
                        if row.get('bb_middle') is not None:
                            candle_dict = {
                                "pair_address": row['pair_address'],
                                "timestamp": row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                                "open": float(row['open']),
                                "high": float(row['high']),
                                "low": float(row['low']),
                                "close": float(row['close']),
                                "volume_token0": float(row['volume_token0']),
                                "volume_token1": float(row['volume_token1']),
                                "num_trades": int(row['num_trades']),
                                "bb_middle": float(row['bb_middle']),
                                "bb_upper": float(row['bb_upper']),
                                "bb_lower": float(row['bb_lower']),
                                "bb_width": float(row['bb_width']),
                                "bb_position": float(row['bb_position']),
                            }
                            all_candles.append(candle_dict)
                            
                            logger.info(f"📊 Candle #{len(all_candles)}: {row['pair_address'][:20]}... | "
                                      f"Close: {row['close']:.6f} | "
                                      f"BB: [{row['bb_lower']:.6f}, {row['bb_middle']:.6f}, {row['bb_upper']:.6f}] | "
                                      f"Position: {row['bb_position']:.2%}")
                    
                    # Save periodically
                    if len(all_candles) >= max_candles:
                        save_to_json(all_candles, output_file)
                        logger.info(f"💾 Saved {len(all_candles)} candles to {output_file}")
                        all_candles = []  # Clear after saving
                
                logger.info(f"   Processed {swap_count} swaps, {candle_count} candles generated")
    
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error during streaming: {e}", exc_info=True)
    finally:
        # Get final candles
        logger.info("\n📊 Collecting final candles...")
        final_candles_df = streamer.get_latest_candles()
        
        if len(final_candles_df) > 0:
            for row in final_candles_df.iter_rows(named=True):
                if row.get('bb_middle') is not None:
                    # Check if already added
                    existing = [c for c in all_candles if c['pair_address'] == row['pair_address'] 
                               and c['timestamp'] == str(row['timestamp'])]
                    if not existing:
                        candle_dict = {
                            "pair_address": row['pair_address'],
                            "timestamp": row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                            "open": float(row['open']),
                            "high": float(row['high']),
                            "low": float(row['low']),
                            "close": float(row['close']),
                            "volume_token0": float(row['volume_token0']),
                            "volume_token1": float(row['volume_token1']),
                            "num_trades": int(row['num_trades']),
                            "bb_middle": float(row['bb_middle']),
                            "bb_upper": float(row['bb_upper']),
                            "bb_lower": float(row['bb_lower']),
                            "bb_width": float(row['bb_width']),
                            "bb_position": float(row['bb_position']),
                        }
                        all_candles.append(candle_dict)
        
        # Final save
        if all_candles:
            save_to_json(all_candles, output_file, append=True)
            logger.info(f"💾 Final save: {len(all_candles)} candles saved to {output_file}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("STREAMING SUMMARY")
        logger.info("="*80)
        logger.info(f"Total swaps processed: {swap_count}")
        logger.info(f"Total candles with BB: {len(all_candles)}")
        logger.info(f"Output file: {output_file}")
        logger.info("="*80)


def save_to_json(candles: List[Dict], output_file: str, append: bool = False):
    """Save candles to JSON file"""
    output_path = Path(output_file)
    
    if append and output_path.exists():
        # Load existing data
        with open(output_path, 'r') as f:
            existing = json.load(f)
        # Merge (avoid duplicates)
        existing_addresses = {(c['pair_address'], c['timestamp']) for c in existing}
        new_candles = [c for c in candles 
                      if (c['pair_address'], c['timestamp']) not in existing_addresses]
        candles = existing + new_candles
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(candles, f, indent=2)
    
    logger.info(f"💾 Saved {len(candles)} candles to {output_file}")


async def main():
    """Main entry point"""
    output_file = "bollinger_bands.json"
    
    logger.info("Starting Bollinger Bands streamer...")
    logger.info(f"Output will be saved to: {output_file}\n")
    
    await stream_and_log_bollinger_bands(
        output_file=output_file,
        max_candles=20,  # Save every 20 candles
        stream_duration_minutes=5  # Stream for 5 minutes
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
