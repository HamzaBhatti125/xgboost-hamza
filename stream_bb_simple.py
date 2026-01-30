#!/usr/bin/env python3
"""
Simple Bollinger Bands Streamer
================================

Streams data from Envio Hypersync and saves candles with Bollinger Bands to JSON.
Includes token0 and token1 addresses instead of pair addresses.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
import polars as pl

from envio_hypersync import LiveSwapStreamer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set API token
ENVIO_API_TOKEN = "1ae7c8f0-5cdf-4316-81f6-0fa3cb84aaa8"
os.environ["ENVIO_API_TOKEN"] = ENVIO_API_TOKEN

# Load pair universe to get token addresses
def load_token_addresses():
    """Load token0 and token1 addresses from pair universe"""
    try:
        df = pl.read_parquet("./Files/pair-universe")
        # Filter for Base chain (8453)
        df_base = df.filter(pl.col('chain_id') == 8453)
        token_map = {}
        for row in df_base.iter_rows(named=True):
            pair_addr = str(row.get('address', '')).lower().strip()
            token0 = str(row.get('token0_address', '')).strip()
            token1 = str(row.get('token1_address', '')).strip()
            if pair_addr and token0 and token1 and token0 != 'None' and token1 != 'None':
                token_map[pair_addr] = {
                    'token0': token0,
                    'token1': token1
                }
        logger.info(f"✅ Loaded token addresses for {len(token_map)} Base chain pairs")
        return token_map
    except Exception as e:
        logger.warning(f"⚠️  Could not load pair universe: {e}")
        import traceback
        traceback.print_exc()
        return {}


async def main():
    """Stream and save Bollinger Bands data continuously"""
    output_file = "bollinger_bands.json"
    
    logger.info("="*80)
    logger.info("BOLLINGER BANDS STREAMER (CONTINUOUS)")
    logger.info("="*80)
    logger.info(f"Output: {output_file}")
    logger.info("Running continuously - use Ctrl+C to stop")
    logger.info("="*80)
    
    # Load token address mapping
    token_map = load_token_addresses()
    
    # Initialize streamer with BB enabled
    streamer = LiveSwapStreamer(calculate_bollinger_bands=True)
    
    # Backfill historical data
    logger.info("\n📥 Backfilling 25 hours of historical data...")
    historical_df = await streamer.backfill_historical_candles(hours=25)
    logger.info(f"✅ Backfilled {len(historical_df)} candles")
    
    # Collect all candles with BB
    all_candles = []
    
    # Process historical candles - get from aggregator which has BB calculated
    logger.info(f"\n📊 Processing historical candles for Bollinger Bands...")
    historical_candles_df = streamer.get_latest_candles()
    
    if len(historical_candles_df) > 0:
        for row in historical_candles_df.iter_rows(named=True):
            if row.get('bb_middle') is not None:
                pair_addr = str(row['pair_address']).lower()
                token_info = token_map.get(pair_addr, {'token0': '', 'token1': ''})
                
                candle_dict = {
                    "token0": token_info['token0'],
                    "token1": token_info['token1'],
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
        logger.info(f"✅ Added {len(all_candles)} historical candles with BB")
        
        # Save initial batch
        save_to_file(all_candles, output_file)
        logger.info(f"💾 Saved initial {len(all_candles)} candles to {output_file}")
    
    # Stream live data continuously
    logger.info("\n🔄 Streaming live data continuously...")
    swap_count = 0
    save_interval = 50  # Save every 50 new candles
    
    try:
        async for swap in streamer.stream_swaps():
            swap_count += 1
            
            # Check every 100 swaps
            if swap_count % 100 == 0:
                candles_df = streamer.get_latest_candles()
                
                # Add new candles with BB
                new_count = 0
                for row in candles_df.iter_rows(named=True):
                    if row.get('bb_middle') is not None:
                        # Check if already added
                        ts = row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
                        pair_addr = str(row['pair_address']).lower()
                        token_info = token_map.get(pair_addr, {'token0': '', 'token1': ''})
                        
                        existing = [c for c in all_candles 
                                  if c['token0'] == token_info['token0']
                                  and c['token1'] == token_info['token1']
                                  and c['timestamp'] == ts]
                        
                        if not existing:
                            candle_dict = {
                                "token0": token_info['token0'],
                                "token1": token_info['token1'],
                                "timestamp": ts,
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
                            new_count += 1
                            
                            if new_count <= 3:  # Log first few
                                logger.info(f"📊 Candle: {token_info['token0'][:20]}.../{token_info['token1'][:20]}... | "
                                          f"Close: {row['close']:.6f} | "
                                          f"BB: [{row['bb_lower']:.6f}, {row['bb_middle']:.6f}, {row['bb_upper']:.6f}]")
                
                logger.info(f"   Processed {swap_count} swaps, {len(all_candles)} total candles ({new_count} new)")
                
                # Save periodically (every 20 new candles or every 500 swaps)
                if new_count >= 20 or swap_count % 500 == 0:
                    save_to_file(all_candles, output_file)
                    logger.info(f"💾 Saved {len(all_candles)} candles to {output_file}")
                
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted")
    finally:
        # Final collection
        logger.info("\n📊 Collecting final candles...")
        final_df = streamer.get_latest_candles()
        for row in final_df.iter_rows(named=True):
            if row.get('bb_middle') is not None:
                ts = row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
                pair_addr = str(row['pair_address']).lower()
                token_info = token_map.get(pair_addr, {'token0': '', 'token1': ''})
                
                existing = [c for c in all_candles 
                          if c['token0'] == token_info['token0']
                          and c['token1'] == token_info['token1']
                          and c['timestamp'] == ts]
                if not existing:
                    candle_dict = {
                        "token0": token_info['token0'],
                        "token1": token_info['token1'],
                        "timestamp": ts,
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
        
        # Final save to JSON
        save_to_file(all_candles, output_file)
        
        logger.info("="*80)
        logger.info("SUMMARY")
        logger.info("="*80)
        logger.info(f"Total candles with BB: {len(all_candles)}")
        logger.info(f"Saved to: {output_file}")
        logger.info("="*80)


def save_to_file(candles, output_file):
    """Save candles to JSON file"""
    logger.info(f"💾 Saving {len(candles)} candles to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(candles, f, indent=2)
    logger.info(f"✅ Saved successfully")


if __name__ == "__main__":
    asyncio.run(main())
