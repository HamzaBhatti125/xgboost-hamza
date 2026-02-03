#!/usr/bin/env python3
"""
Quick test to verify Envio Hypersync connection is working
"""

import asyncio
import os
import sys
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set API token
os.environ["ENVIO_API_TOKEN"] = "1ae7c8f0-5cdf-4316-81f6-0fa3cb84aaa8"

from envio_hypersync import EnvioHypersyncClient, EnvioConfig

async def test_envio():
    """Test Envio Hypersync connection"""
    print("=" * 80)
    print("ENVIO HYPERSYNC CONNECTION TEST")
    print("=" * 80)
    print()
    
    # Test 1: Check API token
    print("1. Checking API token...")
    api_token = os.getenv("ENVIO_API_TOKEN")
    if api_token:
        print(f"   ✅ API token found: {api_token[:20]}...")
    else:
        print("   ❌ API token not found!")
        return False
    print()
    
    # Test 2: Initialize client
    print("2. Initializing Hypersync client...")
    try:
        client = EnvioHypersyncClient()
        print(f"   ✅ Client initialized")
        print(f"   📍 URL: {EnvioConfig.HYPERSYNC_URL}")
        print(f"   🔗 Chain ID: {EnvioConfig.CHAIN_ID} (Base)")
    except Exception as e:
        print(f"   ❌ Failed to initialize client: {e}")
        return False
    print()
    
    # Test 3: Get latest block
    print("3. Querying latest block...")
    try:
        latest_block = await client.get_current_block()
        print(f"   ✅ Latest block: {latest_block:,}")
        print(f"   📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"   ❌ Failed to get latest block: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Test 4: Query recent swaps (small range)
    print("4. Querying recent swap events (last 1000 blocks)...")
    try:
        start_block = max(1, latest_block - 1000)
        end_block = latest_block
        
        print(f"   📊 Querying blocks {start_block:,} to {end_block:,}...")
        
        # Query for V2 swaps using query_logs
        from envio_hypersync import SwapEventProcessor
        processor = SwapEventProcessor()
        
        v2_logs = await client.query_logs(
            from_block=start_block,
            to_block=end_block,
            topics=[EnvioConfig.SWAP_V2_TOPIC]
        )
        
        print(f"   ✅ Found {len(v2_logs)} V2 swap log events")
        
        if len(v2_logs) > 0:
            # Decode first swap
            sample_swap = processor.decode_v2_swap(v2_logs[0])
            if sample_swap:
                print(f"   📈 Sample swap:")
                print(f"      - Block: {sample_swap.block_number}")
                print(f"      - Pair: {sample_swap.pair_address[:20]}...")
                print(f"      - Amount0: {sample_swap.amount0:.6f}")
                print(f"      - Amount1: {sample_swap.amount1:.6f}")
                print(f"      - Price: {sample_swap.price:.6f}")
        
    except Exception as e:
        print(f"   ❌ Failed to query swaps: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Test 5: Check pair universe
    print("5. Checking pair universe...")
    try:
        from envio_hypersync import SwapEventProcessor
        processor = SwapEventProcessor()
        pairs = processor.pair_info
        print(f"   ✅ Loaded {len(pairs):,} pairs from universe")
        if len(pairs) > 0:
            sample_pair = list(pairs.items())[0]
            print(f"   📊 Sample pair: {sample_pair[0][:20]}...")
    except Exception as e:
        print(f"   ⚠️  Could not load pair universe: {e}")
    print()
    
    print("=" * 80)
    print("✅ ALL TESTS PASSED - ENVIO IS WORKING!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(test_envio())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
