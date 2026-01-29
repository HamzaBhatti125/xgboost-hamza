#!/usr/bin/env python3
"""
Envio HyperSync Data Streamer
==============================

Streams real-time DEX swap data from Envio HyperSync API
and converts it to candle format for XGBoost signal generation.
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import polars as pl
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class SwapEvent:
    """Standardized swap event"""
    block_number: int
    block_timestamp: int
    block_hash: str
    transaction_hash: str
    log_index: int
    pool_address: str
    dex_name: str
    swap_type: str
    sell_token_address: str
    buy_token_address: str
    sell_amount: str
    buy_amount: str
    sender: str
    recipient: str
    data: str = ""  # Raw log data for price decoding


class EnvioHyperSyncClient:
    """Python client for Envio HyperSync API"""
    
    # Swap event signatures
    SWAP_V2_SIGNATURE = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
    SWAP_V3_SIGNATURE = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
    
    def __init__(self, api_token: str, chain: str = "base", rpc_url: Optional[str] = None):
        self.api_token = api_token
        self.chain = chain.lower()
        self.rpc_url = rpc_url or "https://base.llamarpc.com"
        
        # API URL based on chain
        self.api_url = self._get_api_url(self.chain)
        
        # Session for HTTP requests
        self.session = requests.Session()
        headers = {
            "Content-Type": "application/json"
        }
        if api_token:
            # Use Bearer token authentication (as per Envio API)
            headers["Authorization"] = f"Bearer {api_token}"
        self.session.headers.update(headers)
        
        # Cache for token info
        self.token_cache = {}
        self.invalid_pools = set()
        
    def _get_api_url(self, chain: str) -> str:
        """Get HyperSync API URL for chain"""
        urls = {
            "base": "https://base.hypersync.xyz",
            "eth": "https://eth.hypersync.xyz",
            "ethereum": "https://eth.hypersync.xyz",
            "arbitrum": "https://arbitrum.hypersync.xyz",
            "optimism": "https://optimism.hypersync.xyz",
            "polygon": "https://polygon.hypersync.xyz",
        }
        return urls.get(chain, urls["base"])
    
    def get_archive_height(self) -> int:
        """Get the latest archived block height"""
        query = {
            "from_block": 0,
            "to_block": 1,
            "logs": [{}],
            "field_selection": {
                "block": ["number"]
            }
        }
        
        try:
            response = self.session.post(f"{self.api_url}/query", json=query)
            response.raise_for_status()
            data = response.json()
            return data.get("archive_height", 0)
        except Exception as e:
            print(f"Error getting archive height: {e}")
            return 0
    
    def query_swaps(self, from_block: int, to_block: int) -> Dict:
        """Query swap events from HyperSync"""
        query = {
            "from_block": from_block,
            "to_block": to_block,
            "logs": [
                {"topics": [[self.SWAP_V2_SIGNATURE]]},
                {"topics": [[self.SWAP_V3_SIGNATURE]]}
            ],
            "field_selection": {
                "block": ["number", "timestamp", "hash"],
                "log": [
                    "block_number",
                    "log_index",
                    "transaction_hash",
                    "address",
                    "data",
                    "topic0",
                    "topic1",
                    "topic2",
                    "topic3"
                ]
            }
        }
        
        try:
            response = self.session.post(f"{self.api_url}/query", json=query)
            response.raise_for_status()
            result = response.json()
            # Handle response structure - data is a list
            if isinstance(result.get("data"), list) and len(result["data"]) > 0:
                # If data is a list, extract the first item which should have blocks and logs
                data_item = result["data"][0]
                if isinstance(data_item, dict):
                    return {
                        "data": {
                            "blocks": data_item.get("blocks", []),
                            "logs": data_item.get("logs", [])
                        },
                        "archive_height": result.get("archive_height", 0)
                    }
            # If data is already in the right format
            if isinstance(result.get("data"), dict):
                return result
            return {"data": {"blocks": [], "logs": []}, "archive_height": result.get("archive_height", 0)}
        except Exception as e:
            print(f"Error querying swaps: {e}")
            return {"data": {"blocks": [], "logs": []}}
    
    def decode_swap_v2(self, log: Dict) -> Optional[SwapEvent]:
        """Decode Uniswap V2 swap event"""
        try:
            # Extract topics - handle both snake_case and camelCase
            topic0 = (log.get("topic0") or log.get("Topic0", "")).lower()
            if topic0 != self.SWAP_V2_SIGNATURE.lower():
                return None
            
            topic1 = log.get("topic1") or log.get("Topic1", "")
            topic2 = log.get("topic2") or log.get("Topic2", "")
            
            sender = "0x" + topic1[-40:] if len(topic1) >= 40 else ""
            recipient = "0x" + topic2[-40:] if len(topic2) >= 40 else ""
            
            # Decode data (amount0In, amount1In, amount0Out, amount1Out)
            data = log.get("data") or log.get("Data", "")
            if not data or len(data) < 256:
                return None
            
            # Simple hex decoding (full implementation would use eth_abi)
            # For now, we'll extract basic info
            pool_address = log.get("address") or log.get("Address", "")
            block_number = log.get("block_number") or log.get("BlockNumber", 0)
            
            data = log.get("data") or log.get("Data", "")
            
            return SwapEvent(
                block_number=int(block_number),
                block_timestamp=0,  # Will be filled from block data
                block_hash="",
                transaction_hash=log.get("transaction_hash") or log.get("TransactionHash", ""),
                log_index=int(log.get("log_index") or log.get("LogIndex", 0)),
                pool_address=pool_address,
                dex_name="Uniswap V2",
                swap_type="v2",
                sell_token_address="",  # Would need pool contract call
                buy_token_address="",
                sell_amount="0",
                buy_amount="0",
                sender=sender,
                recipient=recipient,
                data=data  # Store raw data for decoding
            )
        except Exception as e:
            print(f"Error decoding V2 swap: {e}")
            return None
    
    def decode_swap_v3(self, log: Dict) -> Optional[SwapEvent]:
        """Decode Uniswap V3 swap event"""
        try:
            # Handle both snake_case and camelCase
            topic0 = (log.get("topic0") or log.get("Topic0", "")).lower()
            if topic0 != self.SWAP_V3_SIGNATURE.lower():
                return None
            
            topic1 = log.get("topic1") or log.get("Topic1", "")
            topic2 = log.get("topic2") or log.get("Topic2", "")
            
            sender = "0x" + topic1[-40:] if len(topic1) >= 40 else ""
            recipient = "0x" + topic2[-40:] if len(topic2) >= 40 else ""
            
            pool_address = log.get("address") or log.get("Address", "")
            block_number = log.get("block_number") or log.get("BlockNumber", 0)
            
            data = log.get("data") or log.get("Data", "")
            
            return SwapEvent(
                block_number=int(block_number),
                block_timestamp=0,
                block_hash="",
                transaction_hash=log.get("transaction_hash") or log.get("TransactionHash", ""),
                log_index=int(log.get("log_index") or log.get("LogIndex", 0)),
                pool_address=pool_address,
                dex_name="Uniswap V3",
                swap_type="v3",
                sell_token_address="",
                buy_token_address="",
                sell_amount="0",
                buy_amount="0",
                sender=sender,
                recipient=recipient,
                data=data  # Store raw data for decoding
            )
        except Exception as e:
            print(f"Error decoding V3 swap: {e}")
            return None


class SwapToCandleConverter:
    """Converts swap events to candle format"""
    
    def __init__(self):
        self.swap_buffer = defaultdict(list)  # pool_address -> swaps
        
    def add_swap(self, swap: SwapEvent, block_timestamp: int):
        """Add swap to buffer"""
        swap.block_timestamp = block_timestamp
        self.swap_buffer[swap.pool_address].append(swap)
    
    def aggregate_to_candles(self, window_seconds: int = 60) -> pl.DataFrame:
        """Aggregate swaps to 1-minute candles"""
        candles = []
        
        for pool_address, swaps in self.swap_buffer.items():
            if not swaps:
                continue
            
            # Group by time window
            time_windows = defaultdict(list)
            for swap in swaps:
                # Round to nearest minute
                window_time = (swap.block_timestamp // window_seconds) * window_seconds
                time_windows[window_time].append(swap)
            
            # Create candles for each time window
            for window_time, window_swaps in time_windows.items():
                # Count buys/sells (simplified - would need token info for accurate classification)
                buys = len([s for s in window_swaps if s.buy_amount != "0"])
                sells = len([s for s in window_swaps if s.sell_amount != "0"])
                total_trades = len(window_swaps)
                
                # Estimate volume (simplified)
                volume = total_trades * 1000  # Placeholder
                buy_volume = buys * 1000
                sell_volume = sells * 1000
                
                candle = {
                    "pair_id": hash(pool_address) % 1000000,  # Simplified pair_id
                    "timestamp": datetime.fromtimestamp(window_time),
                    "open": 1.0,  # Placeholder - would need price data
                    "close": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "volume": volume,
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "buys": buys,
                    "sells": sells,
                    "pool_address": pool_address
                }
                candles.append(candle)
        
        if not candles:
            return pl.DataFrame()
        
        df = pl.DataFrame(candles)
        return df.sort("timestamp")
    
    def clear_buffer(self):
        """Clear swap buffer"""
        self.swap_buffer.clear()


class HyperSyncStreamer:
    """Main streaming class that collects data and triggers signal generation"""
    
    def __init__(self, api_token: str, chain: str = "base", signal_callback: Optional[Callable] = None):
        self.client = EnvioHyperSyncClient(api_token, chain)
        self.converter = SwapToCandleConverter()
        self.signal_callback = signal_callback
        self.is_running = False
        self.current_block = 0
        self.start_time = None
        self.swap_count = 0
        
    def start_streaming(self, duration_seconds: int = 60, batch_size: int = 500):
        """Start streaming for specified duration"""
        print(f"\n{'='*80}")
        print("ENVIO HYPERSYNC STREAMING")
        print(f"{'='*80}")
        print(f"Chain: {self.client.chain}")
        print(f"API URL: {self.client.api_url}")
        print(f"Streaming duration: {duration_seconds} seconds")
        print(f"{'='*80}\n")
        
        # Get starting block
        archive_height = self.client.get_archive_height()
        self.current_block = archive_height + 1
        self.start_time = time.time()
        self.is_running = True
        
        print(f"Starting from block: {self.current_block:,}")
        print(f"Archive height: {archive_height:,}\n")
        
        while self.is_running:
            elapsed = time.time() - self.start_time
            
            if elapsed >= duration_seconds:
                # Time to generate signals
                print(f"\n⏰ {duration_seconds} seconds elapsed. Generating signals...")
                self._generate_signals()
                break
            
            try:
                # Query next batch of blocks
                to_block = self.current_block + batch_size
                result = self.client.query_swaps(self.current_block, to_block)
                
                blocks = result.get("data", {}).get("blocks", [])
                logs = result.get("data", {}).get("logs", [])
                
                if blocks:
                    # Process blocks and logs
                    for block in blocks:
                        # Handle both snake_case and camelCase field names
                        # Block number might be hex string or int
                        block_num_raw = block.get("number") or block.get("Number", 0)
                        if isinstance(block_num_raw, str):
                            # Convert hex to int
                            block_num = int(block_num_raw, 16) if block_num_raw.startswith("0x") else int(block_num_raw)
                        else:
                            block_num = int(block_num_raw)
                        
                        # Timestamp might be hex or int
                        block_timestamp_raw = block.get("timestamp") or block.get("Timestamp", 0)
                        if isinstance(block_timestamp_raw, str):
                            block_timestamp = int(block_timestamp_raw, 16) if block_timestamp_raw.startswith("0x") else int(block_timestamp_raw)
                        else:
                            block_timestamp = int(block_timestamp_raw)
                        
                        block_hash = block.get("hash") or block.get("Hash", "")
                        
                        # Get logs for this block
                        block_logs = []
                        for log in logs:
                            log_block_raw = log.get("block_number") or log.get("BlockNumber", 0)
                            if isinstance(log_block_raw, str):
                                log_block_num = int(log_block_raw, 16) if log_block_raw.startswith("0x") else int(log_block_raw)
                            else:
                                log_block_num = int(log_block_raw)
                            if log_block_num == block_num:
                                block_logs.append(log)
                        
                        # Process swaps
                        for log in block_logs:
                            # Handle both snake_case and camelCase
                            topic0 = (log.get("topic0") or log.get("Topic0", "")).lower()
                            
                            swap = None
                            if topic0 == self.client.SWAP_V2_SIGNATURE.lower():
                                swap = self.client.decode_swap_v2(log)
                            elif topic0 == self.client.SWAP_V3_SIGNATURE.lower():
                                swap = self.client.decode_swap_v3(log)
                            
                            if swap:
                                swap.block_timestamp = block_timestamp
                                swap.block_hash = block_hash
                                self.converter.add_swap(swap, block_timestamp)
                                self.swap_count += 1
                        
                        self.current_block = block_num + 1
                    
                    if self.swap_count > 0 and self.swap_count % 100 == 0:
                        print(f"  Processed {self.swap_count:,} swaps | "
                              f"Block: {self.current_block:,} | "
                              f"Elapsed: {elapsed:.1f}s")
                else:
                    # No new blocks, wait a bit
                    time.sleep(2)
                    
            except Exception as e:
                print(f"Error in streaming loop: {e}")
                time.sleep(5)
        
        print(f"\n✓ Streaming complete. Total swaps: {self.swap_count:,}")
        return self.converter.aggregate_to_candles()
    
    def _generate_signals(self):
        """Generate signals from collected data"""
        if self.signal_callback:
            candles = self.converter.aggregate_to_candles()
            if len(candles) > 0:
                print(f"\n📊 Generated {len(candles)} candles from swaps")
                self.signal_callback(candles)
            else:
                print("⚠️  No candles generated from swaps")
    
    def stop(self):
        """Stop streaming"""
        self.is_running = False


if __name__ == "__main__":
    # Test the streamer
    API_TOKEN = "1ae7c8f0-5cdf-4316-81f6-0fa3cb84aaa8"
    
    def test_callback(candles):
        print(f"\n📈 Signal callback received {len(candles)} candles")
        print(candles.head())
    
    streamer = HyperSyncStreamer(API_TOKEN, chain="base", signal_callback=test_callback)
    candles = streamer.start_streaming(duration_seconds=60)
    print(f"\n✓ Test complete. Generated {len(candles)} candles")
