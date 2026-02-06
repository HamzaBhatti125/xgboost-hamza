#!/usr/bin/env python3
"""
Envio Hypersync Client for Base Chain
======================================

Streams live Uniswap V2 and V3 swap events from Base chain using Envio Hypersync.
Converts swap events into OHLCV candles compatible with the existing model.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, AsyncIterator
from dataclasses import dataclass
import polars as pl
from dotenv import load_dotenv
import hypersync
from hypersync import BlockField, LogField, ClientConfig
import logging

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class EnvioConfig:
    """Envio Hypersync configuration for Base chain"""
    
    # Base chain configuration
    CHAIN_ID = 8453  # Base
    HYPERSYNC_URL = "https://base.hypersync.xyz"  # Envio Hypersync endpoint for Base
    
    # Uniswap V2 Factory on Base
    UNISWAP_V2_FACTORY = "0x8909Dc15e40173Ff4699343b6eB8132c65e18eC6"
    
    # Uniswap V3 Factory on Base  
    UNISWAP_V3_FACTORY = "0x33128a8fC17869897dcE68Ed026d694621f6FDfD"
    
    # Event signatures
    SWAP_V2_TOPIC = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"  # Swap(address,uint256,uint256,uint256,uint256,address)
    SWAP_V3_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"  # Swap(address,address,int256,int256,uint160,uint128,int24)
    SYNC_V2_TOPIC = "0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1"  # Sync(uint112,uint112)
    PAIR_CREATED_V2_TOPIC = "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9"  # PairCreated
    POOL_CREATED_V3_TOPIC = "0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118"  # PoolCreated
    
    # Candle configuration
    CANDLE_INTERVAL_SECONDS = 900  # 15 minutes (matching training data)
    CANDLE_INTERVAL_MINUTES = 15  # 15 minutes (matching training data)
    
    # Processing configuration
    BATCH_SIZE = 1000  # Process events in batches
    MAX_BLOCKS_PER_QUERY = 10000  # Hypersync query limit
    POLL_INTERVAL_SECONDS = 12  # Poll for new blocks (Base block time ~2s, but poll less frequently)


# ============================================================================
# ABI FRAGMENTS
# ============================================================================

# Minimal ABIs for decoding events
UNISWAP_V2_SWAP_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": True, "name": "sender", "type": "address"},
        {"indexed": False, "name": "amount0In", "type": "uint256"},
        {"indexed": False, "name": "amount1In", "type": "uint256"},
        {"indexed": False, "name": "amount0Out", "type": "uint256"},
        {"indexed": False, "name": "amount1Out", "type": "uint256"},
        {"indexed": True, "name": "to", "type": "address"}
    ],
    "name": "Swap",
    "type": "event"
}

UNISWAP_V3_SWAP_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": True, "name": "sender", "type": "address"},
        {"indexed": True, "name": "recipient", "type": "address"},
        {"indexed": False, "name": "amount0", "type": "int256"},
        {"indexed": False, "name": "amount1", "type": "int256"},
        {"indexed": False, "name": "sqrtPriceX96", "type": "uint160"},
        {"indexed": False, "name": "liquidity", "type": "uint128"},
        {"indexed": False, "name": "tick", "type": "int24"}
    ],
    "name": "Swap",
    "type": "event"
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SwapEvent:
    """Represents a swap event"""
    block_number: int
    block_timestamp: int
    transaction_hash: str
    log_index: int
    pair_address: str
    token0: str
    token1: str
    amount0: float
    amount1: float
    price: float  # token1/token0
    volume_usd: float  # USD volume for weighting (using token1 as proxy)
    is_v3: bool
    liquidity: float  # Pool liquidity at time of swap
    
    
@dataclass
class Candle:
    """OHLCV candle data"""
    pair_address: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume_token0: float
    volume_token1: float
    num_trades: int
    avg_liquidity: float  # Average liquidity during candle period
    vwap: float  # Volume-weighted average price
    

# ============================================================================
# ENVIO HYPERSYNC CLIENT
# ============================================================================

class EnvioHypersyncClient:
    """Client for Envio Hypersync API"""
    
    def __init__(self, url: str = EnvioConfig.HYPERSYNC_URL):
        self.url = url
        bearer_token = os.getenv("ENVIO_API_TOKEN")
        if not bearer_token:
            raise ValueError(
                "ENVIO_API_TOKEN environment variable is required. "
                "Please set it in your .env file or environment."
            )
        
        self.client = hypersync.HypersyncClient(
            ClientConfig(url=url, bearer_token=bearer_token)
        )
        logger.info(f"Initialized Hypersync client for {url}")
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
            
    async def query_logs(
        self,
        from_block: int,
        to_block: Optional[int] = None,
        addresses: Optional[List[str]] = None,
        topics: Optional[List[str]] = None
    ) -> List[Dict]:
        """Query logs from Hypersync"""
        
        to_block = to_block or from_block + EnvioConfig.MAX_BLOCKS_PER_QUERY
        
        # Build query using official SDK
        query = hypersync.Query(
            from_block=from_block,
            to_block=to_block,
            logs=[
                hypersync.LogSelection(
                    address=addresses if addresses else [],
                    topics=[topics] if topics else []
                )
            ] if addresses or topics else [],
            field_selection=hypersync.FieldSelection(
                log=[
                    LogField.LOG_INDEX,
                    LogField.TRANSACTION_INDEX,
                    LogField.TRANSACTION_HASH,
                    LogField.DATA,
                    LogField.ADDRESS,
                    LogField.TOPIC0,
                    LogField.TOPIC1,
                    LogField.TOPIC2,
                    LogField.TOPIC3,
                    LogField.BLOCK_NUMBER, # Explicitly request Block Number
                ],
                block=[BlockField.NUMBER, BlockField.TIMESTAMP]
            )
        )
        
        try:
            res = await self.client.get(query)
            
            # Create a lookup map for blocks to speed up processing
            # Map block_number -> timestamp
            blocks_map = {b.number: b.timestamp for b in res.data.blocks} if res.data.blocks else {}
            
            # Convert to list of dicts
            logs = []
            if res.data.logs:
                for log in res.data.logs:
                    # Helper function to convert bytes/str to hex string
                    def to_hex(value):
                        if value is None:
                            return None
                        if isinstance(value, bytes):
                            return "0x" + value.hex()
                        if isinstance(value, str):
                            return value if value.startswith("0x") else "0x" + value
                        return str(value)
                    
                    # 1. Base log data
                    log_dict = {
                        "log_index": log.log_index,
                        "transaction_index": log.transaction_index,
                        "transaction_hash": to_hex(log.transaction_hash) if log.transaction_hash else "",
                        "address": to_hex(log.address) if log.address else "",
                        "data": to_hex(log.data) if log.data else "",
                        "topic0": to_hex(log.topic0) if hasattr(log, 'topic0') and log.topic0 else None,
                        "topic1": to_hex(log.topic1) if hasattr(log, 'topic1') and log.topic1 else None,
                        "topic2": to_hex(log.topic2) if hasattr(log, 'topic2') and log.topic2 else None,
                        "topic3": to_hex(log.topic3) if hasattr(log, 'topic3') and log.topic3 else None,
                        # 2. Get block number directly from the log object
                        "block_number": log.block_number 
                    }
                    
                    # 3. Get timestamp from our blocks map
                    # If block is missing, default to 0 or current time to prevent crashes
                    log_dict["block_timestamp"] = blocks_map.get(log.block_number, 0)
                    
                    logs.append(log_dict)
                    
            return logs
                
        except Exception as e:
            logger.error(f"Error querying Hypersync: {e}", exc_info=True)
            return []
            
    async def get_current_block(self) -> int:
        """Get current block height"""
        try:
            height = await self.client.get_height()
            return height
        except Exception as e:
            logger.error(f"Error getting current block: {e}")
            return 0


# ============================================================================
# SWAP EVENT PROCESSOR
# ============================================================================

class SwapEventProcessor:
    """Processes swap events from Hypersync into structured data"""
    
    def __init__(self, pair_universe_path: str = "./Files/pair-universe"):
        self.pair_info: Dict[str, Dict] = {}  # Cache pair info
        self.liquidity_cache: Dict[str, float] = {}  # Cache V2 pair liquidity from Sync events
        self._load_pair_universe(pair_universe_path)
    
    def _load_pair_universe(self, path: str):
        """Load pair universe data with decimals information"""
        try:
            df = pl.read_parquet(path)
            # Create lookup dict: address -> {token0_decimals, token1_decimals, ...}
            for row in df.iter_rows(named=True):
                address = row.get('address', '').lower()
                if address:
                    self.pair_info[address] = {
                        'token0_decimals': row.get('token0_decimals', 18),
                        'token1_decimals': row.get('token1_decimals', 18),
                        'token0_address': row.get('token0_address', ''),
                        'token1_address': row.get('token1_address', ''),
                        'token0_symbol': row.get('token0_symbol', ''),
                        'token1_symbol': row.get('token1_symbol', '')
                    }
            logger.info(f"Loaded {len(self.pair_info)} pairs from universe")
        except Exception as e:
            logger.warning(f"Could not load pair universe: {e}. Will use default 18 decimals.")
    
    def _get_decimals(self, pair_address: str) -> tuple:
        """Get token decimals for a pair, returns (token0_decimals, token1_decimals)"""
        pair_address = pair_address.lower()
        if pair_address in self.pair_info:
            info = self.pair_info[pair_address]
            return (info.get('token0_decimals', 18), info.get('token1_decimals', 18))
        # Default to 18 if not found
        logger.debug(f"Pair {pair_address} not found in universe, using default 18 decimals")
        return (18, 18)
    
    def decode_v2_sync(self, log: Dict) -> None:
        """Decode Uniswap V2 Sync event and update liquidity cache"""
        try:
            # Decode log data: Sync(uint112 reserve0, uint112 reserve1)
            data = bytes.fromhex(log["data"][2:] if log["data"].startswith("0x") else log["data"])
            
            # Extract reserves (2 uint112 values, but stored as uint256 in data)
            reserve0 = int.from_bytes(data[0:32], byteorder='big')
            reserve1 = int.from_bytes(data[32:64], byteorder='big')
            
            # Get decimals for proper calculation
            token0_decimals, token1_decimals = self._get_decimals(log["address"])
            
            # Calculate geometric mean liquidity: sqrt(reserve0 * reserve1)
            # Apply decimals for proper units
            reserve0_adj = reserve0 / (10 ** token0_decimals)
            reserve1_adj = reserve1 / (10 ** token1_decimals)
            
            liquidity = (reserve0_adj * reserve1_adj) ** 0.5
            
            # Cache liquidity for this pair
            self.liquidity_cache[log["address"].lower()] = liquidity
            
        except Exception as e:
            logger.error(f"Error decoding V2 Sync: {e}")
        
    def decode_v2_swap(self, log: Dict) -> Optional[SwapEvent]:
        """Decode Uniswap V2 swap event"""
        try:
            # Decode log data
            data = bytes.fromhex(log["data"][2:] if log["data"].startswith("0x") else log["data"])
            
            # Decode amounts (4 uint256 values)
            amount0_in = int.from_bytes(data[0:32], byteorder='big')
            amount1_in = int.from_bytes(data[32:64], byteorder='big')
            amount0_out = int.from_bytes(data[64:96], byteorder='big')
            amount1_out = int.from_bytes(data[96:128], byteorder='big')
            
            # Get correct decimals for this pair
            token0_decimals, token1_decimals = self._get_decimals(log["address"])
            
            # Calculate net amounts (in is negative, out is positive) using correct decimals
            amount0 = float(amount0_out - amount0_in) / (10 ** token0_decimals)
            amount1 = float(amount1_out - amount1_in) / (10 ** token1_decimals)
            
            # Calculate price (avoid division by zero)
            if amount0 != 0:
                price = abs(amount1 / amount0)
            else:
                price = 0.0
            
            # Calculate volume (use token1 as proxy for USD value)
            volume_usd = abs(amount1)
            
            # Get liquidity from cache (default to 0 if not available)
            liquidity = self.liquidity_cache.get(log["address"].lower(), 0.0)
                
            return SwapEvent(
                block_number=log["block_number"],
                block_timestamp=log["block_timestamp"],
                transaction_hash=log["transaction_hash"],
                log_index=log["log_index"],
                pair_address=log["address"],
                token0="",  # Will be filled from pair info
                token1="",
                amount0=amount0,
                amount1=amount1,
                price=price,
                volume_usd=volume_usd,
                is_v3=False,
                liquidity=liquidity
            )
            
        except Exception as e:
            logger.error(f"Error decoding V2 swap: {e}")
            return None
            
    def decode_v3_swap(self, log: Dict) -> Optional[SwapEvent]:
        """Decode Uniswap V3 swap event"""
        try:
            # Decode log data
            data = bytes.fromhex(log["data"][2:] if log["data"].startswith("0x") else log["data"])
            
            # Decode amounts (int256, int256, uint160, uint128, int24)
            amount0_bytes = data[0:32]
            amount1_bytes = data[32:64]
            # sqrtPriceX96 at data[64:96]
            liquidity_bytes = data[96:128]
            # tick at data[128:160]
            
            # Convert from int256 (signed)
            amount0_raw = int.from_bytes(amount0_bytes, byteorder='big', signed=True)
            amount1_raw = int.from_bytes(amount1_bytes, byteorder='big', signed=True)
            
            # Extract liquidity (uint128)
            liquidity_raw = int.from_bytes(liquidity_bytes, byteorder='big')
            
            # Get correct decimals for this pair
            token0_decimals, token1_decimals = self._get_decimals(log["address"])
            
            # Convert amounts using correct decimals
            amount0 = float(amount0_raw) / (10 ** token0_decimals)
            amount1 = float(amount1_raw) / (10 ** token1_decimals)
            
            # Convert liquidity to human-readable format (geometric mean)
            # V3 liquidity is in sqrt(token0 * token1) units
            liquidity = float(liquidity_raw) / (10 ** ((token0_decimals + token1_decimals) / 2))
            
            # Calculate price
            if amount0 != 0:
                price = abs(amount1 / amount0)
            else:
                price = 0.0
            
            # Calculate volume (use token1 as proxy for USD value)
            volume_usd = abs(amount1)
                
            return SwapEvent(
                block_number=log["block_number"],
                block_timestamp=log["block_timestamp"],
                transaction_hash=log["transaction_hash"],
                log_index=log["log_index"],
                pair_address=log["address"],
                token0="",
                token1="",
                amount0=amount0,
                amount1=amount1,
                price=price,
                volume_usd=volume_usd,
                is_v3=True,
                liquidity=liquidity
            )
            
        except Exception as e:
            logger.error(f"Error decoding V3 swap: {e}")
            return None


# ============================================================================
# CANDLE AGGREGATOR
# ============================================================================

class CandleAggregator:
    """Aggregates swap events into OHLCV candles"""
    
    def __init__(self, interval_seconds: int = EnvioConfig.CANDLE_INTERVAL_SECONDS):
        self.interval_seconds = interval_seconds
        self.swaps_buffer: Dict[str, List[SwapEvent]] = {}  # pair_address -> [swaps]
        
    def add_swap(self, swap: SwapEvent):
        """Add swap to buffer"""
        if swap.pair_address not in self.swaps_buffer:
            self.swaps_buffer[swap.pair_address] = []
        self.swaps_buffer[swap.pair_address].append(swap)
        
    def generate_candles(self, end_timestamp: Optional[int] = None) -> List[Candle]:
        """Generate candles from buffered swaps"""
        if end_timestamp is None:
            end_timestamp = int(datetime.now().timestamp())
            
        candles = []
        
        for pair_address, swaps in self.swaps_buffer.items():
            if not swaps:
                continue
            
            # Helper to safely parse timestamp (Hex str or Int)
            def parse_ts(ts):
                if isinstance(ts, int):
                    return ts
                if isinstance(ts, str) and ts.startswith("0x"):
                    return int(ts, 16)
                return int(ts)

            # Sort by timestamp
            swaps.sort(key=lambda x: parse_ts(x.block_timestamp))
            
            # Group by candle interval
            candle_groups = {}
            for swap in swaps:
                ts = parse_ts(swap.block_timestamp)
                
                candle_start = (ts // self.interval_seconds) * self.interval_seconds
                
                if candle_start not in candle_groups:
                    candle_groups[candle_start] = []
                candle_groups[candle_start].append(swap)
                
            # Create candles with volume-weighted prices
            for candle_start, group_swaps in candle_groups.items():
                if not group_swaps:
                    continue
                
                # Filter valid swaps (price > 0 and volume > 0)
                valid_swaps = [(s.price, s.volume_usd, parse_ts(s.block_timestamp)) 
                               for s in group_swaps if s.price > 0 and s.volume_usd > 0]
                
                if not valid_swaps:
                    continue
                
                # Sort by timestamp for correct OHLC ordering
                valid_swaps.sort(key=lambda x: x[2])
                
                # Extract prices and volumes
                prices = [p for p, _, _ in valid_swaps]
                volumes = [v for _, v, _ in valid_swaps]
                
                # Calculate volume-weighted average price (VWAP)
                total_volume = sum(volumes)
                vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume if total_volume > 0 else prices[-1]
                
                # Calculate average liquidity
                liquidities = [s.liquidity for s in group_swaps if s.liquidity > 0]
                avg_liquidity = sum(liquidities) / len(liquidities) if liquidities else 0.0
                
                candle = Candle(
                    pair_address=pair_address,
                    timestamp=candle_start,
                    open=prices[0],      # First trade price
                    high=max(prices),    # Highest trade price
                    low=min(prices),     # Lowest trade price  
                    close=prices[-1],    # Last trade price
                    volume_token0=sum(abs(s.amount0) for s in group_swaps),
                    volume_token1=sum(abs(s.amount1) for s in group_swaps),
                    num_trades=len(group_swaps),
                    avg_liquidity=avg_liquidity,
                    vwap=vwap
                )
                candles.append(candle)
                
        return candles
        
    def clear_old_swaps(self, before_timestamp: int):
        """Remove swaps older than specified timestamp"""
        for pair_address in list(self.swaps_buffer.keys()):
            self.swaps_buffer[pair_address] = [
                s for s in self.swaps_buffer[pair_address]
                if s.block_timestamp >= before_timestamp
            ]
            if not self.swaps_buffer[pair_address]:
                del self.swaps_buffer[pair_address]


# ============================================================================
# MAIN STREAMER
# ============================================================================

class LiveSwapStreamer:
    """Main class for streaming live swap data"""
    
    def __init__(self, start_block: Optional[int] = None):
        self.client = EnvioHypersyncClient()
        self.processor = SwapEventProcessor()
        self.aggregator = CandleAggregator()
        self.current_block = start_block
        self.pair_addresses: set = set()  # Track discovered pairs
        
    async def discover_pairs(self, from_block: int, to_block: int):
        """Discover new pairs from PairCreated/PoolCreated events"""
        logger.info(f"Discovering pairs from blocks {from_block} to {to_block}")
        
        # Query for V2 pair creation
        v2_logs = await self.client.query_logs(
            from_block=from_block,
            to_block=to_block,
            addresses=[EnvioConfig.UNISWAP_V2_FACTORY],
            topics=[EnvioConfig.PAIR_CREATED_V2_TOPIC]
        )
        
        # Query for V3 pool creation
        v3_logs = await self.client.query_logs(
            from_block=from_block,
            to_block=to_block,
            addresses=[EnvioConfig.UNISWAP_V3_FACTORY],
            topics=[EnvioConfig.POOL_CREATED_V3_TOPIC]
        )
        
        # Extract pair addresses from event data
        # V2 PairCreated: event PairCreated(address indexed token0, address indexed token1, address pair, uint)
        # - pair address is in data (first 32 bytes after skipping token addresses)
        for log in v2_logs:
            try:
                data = log.get("data", "")
                if data and len(data) >= 66:  # 0x + 64 hex chars (32 bytes)
                    # Remove 0x prefix
                    data_hex = data[2:] if data.startswith("0x") else data
                    # Pair address is first 20 bytes (40 hex chars), padded to 32 bytes (64 hex chars)
                    pair_address = "0x" + data_hex[24:64]  # Skip padding (24 chars) + take 40 chars
                    self.pair_addresses.add(pair_address.lower())
                    logger.debug(f"V2 Pair discovered: {pair_address}")
            except Exception as e:
                logger.error(f"Error decoding V2 pair address: {e}")
        
        # V3 PoolCreated: event PoolCreated(address indexed token0, address indexed token1, uint24 indexed fee, int24 tickSpacing, address pool)
        # - pool address is in data (after tickSpacing)
        for log in v3_logs:
            try:
                data = log.get("data", "")
                if data and len(data) >= 130:  # 0x + 128 hex chars (64 bytes: 32 for tickSpacing + 32 for pool)
                    # Remove 0x prefix
                    data_hex = data[2:] if data.startswith("0x") else data
                    # Pool address is second 32 bytes, starting at position 64
                    pool_address = "0x" + data_hex[88:128]  # Skip first 32 bytes (64 chars) + padding (24 chars)
                    self.pair_addresses.add(pool_address.lower())
                    logger.debug(f"V3 Pool discovered: {pool_address}")
            except Exception as e:
                logger.error(f"Error decoding V3 pool address: {e}")
                
        logger.info(f"Discovered {len(self.pair_addresses)} --> {self.pair_addresses} total pairs")
        
    async def stream_swaps(self) -> AsyncIterator[SwapEvent]:
        """Stream swap events in real-time"""
        
        async with self.client:
            # Get starting block
            if self.current_block is None:
                self.current_block = await self.client.get_current_block()
                logger.info(f"Starting from block {self.current_block}")
                
            # Initial pair discovery (last 1000 blocks)
            discover_from = max(0, self.current_block - 10000)
            await self.discover_pairs(discover_from, self.current_block)
            
            while True:
                try:
                    # Get latest block
                    latest_block = await self.client.get_current_block()
                    
                    if latest_block <= self.current_block:
                        # No new blocks, wait
                        await asyncio.sleep(EnvioConfig.POLL_INTERVAL_SECONDS)
                        continue
                        
                    logger.info(f"Processing blocks {self.current_block} to {latest_block}")
                    
                    # Query Sync events first (for V2 liquidity)
                    sync_logs = await self.client.query_logs(
                        from_block=self.current_block,
                        to_block=min(latest_block, self.current_block + EnvioConfig.MAX_BLOCKS_PER_QUERY),
                        topics=[EnvioConfig.SYNC_V2_TOPIC]
                    )
                    
                    # Process Sync events to update liquidity cache
                    for log in sync_logs:
                        self.processor.decode_v2_sync(log)
                    
                    # Query swap events (both V2 and V3)
                    v2_logs = await self.client.query_logs(
                        from_block=self.current_block,
                        to_block=min(latest_block, self.current_block + EnvioConfig.MAX_BLOCKS_PER_QUERY),
                        topics=[EnvioConfig.SWAP_V2_TOPIC]
                    )
                    
                    v3_logs = await self.client.query_logs(
                        from_block=self.current_block,
                        to_block=min(latest_block, self.current_block + EnvioConfig.MAX_BLOCKS_PER_QUERY),
                        topics=[EnvioConfig.SWAP_V3_TOPIC]
                    )
                    
                    logger.info(f"Found {len(v2_logs)} V2 swaps and {len(v3_logs)} V3 swaps")
                    
                    # Process V2 swaps
                    for log in v2_logs:
                        swap = self.processor.decode_v2_swap(log)
                        if swap:
                            self.aggregator.add_swap(swap)
                            yield swap
                            
                    # Process V3 swaps
                    for log in v3_logs:
                        swap = self.processor.decode_v3_swap(log)
                        if swap:
                            self.aggregator.add_swap(swap)
                            yield swap
                            
                    # Update current block
                    self.current_block = min(latest_block, self.current_block + EnvioConfig.MAX_BLOCKS_PER_QUERY)
                    
                    # Periodically discover new pairs
                    if self.current_block % 10000 == 0:
                        await self.discover_pairs(self.current_block - 10000, self.current_block)
                        
                except Exception as e:
                    logger.error(f"Error in stream_swaps: {e}", exc_info=True)
                    await asyncio.sleep(EnvioConfig.POLL_INTERVAL_SECONDS)
                    
    async def backfill_historical_candles(self, hours: int = 25) -> pl.DataFrame:
        """Backfill candles from recent history to populate features properly"""
        logger.info(f"🔄 Backfilling {hours} hours of historical data...")
        
        try:
            # Initialize current_block if not set by querying latest block
            if self.current_block is None:
                self.current_block = await self.client.get_current_block()
                logger.info(f"Initialized current_block to {self.current_block}")

            # Calculate block range (Base chain: ~2 sec per block = 1800 blocks/hour)
            blocks_per_hour = 1800
            from_block = max(0, self.current_block - (hours * blocks_per_hour))
            to_block = self.current_block
            
            logger.info(f"Querying blocks {from_block} to {to_block}")
            
            # Create temporary aggregator for historical data
            historical_aggregator = CandleAggregator(interval_seconds=EnvioConfig.CANDLE_INTERVAL_SECONDS)
            
            # Query in chunks to avoid overwhelming the API
            chunk_size = 10000
            total_swaps = 0
            
            for chunk_start in range(from_block, to_block, chunk_size):
                chunk_end = min(chunk_start + chunk_size, to_block)
                
                # Query Sync events first (for V2 liquidity)
                sync_logs = await self.client.query_logs(
                    from_block=chunk_start,
                    to_block=chunk_end,
                    topics=[EnvioConfig.SYNC_V2_TOPIC]
                )
                
                # Process Sync events to update liquidity cache
                for log in sync_logs:
                    self.processor.decode_v2_sync(log)
                
                # Query V2 and V3 swap events using existing method
                v2_logs = await self.client.query_logs(
                    from_block=chunk_start,
                    to_block=chunk_end,
                    topics=[EnvioConfig.SWAP_V2_TOPIC]
                )
                
                v3_logs = await self.client.query_logs(
                    from_block=chunk_start,
                    to_block=chunk_end,
                    topics=[EnvioConfig.SWAP_V3_TOPIC]
                )
                
                # Process V2 swaps
                for log in v2_logs:
                    swap = self.processor.decode_v2_swap(log)
                    if swap:
                        historical_aggregator.add_swap(swap)
                        total_swaps += 1
                
                # Process V3 swaps
                for log in v3_logs:
                    swap = self.processor.decode_v3_swap(log)
                    if swap:
                        historical_aggregator.add_swap(swap)
                        total_swaps += 1
                
                logger.info(f"  Processed blocks {chunk_start}-{chunk_end}: {total_swaps} swaps so far")
                await asyncio.sleep(0.1)  # Rate limiting
            
            # Generate candles from all historical swaps
            candles = historical_aggregator.generate_candles()
            logger.info(f"✅ Backfill complete: {total_swaps} swaps → {len(candles)} candles")
            
            # Convert to DataFrame
            if not candles:
                return pl.DataFrame()
            
            data = {
                "pair_address": [c.pair_address for c in candles],
                "timestamp": [datetime.fromtimestamp(c.timestamp) for c in candles],
                "open": [c.open for c in candles],
                "high": [c.high for c in candles],
                "low": [c.low for c in candles],
                "close": [c.close for c in candles],
                "volume_token0": [c.volume_token0 for c in candles],
                "volume_token1": [c.volume_token1 for c in candles],
                "num_trades": [c.num_trades for c in candles],
                "avg_liquidity": [c.avg_liquidity for c in candles],
                "vwap": [c.vwap for c in candles],
            }
            
            return pl.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Backfill failed: {e}", exc_info=True)
            return pl.DataFrame()
    
    def get_latest_candles(self) -> pl.DataFrame:
        """Get latest candles from aggregator"""
        candles = self.aggregator.generate_candles()
        print(f"Generated candles ---> {candles[0]} from swaps")
        
        if not candles:
            return pl.DataFrame()
            
        # Convert to polars DataFrame (format compatible with model)
        data = {
            "pair_address": [c.pair_address for c in candles],
            "timestamp": [datetime.fromtimestamp(c.timestamp) for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume_token0": [c.volume_token0 for c in candles],
            "volume_token1": [c.volume_token1 for c in candles],
            "num_trades": [c.num_trades for c in candles],
            "avg_liquidity": [c.avg_liquidity for c in candles],
            "vwap": [c.vwap for c in candles],
        }
        
        return pl.DataFrame(data)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_regime_features(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate market regime features from candle data
    
    Args:
        df: Polars DataFrame with columns: pair_address, timestamp, close, volume_token1, avg_liquidity, vwap
        
    Returns:
        DataFrame with added regime features: amihud_illiquidity, liquidity_change
    """
    if len(df) == 0:
        return df
    
    # Sort by pair and timestamp to ensure proper ordering
    df = df.sort(["pair_address", "timestamp"])
    
    # Calculate log returns
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1).over("pair_address")).log().alias("log_return")
    ])
    
    # Calculate Amihud Illiquidity: |log_return| / volume_usd
    # Using volume_token1 as proxy for volume_usd
    df = df.with_columns([
        (pl.col("log_return").abs() / pl.col("volume_token1").clip(lower_bound=1e-10)).alias("amihud_illiquidity")
    ])
    
    # Calculate Liquidity Change: % change from previous candle
    df = df.with_columns([
        ((pl.col("avg_liquidity") - pl.col("avg_liquidity").shift(1).over("pair_address")) / 
         pl.col("avg_liquidity").shift(1).over("pair_address").clip(lower_bound=1e-10)
        ).alias("liquidity_change")
    ])
    
    # Fill nulls (first candle in each pair will have null for changes)
    df = df.with_columns([
        pl.col("log_return").fill_null(0.0),
        pl.col("amihud_illiquidity").fill_null(0.0),
        pl.col("liquidity_change").fill_null(0.0),
    ])
    
    return df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def main():
    """Example: Stream swaps and generate candles"""
    
    logger.info("Starting Envio Hypersync streamer for Base chain")
    
    streamer = LiveSwapStreamer()
    
    # Stream swaps for a few minutes
    swap_count = 0
    async for swap in streamer.stream_swaps():
        swap_count += 1
        logger.info(f"Swap #{swap_count}: {swap.pair_address} @ {swap.price:.6f}")
        
        # Every 100 swaps, show candle stats
        if swap_count % 100 == 0:
            candles_df = streamer.get_latest_candles()
            logger.info(f"\nGenerated {len(candles_df)} candles")
            if len(candles_df) > 0:
                logger.info(candles_df.head())


if __name__ == "__main__":
    asyncio.run(main())
