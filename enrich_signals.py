#!/usr/bin/env python3
"""
Enrich Trading Signals with Human-Readable Token Information
=============================================================

Converts pair addresses to token symbols using blockchain queries and pair universe.
"""

import pandas as pd
import polars as pl
from web3 import Web3
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Web3 (Base chain)
w3 = Web3(Web3.HTTPProvider('https://mainnet.base.org'))

# Standard ERC20 ABI for token info
ERC20_ABI = json.loads('[{"constant":true,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"payable":false,"stateMutability":"view","type":"function"}]')

# Uniswap V2 Pair ABI
PAIR_ABI = json.loads('[{"constant":true,"inputs":[],"name":"token0","outputs":[{"name":"","type":"address"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"token1","outputs":[{"name":"","type":"address"}],"payable":false,"stateMutability":"view","type":"function"}]')

def get_token_info(token_address):
    """Get token symbol, name, decimals from blockchain"""
    try:
        token_contract = w3.eth.contract(
            address=Web3.to_checksum_address(token_address),
            abi=ERC20_ABI
        )
        
        symbol = token_contract.functions.symbol().call()
        name = token_contract.functions.name().call()
        decimals = token_contract.functions.decimals().call()
        
        return {
            'symbol': symbol,
            'name': name,
            'decimals': decimals
        }
    except Exception as e:
        logger.warning(f"Error getting token info for {token_address}: {e}")
        return {
            'symbol': 'UNKNOWN',
            'name': 'Unknown Token',
            'decimals': 18
        }

def get_pair_tokens(pair_address):
    """Get token0 and token1 addresses from pair"""
    try:
        pair_contract = w3.eth.contract(
            address=Web3.to_checksum_address(pair_address),
            abi=PAIR_ABI
        )
        
        token0 = pair_contract.functions.token0().call()
        token1 = pair_contract.functions.token1().call()
        
        return token0, token1
    except Exception as e:
        logger.warning(f"Error getting pair tokens for {pair_address}: {e}")
        return None, None

def load_pair_universe():
    """Load existing pair universe with symbols"""
    pair_universe_path = Path("Files/pair-universe")
    
    if pair_universe_path.exists():
        try:
            df = pl.read_parquet(pair_universe_path)
            logger.info(f"✓ Loaded pair universe with {len(df)} pairs")
            return df.to_pandas()
        except Exception as e:
            logger.warning(f"Could not load pair universe: {e}")
    
    return None

def enrich_signals_with_symbols(signals_csv='signals_live.csv', output_csv='signals_readable.csv'):
    """Add human-readable token symbols to signals"""
    
    logger.info("Loading signals...")
    df = pd.read_csv(signals_csv)
    
    logger.info(f"Processing {len(df)} signals...")
    
    # Try to load existing pair universe
    pair_universe = load_pair_universe()
    
    # Initialize new columns
    df['token0_symbol'] = ''
    df['token1_symbol'] = ''
    df['token0_name'] = ''
    df['token1_name'] = ''
    df['pair_name'] = ''
    df['token0_address'] = ''
    df['token1_address'] = ''
    
    # Cache to avoid repeated queries
    token_cache = {}
    pair_cache = {}
    
    for idx, row in df.iterrows():
        pair_address = row['pair_address']
        
        if (idx + 1) % 10 == 0:
            logger.info(f"Progress: {idx+1}/{len(df)}")
        
        # Check if we have this pair in universe
        if pair_universe is not None:
            pair_info = pair_universe[pair_universe['address'] == pair_address]
            if not pair_info.empty:
                pair_info = pair_info.iloc[0]
                df.at[idx, 'token0_symbol'] = pair_info.get('token0_symbol', 'UNKNOWN')
                df.at[idx, 'token1_symbol'] = pair_info.get('token1_symbol', 'UNKNOWN')
                df.at[idx, 'token0_name'] = pair_info.get('token0_name', 'Unknown')
                df.at[idx, 'token1_name'] = pair_info.get('token1_name', 'Unknown')
                df.at[idx, 'token0_address'] = pair_info.get('token0_address', '')
                df.at[idx, 'token1_address'] = pair_info.get('token1_address', '')
                df.at[idx, 'pair_name'] = f"{pair_info.get('token0_symbol', 'UNKNOWN')}/{pair_info.get('token1_symbol', 'UNKNOWN')}"
                continue
        
        # If not in universe, query blockchain
        if pair_address not in pair_cache:
            token0_addr, token1_addr = get_pair_tokens(pair_address)
            if token0_addr and token1_addr:
                pair_cache[pair_address] = (token0_addr, token1_addr)
            else:
                df.at[idx, 'pair_name'] = 'UNKNOWN/UNKNOWN'
                continue
        
        token0_addr, token1_addr = pair_cache[pair_address]
        
        # Get token0 info
        if token0_addr not in token_cache:
            token_cache[token0_addr] = get_token_info(token0_addr)
        token0_info = token_cache[token0_addr]
        
        # Get token1 info
        if token1_addr not in token_cache:
            token_cache[token1_addr] = get_token_info(token1_addr)
        token1_info = token_cache[token1_addr]
        
        # Update dataframe
        df.at[idx, 'token0_symbol'] = token0_info['symbol']
        df.at[idx, 'token1_symbol'] = token1_info['symbol']
        df.at[idx, 'token0_name'] = token0_info['name']
        df.at[idx, 'token1_name'] = token1_info['name']
        df.at[idx, 'token0_address'] = token0_addr
        df.at[idx, 'token1_address'] = token1_addr
        df.at[idx, 'pair_name'] = f"{token0_info['symbol']}/{token1_info['symbol']}"
    
    # Reorder columns for better readability
    cols_order = [
        'pair_name', 'token0_symbol', 'token1_symbol',
        'timestamp', 'pred_proba', 'signal',
        'open', 'high', 'low', 'close',
        'volume_token0', 'volume_token1', 'num_trades',
        'return_1c', 'return_4c', 'return_16c',
        'volatility_4h', 'volatility_24h',
        'ema_cross_signal', 'buy_sell_ratio',
        'pair_address', 'token0_address', 'token1_address',
        'token0_name', 'token1_name',
        'generated_at'
    ]
    
    # Only include columns that exist
    cols_order = [col for col in cols_order if col in df.columns]
    df = df[cols_order]
    
    # Sort by confidence (highest first)
    df = df.sort_values('pred_proba', ascending=False)
    
    # Save enriched signals
    df.to_csv(output_csv, index=False)
    logger.info(f"\n✅ Saved enriched signals to {output_csv}")
    
    # Print top 10 signals
    print("\n" + "="*100)
    print("TOP 10 SIGNALS (Human Readable)")
    print("="*100)
    
    for i, (idx, row) in enumerate(df.head(10).iterrows(), 1):
        print(f"\n#{i} - {row['pair_name']}")
        print(f"   Confidence: {row['pred_proba']:.2%}")
        print(f"   Price: ${row['close']:.6f} | 24h Vol: ${row['volume_token1']:.2f}")
        print(f"   Returns: 15m={row['return_1c']:.2f}% | 1h={row['return_4c']:.2f}% | 4h={row['return_16c']:.2f}%")
        print(f"   Volatility: 4h={row['volatility_4h']:.2f} | 24h={row['volatility_24h']:.2f}")
        print(f"   Token0: {row['token0_name']} ({row['token0_symbol']})")
        print(f"   Token1: {row['token1_name']} ({row['token1_symbol']})")
        print(f"   Pair: {row['pair_address']}")
    
    print("\n" + "="*100)
    
    return df

if __name__ == "__main__":
    import sys
    
    # Check if a specific file was passed
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = input_file.replace('.csv', '_readable.csv')
    else:
        input_file = 'signals_live.csv'
        output_file = 'signals_readable.csv'
    
    # Enrich the signals
    df_enriched = enrich_signals_with_symbols(input_file, output_file)
    
    # Generate summary stats
    print("\n📊 SIGNAL SUMMARY")
    print(f"Total Signals: {len(df_enriched)}")
    print(f"Unique Pairs: {df_enriched['pair_name'].nunique()}")
    print(f"Avg Confidence: {df_enriched['pred_proba'].mean():.2%}")
    print(f"High Confidence (>80%): {len(df_enriched[df_enriched['pred_proba'] > 0.8])}")
    print(f"Very High Confidence (>85%): {len(df_enriched[df_enriched['pred_proba'] > 0.85])}")
