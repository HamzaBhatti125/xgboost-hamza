#!/usr/bin/env python3
"""
Get Pair Information
====================

Simple utility to look up detailed information about a trading pair.

Usage:
    python get_pair_info.py 6289079
    python get_pair_info.py --pair-id 5185342
"""

import polars as pl
import argparse
from pathlib import Path


def get_pair_info(pair_id: int, data_path: str = "Files/pair-universe"):
    """Get detailed information about a trading pair"""
    
    data_path = Path(data_path)
    
    # Handle both file and directory paths
    if not data_path.exists():
        # Try .parquet extension if not included
        if not str(data_path).endswith('.parquet'):
            data_path = Path(str(data_path) + '.parquet')
        
        if not data_path.exists():
            print(f"❌ Error: Data path not found: {data_path}")
            print("\nPlease update the path in the script or pass --data-path argument")
            return None
    
    print(f"\n{'='*80}")
    print(f"PAIR INFORMATION: {pair_id}")
    print(f"{'='*80}")
    
    # Load data
    print(f"\nLoading pair-universe data from: {data_path}")
    
    try:
        if data_path.is_file():
            df = pl.read_parquet(data_path)
        else:
            # Directory with multiple parquet files
            df = pl.read_parquet(data_path / "*.parquet")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    print(f"✓ Loaded {len(df):,} total pairs")
    
    # Filter for specific pair
    pair_data = df.filter(pl.col("pair_id") == pair_id)
    
    if len(pair_data) == 0:
        print(f"\n❌ Pair ID {pair_id} not found in dataset")
        print(f"\nTry searching for similar pairs...")
        
        # Show some example pair IDs from the same chain if possible
        if "chain_id" in df.columns:
            print("\nExample pair IDs from Base chain (8453):")
            examples = df.filter(pl.col("chain_id") == 8453).select("pair_id").head(10)
            for row in examples.iter_rows():
                print(f"  - {row[0]}")
        
        return None
    
    print(f"\n✅ Found pair {pair_id}")
    print(f"\n{'='*80}")
    print("PAIR DETAILS")
    print(f"{'='*80}\n")
    
    # Display all columns
    pair_dict = pair_data.to_dicts()[0]
    
    # Group information by category
    basic_info = {}
    token_info = {}
    exchange_info = {}
    volume_info = {}
    flags = {}
    other = {}
    
    for key, value in pair_dict.items():
        if key in ["pair_id", "chain_id", "exchange_slug", "pool_address", "exchange_address"]:
            basic_info[key] = value
        elif "token" in key.lower() or "base" in key.lower() or "quote" in key.lower():
            token_info[key] = value
        elif "exchange" in key.lower() or "dex" in key.lower():
            exchange_info[key] = value
        elif "volume" in key.lower() or "count" in key.lower() or "swap" in key.lower():
            volume_info[key] = value
        elif "flag" in key.lower() or "tax" in key.lower():
            flags[key] = value
        else:
            other[key] = value
    
    # Print grouped information
    if basic_info:
        print("📋 BASIC INFORMATION:")
        for key, value in basic_info.items():
            print(f"  {key:30s}: {value}")
    
    if token_info:
        print("\n🪙 TOKEN INFORMATION:")
        for key, value in token_info.items():
            print(f"  {key:30s}: {value}")
    
    if exchange_info:
        print("\n🏦 EXCHANGE INFORMATION:")
        for key, value in exchange_info.items():
            print(f"  {key:30s}: {value}")
    
    if volume_info:
        print("\n📊 VOLUME & ACTIVITY:")
        for key, value in volume_info.items():
            if isinstance(value, float):
                print(f"  {key:30s}: {value:,.2f}")
            else:
                print(f"  {key:30s}: {value}")
    
    if flags:
        print("\n🚩 FLAGS & FILTERS:")
        for key, value in flags.items():
            if isinstance(value, bool):
                status = "❌ TRUE" if value else "✅ FALSE"
                print(f"  {key:30s}: {status}")
            elif isinstance(value, float):
                print(f"  {key:30s}: {value:.2f}%")
            else:
                print(f"  {key:30s}: {value}")
    
    if other:
        print("\n📎 OTHER INFORMATION:")
        for key, value in other.items():
            print(f"  {key:30s}: {value}")
    
    print(f"\n{'='*80}")
    
    return pair_data


def main():
    parser = argparse.ArgumentParser(
        description="Get detailed information about a trading pair",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python get_pair_info.py 6289079
    python get_pair_info.py --pair-id 5185342
    python get_pair_info.py 4102225 --data-path /custom/path/to/pair-universe
        """
    )
    
    parser.add_argument(
        "pair_id",
        type=int,
        nargs="?",
        help="Trading pair ID to lookup"
    )
    
    parser.add_argument(
        "--pair-id",
        type=int,
        dest="pair_id_alt",
        help="Trading pair ID to lookup (alternative flag)"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="Files/pair-universe",
        help="Path to pair-universe data (default: %(default)s)"
    )
    
    args = parser.parse_args()
    
    # Get pair_id from either positional or flag argument
    pair_id = args.pair_id or args.pair_id_alt
    
    if pair_id is None:
        parser.print_help()
        print("\n❌ Error: Please provide a pair_id")
        print("\nExamples:")
        print("  python get_pair_info.py 6289079")
        print("  python get_pair_info.py --pair-id 5185342")
        return 1
    
    result = get_pair_info(pair_id, args.data_path)
    
    if result is None:
        return 1
    
    print("\n✅ Done!")
    return 0


if __name__ == "__main__":
    exit(main())
