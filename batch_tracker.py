#!/usr/bin/env python3
"""
Batch Tracker - Analyze multiple signal batches
"""

import subprocess
import glob
import sys
from pathlib import Path
import polars as pl

def run_command(cmd):
    """Run command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def track_batch(signal_file, capital=100000):
    """Track a single batch: position sizing + P&L tracking"""
    
    signal_path = Path(signal_file)
    if not signal_path.exists():
        return None
    
    print(f"\n{'='*80}")
    print(f"Processing: {signal_path.name}")
    print(f"{'='*80}")
    
    # Run position sizing
    print("Running position sizing...")
    success, stdout, stderr = run_command(f"python position_sizing.py {signal_file}")
    if not success:
        print(f"❌ Position sizing failed: {stderr}")
        return None
    
    # Run P&L tracking
    print("Running P&L tracking...")
    success, stdout, stderr = run_command(f"python track_with_pnl.py {signal_file} {capital}")
    if not success:
        print(f"❌ P&L tracking failed: {stderr}")
        return None
    
    # Extract results from tracking CSV
    tracking_file = signal_path.stem + "_pnl_tracking.csv"
    if not Path(tracking_file).exists():
        return None
    
    df = pl.read_csv(tracking_file)
    
    # Calculate metrics
    completed = df.filter(pl.col('outcome').is_not_null())
    selected = completed.filter(pl.col('selected') == True)
    
    if len(selected) == 0:
        return None
    
    wins = selected.filter(pl.col('outcome') == 'win')
    losses = selected.filter(pl.col('outcome') == 'loss')
    
    total_pnl = selected['pnl_dollars'].sum()
    total_capital = selected['capital_invested'].sum()
    win_rate = len(wins) / len(selected) if len(selected) > 0 else 0
    
    return {
        'file': signal_path.name,
        'timestamp': signal_path.stem.replace('signals_', ''),
        'total_signals': len(df),
        'selected': len(selected),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_capital': total_capital,
        'return_pct': (total_pnl / capital) * 100 if capital > 0 else 0
    }

def main():
    """Track all batches from a given date pattern"""
    
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
    else:
        pattern = "signals_2026-02-02_*.csv"
    
    capital = 100000
    if len(sys.argv) > 2:
        capital = int(sys.argv[2])
    
    print(f"\n{'='*80}")
    print(f"BATCH TRACKER - Pattern: {pattern}, Capital: ${capital:,}")
    print(f"{'='*80}\n")
    
    # Find all matching signal files
    signal_files = sorted(glob.glob(pattern))
    
    # Exclude already processed files
    signal_files = [f for f in signal_files if not ('positioned' in f or 'tracking' in f or 'readable' in f or 'tracked' in f)]
    
    print(f"Found {len(signal_files)} batches to process\n")
    
    if len(signal_files) == 0:
        print("No signal files found!")
        return
    
    # Track each batch
    results = []
    for signal_file in signal_files:
        result = track_batch(signal_file, capital)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY OF ALL BATCHES")
    print(f"{'='*80}\n")
    
    if len(results) == 0:
        print("No results to display")
        return
    
    # Create results dataframe
    df_results = pl.DataFrame(results)
    
    print(df_results.select([
        'timestamp',
        'total_signals',
        'selected',
        'wins',
        'losses',
        'win_rate',
        'total_pnl',
        'return_pct'
    ]))
    
    # Aggregate statistics
    total_batches = len(results)
    profitable_batches = sum(1 for r in results if r['total_pnl'] > 0)
    total_pnl = sum(r['total_pnl'] for r in results)
    avg_win_rate = sum(r['win_rate'] for r in results) / total_batches
    total_selected = sum(r['selected'] for r in results)
    total_wins = sum(r['wins'] for r in results)
    
    print(f"\n{'='*80}")
    print("AGGREGATE PERFORMANCE")
    print(f"{'='*80}")
    print(f"Total Batches Tracked: {total_batches}")
    print(f"Profitable Batches: {profitable_batches} ({profitable_batches/total_batches*100:.1f}%)")
    print(f"Total Signals Selected: {total_selected}")
    print(f"Total Wins: {total_wins} ({total_wins/total_selected*100:.1f}%)")
    print(f"Average Win Rate: {avg_win_rate*100:.1f}%")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Total Portfolio Return: {(total_pnl/capital)*100:.2f}%")
    print(f"Average Return per Batch: {(total_pnl/capital/total_batches)*100:.2f}%")
    
    # Best and worst
    best_batch = max(results, key=lambda x: x['total_pnl'])
    worst_batch = min(results, key=lambda x: x['total_pnl'])
    
    print(f"\n📈 Best Batch: {best_batch['timestamp']}")
    print(f"   P&L: ${best_batch['total_pnl']:,.2f} ({best_batch['return_pct']:.2f}%)")
    print(f"   Win Rate: {best_batch['win_rate']*100:.1f}%")
    
    print(f"\n📉 Worst Batch: {worst_batch['timestamp']}")
    print(f"   P&L: ${worst_batch['total_pnl']:,.2f} ({worst_batch['return_pct']:.2f}%)")
    print(f"   Win Rate: {worst_batch['win_rate']*100:.1f}%")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
