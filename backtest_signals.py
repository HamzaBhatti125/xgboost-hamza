#!/usr/bin/env python3
"""
Backtest Multiple Signal Batches
=================================

Run track_with_pnl.py on multiple signal files to get aggregate performance.
"""

import polars as pl
from pathlib import Path
import sys
from track_with_pnl import PositionTracker

def backtest_batch(signal_file: str, starting_capital: float = 10000):
    """Backtest a single signal batch"""
    tracker = PositionTracker(starting_capital=starting_capital)
    
    try:
        tracker.track_with_positions(signal_file)
        
        # Read the results
        pnl_file = signal_file.replace('.csv', '_pnl_tracking.csv')
        if Path(pnl_file).exists():
            df = pl.read_csv(pnl_file)
            
            # Calculate metrics
            completed = df.filter(pl.col('outcome').is_in(['win', 'loss', 'neutral']))
            
            if len(completed) > 0:
                total_pnl = completed['pnl_dollars'].sum()
                win_rate = (completed.filter(pl.col('outcome') == 'win').shape[0] / len(completed)) * 100
                capital_deployed = completed['capital_invested'].sum()
                
                return {
                    'file': signal_file,
                    'total_signals': len(df),
                    'completed': len(completed),
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'capital_deployed': capital_deployed,
                    'return_pct': (total_pnl / starting_capital) * 100
                }
            else:
                return {
                    'file': signal_file,
                    'total_signals': len(df),
                    'completed': 0,
                    'total_pnl': 0,
                    'win_rate': 0,
                    'capital_deployed': 0,
                    'return_pct': 0
                }
    except Exception as e:
        print(f"❌ Error processing {signal_file}: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python backtest_signals.py <signal_file1> <signal_file2> ... [starting_capital]")
        print("Example: python backtest_signals.py signals_*.csv 10000")
        sys.exit(1)
    
    # Parse arguments
    signal_files = []
    starting_capital = 10000
    
    for arg in sys.argv[1:]:
        if arg.endswith('.csv'):
            # Skip tracking files to avoid duplicates
            if '_pnl_tracking.csv' not in arg and '_pnl_report' not in arg:
                signal_files.append(arg)
        else:
            try:
                starting_capital = float(arg)
            except ValueError:
                pass
    
    if not signal_files:
        print("❌ No signal files provided")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"BACKTESTING {len(signal_files)} SIGNAL BATCHES")
    print(f"Starting Capital: ${starting_capital:,.2f}")
    print(f"{'='*80}\n")
    
    # Process each batch
    results = []
    for signal_file in signal_files:
        print(f"\n{'─'*80}")
        print(f"Processing: {signal_file}")
        print(f"{'─'*80}")
        
        result = backtest_batch(signal_file, starting_capital)
        if result:
            results.append(result)
    
    # Aggregate results
    if not results:
        print("\n❌ No results to show")
        return
    
    results_df = pl.DataFrame(results)
    
    print(f"\n\n{'='*80}")
    print(f"AGGREGATE BACKTEST RESULTS")
    print(f"{'='*80}\n")
    
    total_pnl = results_df['total_pnl'].sum()
    total_completed = results_df['completed'].sum()
    avg_win_rate = results_df.filter(pl.col('completed') > 0)['win_rate'].mean() if len(results_df.filter(pl.col('completed') > 0)) > 0 else 0
    total_return_pct = (total_pnl / starting_capital) * 100
    
    print(f"📊 Total Batches: {len(results)}")
    print(f"📈 Total Completed Signals: {total_completed}")
    print(f"💰 Total P&L: ${total_pnl:,.2f} ({total_return_pct:+.2f}%)")
    print(f"🎯 Average Win Rate: {avg_win_rate:.1f}%")
    print(f"💼 Final Capital: ${starting_capital + total_pnl:,.2f}")
    
    # Show per-batch breakdown
    print(f"\n{'─'*80}")
    print(f"PER-BATCH BREAKDOWN")
    print(f"{'─'*80}\n")
    
    print(results_df.select([
        'file',
        'total_signals',
        'completed',
        pl.col('total_pnl').round(2).alias('pnl_$'),
        pl.col('win_rate').round(1).alias('win_rate_%'),
        pl.col('return_pct').round(3).alias('return_%')
    ]))
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
