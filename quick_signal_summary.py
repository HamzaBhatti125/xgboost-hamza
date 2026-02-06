#!/usr/bin/env python3
"""
Quick Signal Summary with Regime Info
======================================

Shows summary of signals including regime metadata.
"""

import polars as pl
import sys
from pathlib import Path

def summarize_signals(filepath: str):
    """Summarize signal file with regime info"""
    if not Path(filepath).exists():
        print(f"❌ File not found: {filepath}")
        return
    
    df = pl.read_csv(filepath)
    
    print("="*80)
    print(f"SIGNAL SUMMARY: {Path(filepath).name}")
    print("="*80)
    print()
    
    # Basic stats
    print(f"📊 Total signals generated: {len(df)}")
    print(f"✅ Signals selected for trading: {df.filter(pl.col('selected') == True).shape[0]}")
    print()
    
    # Regime info
    regime_info = df.select([
        'market_regime', 'regime_confidence', 
        'emergency_exit', 'reduce_exposure', 'increase_exposure'
    ]).unique()
    
    print("🌍 MARKET REGIME:")
    for row in regime_info.iter_rows(named=True):
        print(f"   Regime: {row['market_regime'].upper()}")
        print(f"   Confidence: {row['regime_confidence']:.0%}")
        print(f"   Emergency Exit: {'YES ⚠️' if row['emergency_exit'] else 'NO ✅'}")
        print(f"   Reduce Exposure: {'YES' if row['reduce_exposure'] else 'NO'}")
        print(f"   Increase Exposure: {'YES' if row['increase_exposure'] else 'NO'}")
    print()
    
    # Selected signals stats
    selected = df.filter(pl.col('selected') == True)
    if len(selected) > 0:
        print("💰 SELECTED SIGNALS:")
        print(f"   Total capital deployed: {selected['adjusted_position_size'].sum():.1%}")
        print(f"   Avg position size: {selected['adjusted_position_size'].mean():.2%}")
        print(f"   Min/Max position: {selected['adjusted_position_size'].min():.2%} / {selected['adjusted_position_size'].max():.2%}")
        print(f"   Avg confidence: {selected['pred_proba'].mean():.3f}")
        print(f"   Confidence range: {selected['pred_proba'].min():.3f} - {selected['pred_proba'].max():.3f}")
        print()
        
        print("   Top 3 signals:")
        for i, row in enumerate(selected.head(3).iter_rows(named=True), 1):
            print(f"      {i}. Confidence: {row['pred_proba']:.3f}, Size: {row['adjusted_position_size']:.2%}")
    else:
        print("⚠️  No signals selected for trading")
    
    print()
    print("="*80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_signal_summary.py <signals_file.csv>")
        sys.exit(1)
    
    summarize_signals(sys.argv[1])
