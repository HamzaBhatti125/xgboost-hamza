#!/usr/bin/env python3
"""
Enhanced Signal Tracker with Position Sizing & P&L
===================================================

Tracks signal performance WITH position sizing to calculate:
- Capital invested per signal
- Actual profit/loss in dollars
- Portfolio-level returns
- Risk-adjusted metrics
"""

import polars as pl
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import sys

class PositionTracker:
    """Track signals with position sizing and P&L calculation"""
    
    def __init__(self, cache_path: str = "live_candles_cache.pkl", starting_capital: float = 10000):
        self.cache_path = Path(cache_path)
        self.candle_cache = {}
        self.starting_capital = starting_capital
        self.load_cache()
    
    def load_cache(self):
        """Load candle cache"""
        if self.cache_path.exists():
            with open(self.cache_path, 'rb') as f:
                self.candle_cache = pickle.load(f)
            print(f"✅ Loaded {len(self.candle_cache)} pairs from cache")
        else:
            print("⚠️ No cache found")
    
    def track_with_positions(self, signal_file: str, position_file: str = None):
        """Track signals with position sizing and calculate P&L"""
        
        # Load signals
        signals_df = pl.read_csv(signal_file, try_parse_dates=False)
        print(f"\n📊 Tracking {len(signals_df)} signals from {signal_file}")
        
        # Check if signal file already has position sizing columns
        if 'position_size_pct' in signals_df.columns and 'kelly_fraction' in signals_df.columns:
            print(f"✅ Using position sizing from signal file")
        else:
            # Check if positioned file exists
            if position_file is None:
                position_file = signal_file.replace('.csv', '_positioned.csv')
            
            if Path(position_file).exists():
                positions_df = pl.read_csv(position_file)
                print(f"✅ Loaded position sizing from {position_file}")
                
                # Use positioned file directly if it has all columns
                if 'position_size_scaled' in positions_df.columns:
                    signals_df = positions_df
                    signals_df = signals_df.with_columns([
                        (pl.col('position_size_scaled') * 100).alias('position_size_pct')
                    ])
                else:
                    # Merge with signals
                    signals_df = signals_df.join(
                        positions_df.select(['pair_address', 'position_size_pct', 'kelly_fraction', 'expected_value']),
                        on='pair_address',
                        how='left'
                    )
            else:
                print("⚠️ No position sizing found - using fixed 2% per signal")
                signals_df = signals_df.with_columns([
                    pl.lit(2.0).alias('position_size_pct'),
                    pl.lit(0.0).alias('kelly_fraction'),
                    pl.lit(0.0).alias('expected_value')
                ])
        
        # Filter to only selected signals
        if 'selected' in signals_df.columns:
            before_filter = len(signals_df)
            signals_df = signals_df.filter(pl.col('selected') == True)
            print(f"🎯 Filtered to {len(signals_df)} selected signals (from {before_filter} total)")
        
        # Calculate capital per position
        signals_df = signals_df.with_columns([
            (pl.col('position_size_pct') / 100 * self.starting_capital).alias('capital_invested')
        ])
        
        # Track each signal
        results = []
        for i, row in enumerate(signals_df.iter_rows(named=True)):
            if (i + 1) % 50 == 0:
                print(f"   Processed {i+1}/{len(signals_df)} signals...")
            result = self._track_signal(row)
            results.append(result)
        
        # Create results dataframe
        results_df = pl.DataFrame(results)
        
        # Calculate P&L
        results_df = results_df.with_columns([
            # Profit/Loss in dollars
            (pl.col('capital_invested') * pl.col('return_pct') / 100).alias('pnl_dollars'),
            
            # Return on capital (%)
            pl.col('return_pct').alias('return_on_capital'),
        ])
        
        # Generate summary
        self._generate_pnl_summary(results_df, signal_file)
        
        # Save detailed results
        output_file = signal_file.replace('.csv', '_pnl_tracking.csv')
        results_df.write_csv(output_file)
        print(f"\n💾 Detailed tracking saved to: {output_file}")
        
        return results_df
    
    def _track_signal(self, signal: dict) -> dict:
        """Track individual signal with position sizing"""
        pair = signal['pair_address']
        signal_time = datetime.fromisoformat(signal['timestamp'])
        entry_price = signal['close']
        capital = signal.get('capital_invested', 0)
        
        # Find exit candle (4 hours = 16 candles later)
        exit_time = signal_time + timedelta(hours=4)
        
        # Get candles for this pair
        if pair not in self.candle_cache:
            return {
                **signal,
                'status': 'no_data',
                'exit_price': None,
                'return_pct': 0,
                'outcome': 'unknown'
            }
        
        pair_candles_df = self.candle_cache[pair]
        
        if pair_candles_df.is_empty():
            return {
                **signal,
                'status': 'no_data',
                'exit_price': None,
                'return_pct': 0,
                'outcome': 'unknown'
            }
        
        # Filter candles in the 4-hour window (timestamps are already datetime objects)
        window_candles = pair_candles_df.filter(
            (pl.col('timestamp') > signal_time) & (pl.col('timestamp') <= exit_time)
        )
        
        if window_candles.is_empty():
            return {
                **signal,
                'status': 'pending',
                'exit_price': entry_price,
                'return_pct': 0,
                'outcome': 'pending'
            }
        
        # Check for win (+2%) or loss (-1%)
        max_price = window_candles['high'].max()
        min_price = window_candles['low'].min()
        
        max_return = ((max_price - entry_price) / entry_price) * 100
        min_return = ((min_price - entry_price) / entry_price) * 100
        
        # Determine outcome
        if max_return >= 2.0:
            outcome = 'win'
            exit_price = entry_price * 1.02  # Take profit at +2%
            return_pct = 2.0
        elif min_return <= -1.0:
            outcome = 'loss'
            exit_price = entry_price * 0.99  # Stop loss at -1%
            return_pct = -1.0
        else:
            # Neither target hit - use last candle price
            outcome = 'neutral'
            exit_price = window_candles['close'][-1]
            return_pct = ((exit_price - entry_price) / entry_price) * 100
        
        return {
            **signal,
            'status': 'completed',
            'exit_price': exit_price,
            'return_pct': return_pct,
            'outcome': outcome,
            'max_return': max_return,
            'min_return': min_return
        }
    
    def _generate_pnl_summary(self, results_df: pl.DataFrame, signal_file: str):
        """Generate P&L summary report"""
        
        # Filter completed signals
        completed = results_df.filter(pl.col('status') == 'completed')
        
        if len(completed) == 0:
            print("\n⚠️ No completed signals yet")
            return
        
        # Calculate metrics
        total_signals = len(completed)
        wins = completed.filter(pl.col('outcome') == 'win')
        losses = completed.filter(pl.col('outcome') == 'loss')
        neutrals = completed.filter(pl.col('outcome') == 'neutral')
        
        win_count = len(wins)
        loss_count = len(losses)
        neutral_count = len(neutrals)
        
        win_rate = (win_count / (win_count + loss_count) * 100) if (win_count + loss_count) > 0 else 0
        
        # P&L calculations
        total_capital_deployed = completed['capital_invested'].sum()
        total_pnl = completed['pnl_dollars'].sum()
        total_return_pct = (total_pnl / total_capital_deployed * 100) if total_capital_deployed > 0 else 0
        
        avg_win_pnl = wins['pnl_dollars'].mean() if len(wins) > 0 else 0
        avg_loss_pnl = losses['pnl_dollars'].mean() if len(losses) > 0 else 0
        
        # Best/worst trades
        best_trade_df = completed.sort('pnl_dollars', descending=True).head(1)
        worst_trade_df = completed.sort('pnl_dollars', descending=False).head(1)
        
        # Extract values safely from single-row DataFrames
        if not best_trade_df.is_empty():
            best_pair = best_trade_df['pair_address'][0]
            best_capital = best_trade_df['capital_invested'][0]
            best_return = best_trade_df['return_pct'][0]
            best_pnl = best_trade_df['pnl_dollars'][0]
        else:
            best_pair = 'N/A'
            best_capital = 0
            best_return = 0
            best_pnl = 0
        
        if not worst_trade_df.is_empty():
            worst_pair = worst_trade_df['pair_address'][0]
            worst_capital = worst_trade_df['capital_invested'][0]
            worst_return = worst_trade_df['return_pct'][0]
            worst_pnl = worst_trade_df['pnl_dollars'][0]
        else:
            worst_pair = 'N/A'
            worst_capital = 0
            worst_return = 0
            worst_pnl = 0
        
        neutral_avg_pnl = neutrals['pnl_dollars'].mean() if len(neutrals) > 0 else 0
        
        # Generate markdown report
        report = f"""# P&L Tracking Report
**Signal File:** {signal_file}
**Starting Capital:** ${self.starting_capital:,.2f}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Portfolio Performance

| Metric | Value |
|--------|-------|
| **Total Return** | **${total_pnl:,.2f}** ({total_return_pct:+.2f}%) |
| **Capital Deployed** | ${total_capital_deployed:,.2f} |
| **Final Portfolio Value** | ${self.starting_capital + total_pnl:,.2f} |
| **Win Rate** | {win_rate:.1f}% ({win_count}/{win_count + loss_count}) |

---

## Signal Breakdown

| Outcome | Count | % | Avg P&L |
|---------|-------|---|---------|
| **Win** | {win_count} | {win_count/total_signals*100:.1f}% | ${avg_win_pnl:,.2f} |
| **Loss** | {loss_count} | {loss_count/total_signals*100:.1f}% | ${avg_loss_pnl:,.2f} |
| **Neutral** | {neutral_count} | {neutral_count/total_signals*100:.1f}% | ${neutral_avg_pnl:,.2f} |

---

## Best & Worst Trades

### 🏆 Best Trade
- **Pair:** {best_pair[:16]}...
- **Capital:** ${best_capital:,.2f}
- **Return:** {best_return:+.2f}%
- **P&L:** ${best_pnl:,.2f}

### 📉 Worst Trade
- **Pair:** {worst_pair[:16]}...
- **Capital:** ${worst_capital:,.2f}
- **Return:** {worst_return:+.2f}%
- **P&L:** ${worst_pnl:,.2f}

---

## Risk Metrics

| Metric | Value |
|--------|-------|
| Risk/Reward Ratio | {abs(avg_win_pnl / avg_loss_pnl) if avg_loss_pnl != 0 else 0:.2f}:1 |
| Avg Win | ${avg_win_pnl:,.2f} |
| Avg Loss | ${avg_loss_pnl:,.2f} |
| Max Drawdown | ${worst_pnl:,.2f} |
| Max Gain | ${best_pnl:,.2f} |

---

## Position Sizing Impact

| Metric | Value |
|--------|-------|
| Avg Position Size | {completed['position_size_pct'].mean():.2f}% |
| Total Positions | {total_signals} |
| Avg Capital per Trade | ${total_capital_deployed/total_signals:,.2f} |
| Capital Efficiency | {total_return_pct:.2f}% |

---

## Summary

{self._generate_summary_text(win_rate, total_return_pct, total_pnl)}
"""
        
        # Save report
        report_file = signal_file.replace('.csv', '_pnl_report.md')
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📊 P&L Report saved to: {report_file}")
        
        # Print key metrics to console
        print("\n" + "="*80)
        print("PORTFOLIO PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Total Return: ${total_pnl:,.2f} ({total_return_pct:+.2f}%)")
        print(f"Capital Deployed: ${total_capital_deployed:,.2f}")
        print(f"Final Value: ${self.starting_capital + total_pnl:,.2f}")
        print(f"Win Rate: {win_rate:.1f}% ({win_count}/{win_count + loss_count})")
        print(f"Avg Win: ${avg_win_pnl:,.2f} | Avg Loss: ${avg_loss_pnl:,.2f}")
        print("="*80)
    
    def _generate_summary_text(self, win_rate: float, return_pct: float, pnl: float) -> str:
        """Generate summary interpretation"""
        
        if return_pct > 0:
            performance = "✅ **PROFITABLE**"
        else:
            performance = "❌ **UNPROFITABLE**"
        
        if win_rate >= 60:
            quality = "excellent"
        elif win_rate >= 50:
            quality = "good"
        else:
            quality = "needs improvement"
        
        return f"""
{performance}

The position-sized portfolio generated a **{return_pct:+.2f}% return** (${pnl:,.2f}) with a **{win_rate:.1f}% win rate**.

Performance is **{quality}** based on win rate. Position sizing {'amplified gains' if return_pct > 0 else 'reduced losses'} 
compared to equal-weighted strategy.

{'Continue with current strategy.' if return_pct > 0 else 'Consider adjusting position sizing or signal selection criteria.'}
"""


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python track_with_pnl.py <signal_file.csv> [starting_capital]")
        print("Example: python track_with_pnl.py signals_2026-01-29_04-39-38.csv 10000")
        sys.exit(1)
    
    signal_file = sys.argv[1]
    starting_capital = float(sys.argv[2]) if len(sys.argv) > 2 else 10000
    
    tracker = PositionTracker(starting_capital=starting_capital)
    results = tracker.track_with_positions(signal_file)
    
    print("\n✅ Tracking complete!")


if __name__ == "__main__":
    main()
