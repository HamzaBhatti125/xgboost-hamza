#!/usr/bin/env python3
"""
Signal Tracking and Feedback System
====================================

Tracks generated signals and their outcomes to enable:
1. Online learning (retrain model with new outcomes)
2. Performance monitoring
3. Threshold optimization
4. Future RL implementation

Usage:
    # Generate and track signals
    python signal_tracker.py --generate
    
    # Mark signal outcome
    python signal_tracker.py --outcome SIGNAL_ID --result win --exit_price 1.08
    
    # Retrain with feedback
    python signal_tracker.py --retrain
"""

import polars as pl
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse


class SignalTracker:
    """Track signals and their outcomes for feedback learning"""
    
    def __init__(self, db_path: str = "signal_history.jsonl"):
        self.db_path = Path(db_path)
        self.signals = []
        self._load_history()
    
    def _load_history(self):
        """Load existing signal history"""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                for line in f:
                    if line.strip():
                        self.signals.append(json.loads(line))
    
    def _save_signal(self, signal: Dict):
        """Append signal to history file"""
        with open(self.db_path, 'a') as f:
            f.write(json.dumps(signal) + '\n')
        self.signals.append(signal)
    
    def record_signal(self, signal_data: Dict) -> str:
        """Record a new signal and return its ID"""
        signal_id = f"{signal_data.get('timestamp', datetime.now().isoformat())}_{signal_data.get('pair_id', 0)}_{signal_data.get('pool_address', '')[:10]}"
        
        signal = {
            "signal_id": signal_id,
            "timestamp": signal_data.get("timestamp", datetime.now().isoformat()),
            "pair_id": signal_data.get("pair_id", 0),
            "pool_address": signal_data.get("pool_address", ""),
            "pred_proba": signal_data.get("pred_proba", 0.0),
            "signal": signal_data.get("signal", 0),
            "current_price": signal_data.get("current_price", 0.0),
            "take_profit_price": signal_data.get("take_profit_price", 0.0),
            "stop_loss_price": signal_data.get("stop_loss_price", 0.0),
            "entry_price": signal_data.get("current_price", 0.0),  # Entry at current price
            "entry_timestamp": datetime.now().isoformat(),
            "status": "pending",  # pending, executed, closed
            "outcome": None,  # win, loss, or None
            "exit_price": None,
            "exit_timestamp": None,
            "return_pct": None,
            "holding_days": None,
            "exit_reason": None,  # take_profit, stop_loss, timeout, manual
        }
        
        self._save_signal(signal)
        return signal_id
    
    def record_outcome(self, signal_id: str, outcome: str, exit_price: float, 
                      exit_reason: str = "manual", holding_days: float = None):
        """Record the outcome of a signal"""
        # Find signal
        signal = None
        for s in self.signals:
            if s.get("signal_id") == signal_id:
                signal = s
                break
        
        if not signal:
            print(f"⚠️  Signal {signal_id} not found")
            return False
        
        # Calculate return
        entry_price = signal.get("entry_price", 0)
        if entry_price > 0:
            return_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            return_pct = 0.0
        
        # Update signal
        signal["status"] = "closed"
        signal["outcome"] = outcome  # "win" or "loss"
        signal["exit_price"] = exit_price
        signal["exit_timestamp"] = datetime.now().isoformat()
        signal["return_pct"] = return_pct
        signal["exit_reason"] = exit_reason
        if holding_days:
            signal["holding_days"] = holding_days
        
        # Save updated history
        self._save_all()
        return True
    
    def _save_all(self):
        """Save all signals to file"""
        with open(self.db_path, 'w') as f:
            for signal in self.signals:
                f.write(json.dumps(signal) + '\n')
    
    def get_pending_signals(self) -> List[Dict]:
        """Get all pending signals waiting for outcomes"""
        return [s for s in self.signals if s.get("status") == "pending"]
    
    def get_completed_signals(self) -> List[Dict]:
        """Get all completed signals with outcomes"""
        return [s for s in self.signals if s.get("status") == "closed"]
    
    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics"""
        completed = self.get_completed_signals()
        
        if len(completed) == 0:
            return {
                "total_signals": len(self.signals),
                "completed": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "total_return": 0.0
            }
        
        wins = [s for s in completed if s.get("outcome") == "win"]
        losses = [s for s in completed if s.get("outcome") == "loss"]
        
        returns = [s.get("return_pct", 0) for s in completed]
        avg_return = sum(returns) / len(returns) if returns else 0.0
        
        return {
            "total_signals": len(self.signals),
            "completed": len(completed),
            "pending": len(self.get_pending_signals()),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(completed) if completed else 0.0,
            "avg_return": avg_return,
            "total_return": sum(returns),
            "avg_win": sum([s.get("return_pct", 0) for s in wins]) / len(wins) if wins else 0.0,
            "avg_loss": sum([s.get("return_pct", 0) for s in losses]) / len(losses) if losses else 0.0,
        }
    
    def export_for_retraining(self, output_path: str = "feedback_data.parquet") -> pl.DataFrame:
        """Export completed signals for model retraining"""
        completed = self.get_completed_signals()
        
        if len(completed) == 0:
            return pl.DataFrame()
        
        # Convert to DataFrame
        data = []
        for signal in completed:
            data.append({
                "pair_id": signal.get("pair_id", 0),
                "timestamp": signal.get("timestamp", ""),
                "pred_proba": signal.get("pred_proba", 0.0),
                "actual_outcome": 1 if signal.get("outcome") == "win" else 0,
                "return_pct": signal.get("return_pct", 0.0),
                "exit_reason": signal.get("exit_reason", ""),
                "holding_days": signal.get("holding_days", 0),
            })
        
        df = pl.DataFrame(data)
        df.write_parquet(output_path)
        return df


def main():
    parser = argparse.ArgumentParser(description="Track signals and outcomes for feedback learning")
    parser.add_argument("--generate", action="store_true", help="Generate and track new signals")
    parser.add_argument("--outcome", type=str, help="Signal ID to mark outcome")
    parser.add_argument("--result", choices=["win", "loss"], help="Outcome: win or loss")
    parser.add_argument("--exit-price", type=float, help="Exit price")
    parser.add_argument("--exit-reason", default="manual", help="Exit reason")
    parser.add_argument("--holding-days", type=float, help="Holding period in days")
    parser.add_argument("--stats", action="store_true", help="Show performance statistics")
    parser.add_argument("--retrain", action="store_true", help="Export data for retraining")
    parser.add_argument("--pending", action="store_true", help="Show pending signals")
    
    args = parser.parse_args()
    
    tracker = SignalTracker()
    
    if args.generate:
        # Generate signals and track them
        from live_signal_generator import generate_signals_with_real_prices
        from hypersync_streamer import HyperSyncStreamer
        
        print("Generating and tracking signals...")
        # This would integrate with your signal generation
        # For now, just show the concept
        print("✓ Signal tracking ready. Use with signal generation pipeline.")
        
    elif args.outcome:
        if not args.result or not args.exit_price:
            print("❌ Error: --result and --exit-price required")
            return 1
        
        success = tracker.record_outcome(
            args.outcome,
            args.result,
            args.exit_price,
            args.exit_reason,
            args.holding_days
        )
        
        if success:
            print(f"✓ Recorded outcome: {args.result} for signal {args.outcome}")
        else:
            print(f"❌ Failed to record outcome")
            return 1
    
    elif args.stats:
        stats = tracker.get_performance_stats()
        print(f"\n{'='*80}")
        print("SIGNAL PERFORMANCE STATISTICS")
        print(f"{'='*80}\n")
        print(f"Total Signals: {stats['total_signals']}")
        print(f"Completed: {stats['completed']}")
        print(f"Pending: {stats['pending']}")
        print(f"\nResults:")
        print(f"  Wins: {stats['wins']}")
        print(f"  Losses: {stats['losses']}")
        print(f"  Win Rate: {stats['win_rate']:.1%}")
        print(f"\nReturns:")
        print(f"  Average Return: {stats['avg_return']:+.2f}%")
        print(f"  Total Return: {stats['total_return']:+.2f}%")
        if stats['wins'] > 0:
            print(f"  Average Win: {stats['avg_win']:+.2f}%")
        if stats['losses'] > 0:
            print(f"  Average Loss: {stats['avg_loss']:+.2f}%")
    
    elif args.pending:
        pending = tracker.get_pending_signals()
        print(f"\n{'='*80}")
        print(f"PENDING SIGNALS ({len(pending)})")
        print(f"{'='*80}\n")
        for signal in pending:
            print(f"ID: {signal['signal_id']}")
            print(f"  Pair: {signal['pair_id']} | Pool: {signal['pool_address'][:20]}...")
            print(f"  Entry: {signal['entry_price']:.8f} | TP: {signal['take_profit_price']:.8f} | SL: {signal['stop_loss_price']:.8f}")
            print(f"  Generated: {signal['timestamp']}")
            print()
    
    elif args.retrain:
        df = tracker.export_for_retraining()
        if len(df) > 0:
            print(f"✓ Exported {len(df)} completed signals for retraining")
            print(f"  Saved to: feedback_data.parquet")
        else:
            print("⚠️  No completed signals to export")
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    exit(main())
