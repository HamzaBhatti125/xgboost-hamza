#!/usr/bin/env python3
"""
Market Regime Analyzer
======================

Analyzes market conditions to identify regimes and provide actionable signals
for risk management, exits, and reentry opportunities.
"""

import pickle
import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_CALM = "bull_calm"           # Uptrend, low volatility
    BULL_VOLATILE = "bull_volatile"   # Uptrend, high volatility
    BEAR_CALM = "bear_calm"           # Downtrend, low volatility
    BEAR_VOLATILE = "bear_volatile"   # Downtrend, high volatility
    SIDEWAYS = "sideways"             # Range-bound
    PANIC = "panic"                   # Extreme volatility, sharp decline
    RECOVERY = "recovery"             # Bounce from panic


@dataclass
class RegimeSignals:
    """Actionable signals based on regime"""
    regime: MarketRegime
    confidence: float
    
    # Risk management
    emergency_exit: bool
    reduce_exposure: bool
    increase_exposure: bool
    
    # Support/resistance
    support_level: Optional[float]
    resistance_level: Optional[float]
    
    # Metrics
    trend_strength: float
    volatility_percentile: float
    drawdown_pct: float
    volume_surge: float
    
    # Reentry signals
    reentry_opportunity: bool
    reentry_reason: str
    
    # Market breadth
    pct_pairs_up: float
    pct_pairs_down: float
    


class MarketRegimeAnalyzer:
    """Analyzes market conditions and identifies regimes"""
    
    def __init__(self, cache_file: str = "live_candles_cache.pkl"):
        self.cache_file = cache_file
        self.cache = None
        self.lookback_periods = {
            'short': 16,   # 4 hours
            'medium': 96,  # 24 hours
            'long': 288,   # 72 hours
        }
        
    def load_cache(self) -> Dict:
        """Load candle cache with corruption recovery"""
        try:
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            return self.cache
        except (EOFError, pickle.UnpicklingError) as e:
            # Cache is corrupted, try backup
            backup_file = self.cache_file + '.backup'
            if Path(backup_file).exists():
                logger.warning(f"⚠️  Cache corrupted, loading from backup...")
                try:
                    with open(backup_file, 'rb') as f:
                        self.cache = pickle.load(f)
                    # Restore the corrupted file
                    import shutil
                    shutil.copy(backup_file, self.cache_file)
                    logger.info(f"✅ Cache restored from backup")
                    return self.cache
                except Exception as backup_err:
                    logger.error(f"❌ Backup also corrupted: {backup_err}")
            
            # No backup or backup failed
            raise RuntimeError(
                f"Cache file is corrupted and no valid backup found.\n"
                f"Please delete {self.cache_file} and run the live signal generator to rebuild the cache."
            )
    
    def analyze_market(self) -> RegimeSignals:
        """Main analysis function - returns current market regime and signals"""
        
        if self.cache is None:
            self.load_cache()
        
        # Calculate market-wide metrics
        market_metrics = self._calculate_market_metrics()
        
        # Identify regime
        regime, confidence = self._identify_regime(market_metrics)
        
        # Generate actionable signals
        signals = self._generate_signals(regime, market_metrics)
        
        return signals
    
    def _calculate_market_metrics(self) -> Dict:
        """Calculate aggregate market metrics across all pairs"""
        
        metrics = {
            'timestamp': datetime.now(),
            'total_pairs': len(self.cache),
            'pairs_analyzed': 0,
        }
        
        # Aggregate data from all pairs
        all_returns_4h = []
        all_returns_24h = []
        all_volatility_4h = []
        all_volume_4h = []
        all_latest_prices = []
        
        pairs_up_4h = 0
        pairs_down_4h = 0
        
        for pair_address, pair_df in self.cache.items():
            if len(pair_df) < 96:  # Need at least 24 hours of data
                continue
            
            pair_df = pair_df.sort('timestamp')
            
            # Latest candles
            latest_16 = pair_df.tail(16)  # 4 hours
            latest_96 = pair_df.tail(96)  # 24 hours
            
            if len(latest_16) < 16 or len(latest_96) < 96:
                continue
            
            metrics['pairs_analyzed'] += 1
            
            # Calculate returns
            latest_close = latest_16['close'][-1]
            close_4h_ago = latest_16['close'][0]
            close_24h_ago = latest_96['close'][0]
            
            return_4h = (latest_close - close_4h_ago) / close_4h_ago
            return_24h = (latest_close - close_24h_ago) / close_24h_ago
            
            all_returns_4h.append(return_4h)
            all_returns_24h.append(return_24h)
            
            if return_4h > 0:
                pairs_up_4h += 1
            elif return_4h < 0:
                pairs_down_4h += 1
            
            # Volatility (std of returns)
            candle_returns = []
            for i in range(1, len(latest_96)):
                ret = (latest_96['close'][i] - latest_96['close'][i-1]) / latest_96['close'][i-1]
                candle_returns.append(ret)
            
            volatility = np.std(candle_returns) if candle_returns else 0
            all_volatility_4h.append(volatility)
            
            # Volume
            total_volume = latest_16['volume_token1'].sum()
            all_volume_4h.append(total_volume)
            
            all_latest_prices.append(latest_close)
        
        # Aggregate statistics
        metrics['median_return_4h'] = np.median(all_returns_4h) if all_returns_4h else 0
        metrics['median_return_24h'] = np.median(all_returns_24h) if all_returns_24h else 0
        metrics['mean_return_4h'] = np.mean(all_returns_4h) if all_returns_4h else 0
        metrics['mean_return_24h'] = np.mean(all_returns_24h) if all_returns_24h else 0
        
        metrics['median_volatility'] = np.median(all_volatility_4h) if all_volatility_4h else 0
        metrics['volatility_95th'] = np.percentile(all_volatility_4h, 95) if all_volatility_4h else 0
        
        metrics['pct_pairs_up'] = (pairs_up_4h / metrics['pairs_analyzed'] * 100) if metrics['pairs_analyzed'] > 0 else 0
        metrics['pct_pairs_down'] = (pairs_down_4h / metrics['pairs_analyzed'] * 100) if metrics['pairs_analyzed'] > 0 else 0
        
        # Volume surge detection
        metrics['volume_surge'] = np.mean(all_volume_4h) / (np.median(all_volume_4h) + 1e-10) if all_volume_4h else 1.0
        
        # Drawdown calculation (from 24h high)
        metrics['drawdown_from_24h_high'] = self._calculate_market_drawdown()
        
        # Trend strength (0-100)
        trend_strength = abs(metrics['mean_return_24h']) * 100
        metrics['trend_strength'] = min(trend_strength, 100)
        
        return metrics
    
    def _calculate_market_drawdown(self) -> float:
        """Calculate average drawdown from 24h high across pairs"""
        
        drawdowns = []
        
        for pair_address, pair_df in self.cache.items():
            if len(pair_df) < 96:
                continue
            
            latest_96 = pair_df.tail(96).sort('timestamp')
            
            if len(latest_96) < 96:
                continue
            
            highest = latest_96['high'].max()
            current = latest_96['close'][-1]
            
            if highest > 0:
                drawdown = (current - highest) / highest
                drawdowns.append(drawdown)
        
        return np.median(drawdowns) * 100 if drawdowns else 0  # Return as percentage
    
    def _identify_regime(self, metrics: Dict) -> Tuple[MarketRegime, float]:
        """Identify market regime based on metrics"""
        
        return_4h = metrics['median_return_4h']
        return_24h = metrics['median_return_24h']
        volatility = metrics['median_volatility']
        vol_95th = metrics['volatility_95th']
        drawdown = metrics['drawdown_from_24h_high']
        volume_surge = metrics['volume_surge']
        pct_pairs_down = metrics['pct_pairs_down']
        
        confidence = 0.0
        regime = MarketRegime.SIDEWAYS
        
        # PANIC: Extreme conditions
        # Trigger PANIC if:
        # 1. Severe drawdown (>12%) regardless of breadth, OR
        # 2. Moderate drawdown (>10%) + majority pairs down (>55%), OR
        # 3. Extreme volatility + sharp 4h decline
        if drawdown < -12 or \
           (drawdown < -10 and pct_pairs_down > 55) or \
           (volatility > vol_95th * 0.9 and return_4h < -0.05):
            regime = MarketRegime.PANIC
            confidence = min(abs(drawdown) / 15, 1.0)  # Scale confidence based on severity
        
        # RECOVERY: Bounce from panic
        elif drawdown < -5 and return_4h > 0.02 and pct_pairs_down < 40:
            regime = MarketRegime.RECOVERY 
            confidence = 0.7
        
        # BULL: Positive returns
        elif return_24h > 0.02:
            if volatility > vol_95th * 0.7:
                regime = MarketRegime.BULL_VOLATILE
                confidence = 0.75
            else:
                regime = MarketRegime.BULL_CALM
                confidence = 0.85
        
        # BEAR: Negative returns
        elif return_24h < -0.02:
            if volatility > vol_95th * 0.7:
                regime = MarketRegime.BEAR_VOLATILE
                confidence = 0.75
            else:
                regime = MarketRegime.BEAR_CALM
                confidence = 0.8
        
        # SIDEWAYS: Range-bound
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.6
        
        return regime, confidence
    
    def _generate_signals(self, regime: MarketRegime, metrics: Dict) -> RegimeSignals:
        """Generate actionable signals based on regime"""
        
        # Extract metrics
        return_4h = metrics['median_return_4h']
        return_24h = metrics['median_return_24h']
        volatility = metrics['median_volatility']
        drawdown = metrics['drawdown_from_24h_high']
        pct_pairs_up = metrics['pct_pairs_up']
        pct_pairs_down = metrics['pct_pairs_down']
        trend_strength = metrics['trend_strength']
        volume_surge = metrics['volume_surge']
        
        # Initialize signals
        emergency_exit = False
        reduce_exposure = False
        increase_exposure = False
        reentry_opportunity = False
        reentry_reason = ""
        
        # Support/resistance (simplified - could be enhanced with clustering)
        support_level = None
        resistance_level = None
        
        # Regime-specific logic
        if regime == MarketRegime.PANIC:
            emergency_exit = True
            reduce_exposure = True
            reentry_opportunity = False
            reentry_reason = "Wait for panic to subside"
        
        elif regime == MarketRegime.RECOVERY:
            emergency_exit = False
            reduce_exposure = False
            increase_exposure = True
            reentry_opportunity = True
            reentry_reason = "Strong bounce from oversold, high R:R"
        
        elif regime == MarketRegime.BULL_CALM:
            emergency_exit = False
            reduce_exposure = False
            increase_exposure = True
            reentry_opportunity = True
            reentry_reason = "Stable uptrend, favorable conditions"
        
        elif regime == MarketRegime.BULL_VOLATILE:
            emergency_exit = False
            reduce_exposure = True  # Scale down position sizes
            increase_exposure = False
            reentry_opportunity = True
            reentry_reason = "Uptrend but volatile - use tighter stops"
        
        elif regime == MarketRegime.BEAR_VOLATILE:
            emergency_exit = False
            reduce_exposure = True
            increase_exposure = False
            reentry_opportunity = False
            reentry_reason = "High volatility downtrend - wait for stability"
        
        elif regime == MarketRegime.BEAR_CALM:
            emergency_exit = False
            reduce_exposure = True
            increase_exposure = False
            reentry_opportunity = False
            reentry_reason = "Steady downtrend - preserve capital"
        
        elif regime == MarketRegime.SIDEWAYS:
            emergency_exit = False
            reduce_exposure = False
            increase_exposure = False
            reentry_opportunity = True
            reentry_reason = "Range-bound market - normal operations"
        
        # Calculate volatility percentile (simplified)
        volatility_percentile = min(volatility * 1000, 100)  # Scale to 0-100
        
        # Calculate confidence for regime identification
        confidence = self._identify_regime(metrics)[1]
        
        return RegimeSignals(
            regime=regime,
            confidence=confidence,
            emergency_exit=emergency_exit,
            reduce_exposure=reduce_exposure,
            increase_exposure=increase_exposure,
            support_level=support_level,
            resistance_level=resistance_level,
            trend_strength=trend_strength,
            volatility_percentile=volatility_percentile,
            drawdown_pct=drawdown,
            volume_surge=volume_surge,
            reentry_opportunity=reentry_opportunity,
            reentry_reason=reentry_reason,
            pct_pairs_up=pct_pairs_up,
            pct_pairs_down=pct_pairs_down,
        )
    
    def print_analysis(self, signals: RegimeSignals):
        """Print formatted analysis"""
        
        print(f"\n{'='*80}")
        print("MARKET REGIME ANALYSIS")
        print(f"{'='*80}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n📊 Current Regime: {signals.regime.value.upper()}")
        print(f"   Confidence: {signals.confidence*100:.1f}%")
        
        print(f"\n📈 Market Metrics:")
        print(f"   Trend Strength: {signals.trend_strength:.2f}/100")
        print(f"   Volatility Percentile: {signals.volatility_percentile:.1f}%")
        print(f"   Drawdown from 24h High: {signals.drawdown_pct:.2f}%")
        print(f"   Volume Surge: {signals.volume_surge:.2f}x")
        
        print(f"\n🎯 Market Breadth:")
        print(f"   Pairs Up (4h): {signals.pct_pairs_up:.1f}%")
        print(f"   Pairs Down (4h): {signals.pct_pairs_down:.1f}%")
        
        print(f"\n⚠️  Risk Management:")
        if signals.emergency_exit:
            print(f"   🚨 EMERGENCY EXIT: Close all positions immediately!")
        elif signals.reduce_exposure:
            print(f"   ⚠️  REDUCE EXPOSURE: Scale down position sizes by 50%")
        elif signals.increase_exposure:
            print(f"   ✅ INCREASE EXPOSURE: Favorable conditions for larger positions")
        else:
            print(f"   ➖ NEUTRAL: Maintain current exposure")
        
        print(f"\n🔄 Reentry Signals:")
        if signals.reentry_opportunity:
            print(f"   ✅ REENTRY OPPORTUNITY")
            print(f"   Reason: {signals.reentry_reason}")
        else:
            print(f"   ❌ NO REENTRY")
            print(f"   Reason: {signals.reentry_reason}")
        
        if signals.support_level:
            print(f"\n📍 Key Levels:")
            print(f"   Support: ${signals.support_level:.2f}")
            print(f"   Resistance: ${signals.resistance_level:.2f}")
        
        print(f"\n{'='*80}\n")


def main():
    """Run market regime analysis"""
    
    analyzer = MarketRegimeAnalyzer()
    
    print("Loading cache...")
    analyzer.load_cache()
    
    print(f"Analyzing {len(analyzer.cache):,} pairs...")
    
    signals = analyzer.analyze_market()
    
    analyzer.print_analysis(signals)
    
    # Save to file for integration with live system
    output = {
        'timestamp': datetime.now().isoformat(),
        'regime': signals.regime.value,
        'confidence': signals.confidence,
        'emergency_exit': signals.emergency_exit,
        'reduce_exposure': signals.reduce_exposure,
        'increase_exposure': signals.increase_exposure,
        'reentry_opportunity': signals.reentry_opportunity,
        'metrics': {
            'trend_strength': signals.trend_strength,
            'volatility_percentile': signals.volatility_percentile,
            'drawdown_pct': signals.drawdown_pct,
            'volume_surge': signals.volume_surge,
            'pct_pairs_up': signals.pct_pairs_up,
            'pct_pairs_down': signals.pct_pairs_down,
        }
    }
    
    import json
    with open('market_regime.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("✅ Regime analysis saved to market_regime.json")


if __name__ == "__main__":
    main()
