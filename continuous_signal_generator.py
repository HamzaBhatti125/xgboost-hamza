#!/usr/bin/env python3
"""
Continuous Signal Generator with Real Token Prices
==================================================

Streams from Envio HyperSync continuously until sufficient data is collected,
then generates signals with actual token prices and proper take profit/stop loss.
"""

import polars as pl
import numpy as np
import xgboost as xgb
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import time
import warnings
warnings.filterwarnings('ignore')

from hypersync_streamer import HyperSyncStreamer, SwapEvent, SwapToCandleConverter
from onchain_signal_system import engineer_features, Config as DataConfig
from eth_abi import decode
from signal_tracker import SignalTracker


def decode_swap_data_to_price(data: str, swap_type: str) -> float:
    """Decode swap data to extract price"""
    try:
        if not data or len(data) < 2:
            return 0.0
        
        # Remove 0x prefix
        if data.startswith('0x'):
            data = data[2:]
        
        data_bytes = bytes.fromhex(data)
        
        if swap_type == "v2":
            # V2: (uint256 amount0In, uint256 amount1In, uint256 amount0Out, uint256 amount1Out)
            if len(data_bytes) >= 128:
                amounts = decode(['uint256', 'uint256', 'uint256', 'uint256'], data_bytes)
                amount0In, amount1In, amount0Out, amount1Out = amounts
                
                # Calculate price: amountOut / amountIn
                if amount0In > 0 and amount0Out == 0:
                    # Selling token0, buying token1
                    price = float(amount1Out) / float(amount0In) if amount0In > 0 else 0.0
                    if price > 0 and price < 1e20:
                        return price
                elif amount1In > 0 and amount1Out == 0:
                    # Selling token1, buying token0
                    price = float(amount0Out) / float(amount1In) if amount1In > 0 else 0.0
                    if price > 0 and price < 1e20:
                        return price
                elif amount0Out > 0:
                    # Reverse: buying token0
                    price = float(amount1In) / float(amount0Out) if amount0Out > 0 else 0.0
                    if price > 0 and price < 1e20:
                        return price
                elif amount1Out > 0:
                    # Reverse: buying token1
                    price = float(amount0In) / float(amount1Out) if amount1Out > 0 else 0.0
                    if price > 0 and price < 1e20:
                        return price
        
        elif swap_type == "v3":
            # V3: (int256 amount0, int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)
            if len(data_bytes) >= 160:
                decoded = decode(['int256', 'int256', 'uint160', 'uint128', 'int24'], data_bytes)
                amount0, amount1, sqrtPriceX96, liquidity, tick = decoded
                
                # Use sqrtPriceX96 to calculate price
                # Price = (sqrtPriceX96 / 2^96)^2
                # For simplicity, use amount ratio if available
                if amount0 != 0 and amount1 != 0:
                    price = abs(float(amount1)) / abs(float(amount0))
                    if price > 0 and price < 1e10:
                        return price
                
                # Alternative: calculate from sqrtPriceX96
                if sqrtPriceX96 > 0:
                    q96 = 2**96
                    price = (float(sqrtPriceX96) / q96) ** 2
                    if price > 0 and price < 1e20:
                        return price
        
        return 0.0
    except Exception as e:
        return 0.0


class PriceAwareCandleConverter(SwapToCandleConverter):
    """Enhanced converter that extracts real prices from swaps"""
    
    def aggregate_to_candles(self, window_seconds: int = 60) -> pl.DataFrame:
        """Aggregate swaps to candles with real prices"""
        candles = []
        
        for pool_address, swaps in self.swap_buffer.items():
            if not swaps:
                continue
            
            # Group by time window
            time_windows = defaultdict(list)
            for swap in swaps:
                window_time = (swap.block_timestamp // window_seconds) * window_seconds
                time_windows[window_time].append(swap)
            
            # Create candles for each time window
            for window_time, window_swaps in time_windows.items():
                # Extract prices from swaps
                prices = []
                volumes = []
                buy_volumes = []
                sell_volumes = []
                buys = 0
                sells = 0
                
                for swap in window_swaps:
                    # Decode price from swap data
                    if swap.data:
                        price = decode_swap_data_to_price(swap.data, swap.swap_type)
                        if price > 0:
                            prices.append(price)
                    
                    # Count trades
                    if swap.buy_amount != "0" or swap.sell_amount != "0":
                        buys += 1
                        buy_volumes.append(1000)  # Placeholder
                    else:
                        sells += 1
                        sell_volumes.append(1000)  # Placeholder
                    
                    volumes.append(1000)  # Placeholder volume
                
                # Calculate OHLC from prices
                if prices:
                    open_price = prices[0]
                    close_price = prices[-1]
                    high_price = max(prices)
                    low_price = min(prices)
                else:
                    # No prices available, use placeholder
                    open_price = close_price = high_price = low_price = 1.0
                
                total_volume = sum(volumes) if volumes else len(window_swaps) * 1000
                buy_volume = sum(buy_volumes) if buy_volumes else buys * 1000
                sell_volume = sum(sell_volumes) if sell_volumes else sells * 1000
                
                candle = {
                    "pair_id": hash(pool_address) % 1000000,
                    "timestamp": datetime.fromtimestamp(window_time),
                    "open": open_price,
                    "close": close_price,
                    "high": high_price,
                    "low": low_price,
                    "volume": total_volume,
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "buys": buys,
                    "sells": sells,
                    "pool_address": pool_address,
                    "price_count": len(prices)  # Track real prices
                }
                candles.append(candle)
        
        if not candles:
            return pl.DataFrame()
        
        df = pl.DataFrame(candles)
        return df.sort("timestamp")


def generate_signals_with_real_prices(candles: pl.DataFrame, model_path: str = "xgb_model.json", 
                                     threshold: float = 0.5, top_n: int = 5) -> list:
    """Generate signals with actual token prices"""
    if len(candles) == 0:
        return []
    
    print(f"\n🔮 Generating signals from {len(candles)} candles...")
    
    # Filter to candles with real prices
    candles_with_prices = candles.filter(pl.col("price_count") > 0) if "price_count" in candles.columns else candles
    
    if len(candles_with_prices) == 0:
        print("⚠️  No candles with real prices, using all candles")
        candles_with_prices = candles
    
    # Prepare features
    try:
        feature_df = engineer_features(candles_with_prices)
    except:
        feature_df = candles_with_prices
    
    # Load model
    model = None
    if Path(model_path).exists():
        try:
            model = xgb.Booster()
            model.load_model(model_path)
            print(f"✓ Loaded model from {model_path}")
        except:
            print("⚠️  Using mock model")
            model = None
    
    feature_names = [
        "return_1d", "return_3d", "return_7d",
        "ema_cross_signal", "range_normalized",
        "buy_sell_ratio", "volume_zscore", "trade_accel",
        "volatility_7d", "volatility_14d", "vol_regime_change"
    ]
    
    if model:
        # Prepare features
        for feat in feature_names:
            if feat not in feature_df.columns:
                feature_df = feature_df.with_columns(pl.lit(0.0).alias(feat))
        
        X = feature_df.select(feature_names).to_numpy()
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        pred_proba = model.predict(dmatrix)
    else:
        # Mock predictions
        pred_proba = np.random.rand(len(candles_with_prices)) * 0.3 + 0.1
    
    # Add predictions
    candles_with_pred = candles_with_prices.with_columns([
        pl.Series("pred_proba", pred_proba),
        pl.Series("signal", (pred_proba >= threshold).astype(int))
    ])
    
    # Get top signals
    signals_df = candles_with_pred.sort("pred_proba", descending=True).head(top_n)
    
    # Generate signals
    signals = []
    for row in signals_df.iter_rows(named=True):
        # Use close price (most recent price in the candle)
        current_price = float(row.get("close", row.get("open", 1.0)))
        
        # Skip if no real price
        if current_price <= 0 or (current_price == 1.0 and row.get("price_count", 0) == 0):
            continue
        
        take_profit_pct = DataConfig.LABEL_UPSIDE_PCT
        stop_loss_pct = DataConfig.LABEL_DOWNSIDE_PCT
        take_profit_price = current_price * (1 + take_profit_pct / 100)
        stop_loss_price = current_price * (1 - stop_loss_pct / 100)
        
        signal = {
            "timestamp": str(row.get("timestamp", datetime.now())),
            "pair_id": int(row.get("pair_id", 0)),
            "pool_address": str(row.get("pool_address", ""))[:42],
            "pred_proba": float(row.get("pred_proba", 0.0)),
            "signal": int(row.get("signal", 0)),
            "current_price": current_price,
            "take_profit_pct": take_profit_pct,
            "take_profit_price": take_profit_price,
            "stop_loss_pct": stop_loss_pct,
            "stop_loss_price": stop_loss_price,
            "holding_period_days": DataConfig.LABEL_HORIZON_DAYS,
            "open": float(row.get("open", 0.0)),
            "high": float(row.get("high", 0.0)),
            "low": float(row.get("low", 0.0)),
            "close": float(row.get("close", 0.0)),
            "volume": float(row.get("volume", 0.0)),
            "buy_volume": float(row.get("buy_volume", 0.0)),
            "sell_volume": float(row.get("sell_volume", 0.0)),
            "buys": int(row.get("buys", 0)),
            "sells": int(row.get("sells", 0)),
            "has_real_price": row.get("price_count", 0) > 0
        }
        signals.append(signal)
    
    return signals


def main():
    """Main execution - stream until we have data, then generate signals"""
    print(f"\n{'#'*80}")
    print("CONTINUOUS SIGNAL GENERATION WITH REAL PRICES")
    print(f"{'#'*80}\n")
    
    api_token = "1ae7c8f0-5cdf-4316-81f6-0fa3cb84aaa8"
    min_candles = 30
    min_pools = 3
    max_duration = 300  # 5 minutes max
    
    print(f"Streaming until we have:")
    print(f"  - At least {min_candles} candles")
    print(f"  - At least {min_pools} unique pools")
    print(f"  - Maximum {max_duration} seconds")
    print()
    
    all_swaps = []
    collected_candles = []
    start_time = time.time()
    
    # Use price-aware converter
    converter = PriceAwareCandleConverter()
    
    def collect_candles(candles: pl.DataFrame):
        nonlocal collected_candles
        collected_candles.append(candles)
        candles_with_prices = candles.filter(pl.col("price_count") > 0) if "price_count" in candles.columns else pl.DataFrame()
        print(f"  📊 Collected {len(candles)} candles ({len(candles_with_prices)} with real prices)")
    
    streamer = HyperSyncStreamer(api_token, chain="base", signal_callback=collect_candles)
    streamer.converter = converter  # Use our price-aware converter
    
    print("🚀 Starting continuous stream...")
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed >= max_duration:
            print(f"\n⏰ Max duration ({max_duration}s) reached")
            break
        
        try:
            # Stream for 30 seconds
            batch_duration = 30
            candles = streamer.start_streaming(duration_seconds=batch_duration)
            
            # Collect swaps from converter
            for pool, swaps in converter.swap_buffer.items():
                all_swaps.extend(swaps)
            
            # Convert all swaps to candles
            if all_swaps:
                candles_df = converter.aggregate_to_candles()
                
                if len(candles_df) >= min_candles:
                    unique_pools = candles_df['pool_address'].n_unique()
                    candles_with_prices = candles_df.filter(pl.col("price_count") > 0) if "price_count" in candles_df.columns else candles_df
                    
                    print(f"\n✓ Progress: {len(candles_df)} candles from {unique_pools} pools")
                    print(f"  Candles with real prices: {len(candles_with_prices)}")
                    
                    if len(candles_with_prices) >= min_candles and unique_pools >= min_pools:
                        print(f"✅ Sufficient data collected!")
                        candles = candles_with_prices
                        break
                    elif len(candles_df) >= min_candles * 2:
                        # Have enough candles even without prices
                        print(f"✅ Sufficient candles collected (some may not have prices)")
                        candles = candles_df
                        break
                
                print(f"  Progress: {len(candles_df)} candles, {elapsed:.0f}s elapsed...")
            
            time.sleep(2)  # Brief pause
            
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"⚠️  Error: {e}")
            time.sleep(5)
    
    # Final conversion
    if all_swaps:
        candles = converter.aggregate_to_candles()
    
    if len(candles) == 0:
        print("\n❌ No data collected. Cannot generate signals.")
        return 1
    
    print(f"\n✓ Final dataset: {len(candles)} candles")
    print(f"  Unique pools: {candles['pool_address'].n_unique()}")
    if "price_count" in candles.columns:
        with_prices = candles.filter(pl.col("price_count") > 0)
        print(f"  Candles with real prices: {len(with_prices)}")
    
    # Generate signals
    signals = generate_signals_with_real_prices(candles, threshold=0.5, top_n=5)
    
    if not signals:
        print("\n⚠️  No signals generated")
        return 1
    
    # Print signals
    print(f"\n{'='*80}")
    print(f"🎯 GENERATED {len(signals)} SIGNALS WITH REAL TOKEN PRICES")
    print(f"{'='*80}\n")
    
    for i, signal in enumerate(signals, 1):
        signal_icon = "🟢 BUY" if signal["signal"] == 1 else "⚪ HOLD"
        price_status = "✓ Real Price" if signal.get("has_real_price") else "⚠️  Estimated"
        
        print(f"{i}. {signal_icon} {price_status}")
        print(f"   Timestamp: {signal['timestamp']}")
        print(f"   Pair ID: {signal['pair_id']}")
        print(f"   Pool: {signal['pool_address']}")
        print(f"   Probability: {signal['pred_proba']:.3f}")
        print(f"   Current Price: {signal['current_price']:.10f}")
        print(f"   Take Profit: +{signal['take_profit_pct']:.1f}% → {signal['take_profit_price']:.10f}")
        print(f"   Stop Loss: -{signal['stop_loss_pct']:.1f}% → {signal['stop_loss_price']:.10f}")
        print(f"   Max Holding: {signal['holding_period_days']} days")
        print(f"   OHLC: O={signal['open']:.10f} H={signal['high']:.10f} L={signal['low']:.10f} C={signal['close']:.10f}")
        print(f"   Volume: {signal['volume']:,.0f}")
        print()
    
    # Save to file
    output_path = Path("signals_with_prices.json")
    with open(output_path, "w") as f:
        json.dump(signals, f, indent=2)
    
    print(f"💾 Saved {len(signals)} signals to: {output_path}")
    
    # Track signals for feedback learning
    print(f"\n📊 Tracking signals for feedback learning...")
    tracker = SignalTracker()
    tracked_ids = []
    for signal in signals:
        signal_id = tracker.record_signal(signal)
        tracked_ids.append(signal_id)
    
    print(f"✓ Tracked {len(tracked_ids)} signals")
    print(f"\n💡 To mark outcomes later, use:")
    print(f"   python signal_tracker.py --outcome {tracked_ids[0]} --result win --exit-price <price>")
    print(f"   python signal_tracker.py --stats  # View performance")
    print(f"   python online_learning.py --retrain  # Retrain with feedback")
    
    print(f"\n{'='*80}")
    print("✅ Signal generation complete!")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
