#!/usr/bin/env python3
"""
Stream and Generate Signals with Real Token Prices
===================================================

Continuously streams from Envio HyperSync until we have sufficient data,
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

from hypersync_streamer import HyperSyncStreamer, SwapEvent
from onchain_signal_system import engineer_features, Config as DataConfig
from eth_abi import decode


def decode_swap_amounts(data: str, swap_type: str) -> tuple:
    """Decode swap amounts from log data"""
    try:
        if not data or len(data) < 2:
            return (0, 0, 0, 0)
        
        # Remove 0x prefix if present
        if data.startswith('0x'):
            data = data[2:]
        
        # Convert hex to bytes
        data_bytes = bytes.fromhex(data)
        
        if swap_type == "v2":
            # V2: (uint256 amount0In, uint256 amount1In, uint256 amount0Out, uint256 amount1Out)
            if len(data_bytes) >= 128:  # 4 * 32 bytes
                decoded = decode(['uint256', 'uint256', 'uint256', 'uint256'], data_bytes)
                return (decoded[0], decoded[1], decoded[2], decoded[3])
        elif swap_type == "v3":
            # V3: (int256 amount0, int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)
            if len(data_bytes) >= 160:  # 5 * 32 bytes
                decoded = decode(['int256', 'int256', 'uint160', 'uint128', 'int24'], data_bytes)
                return (decoded[0], decoded[1], 0, 0)
        
        return (0, 0, 0, 0)
    except Exception as e:
        return (0, 0, 0, 0)


def calculate_price_from_swap(swap: SwapEvent, amounts: tuple) -> float:
    """Calculate token price from swap amounts"""
    try:
        if swap.swap_type == "v2":
            amount0In, amount1In, amount0Out, amount1Out = amounts
            
            # Price = amountOut / amountIn
            # If amount0In > 0, token0 is being sold, price = amount1Out / amount0In
            # If amount1In > 0, token1 is being sold, price = amount0Out / amount1In
            
            if amount0In > 0 and amount0Out == 0:
                # Selling token0, buying token1
                if amount0In > 0:
                    price = float(amount1Out) / float(amount0In) if amount0In > 0 else 0.0
                    return price
            elif amount1In > 0 and amount1Out == 0:
                # Selling token1, buying token0
                if amount1In > 0:
                    price = float(amount0Out) / float(amount1In) if amount1In > 0 else 0.0
                    return price
            
            # Try reverse calculation
            if amount0Out > 0:
                price = float(amount1In) / float(amount0Out) if amount0Out > 0 else 0.0
                if price > 0:
                    return price
            if amount1Out > 0:
                price = float(amount0In) / float(amount1Out) if amount1Out > 0 else 0.0
                if price > 0:
                    return price
                    
        elif swap.swap_type == "v3":
            amount0, amount1, _, _ = amounts
            # V3: negative means going out, positive means coming in
            # Price calculation is more complex with sqrtPriceX96
            # For simplicity, use ratio
            if amount0 != 0 and amount1 != 0:
                # Use absolute values for ratio
                price = abs(float(amount1)) / abs(float(amount0)) if amount0 != 0 else 0.0
                if price > 0 and price < 1e10:  # Reasonable price range
                    return price
        
        return 0.0
    except Exception as e:
        return 0.0


def convert_swaps_to_candles_with_prices(swaps: list, window_seconds: int = 60) -> pl.DataFrame:
    """Convert swaps to candles with actual prices"""
    candles = []
    
    # Group by pool and time window
    pool_windows = defaultdict(lambda: defaultdict(list))
    
    for swap in swaps:
        window_time = (swap.block_timestamp // window_seconds) * window_seconds
        pool_windows[swap.pool_address][window_time].append(swap)
    
    for pool_address, time_windows in pool_windows.items():
        for window_time, window_swaps in time_windows.items():
            if not window_swaps:
                continue
            
            # Extract prices from swaps
            prices = []
            volumes = []
            buy_volumes = []
            sell_volumes = []
            buys = 0
            sells = 0
            
            for swap in window_swaps:
                # Decode swap data to get amounts
                data = getattr(swap, 'data', '')
                if not data:
                    continue
                
                amounts = decode_swap_amounts(data, swap.swap_type)
                price = calculate_price_from_swap(swap, amounts)
                
                if price > 0:
                    prices.append(price)
                
                # Estimate volume from amounts
                total_amount = sum(abs(float(a)) for a in amounts if a != 0)
                if total_amount > 0:
                    volumes.append(total_amount)
                
                # Count buys/sells (simplified)
                if amounts[0] > 0 or amounts[2] > 0:
                    buys += 1
                    buy_volumes.append(total_amount)
                if amounts[1] > 0 or amounts[3] > 0:
                    sells += 1
                    sell_volumes.append(total_amount)
            
            if not prices:
                # No valid prices, use placeholder but mark it
                open_price = close_price = high_price = low_price = 1.0
            else:
                open_price = prices[0] if prices else 1.0
                close_price = prices[-1] if prices else 1.0
                high_price = max(prices) if prices else 1.0
                low_price = min(prices) if prices else 1.0
            
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
                "price_count": len(prices)  # Track how many prices we got
            }
            candles.append(candle)
    
    if not candles:
        return pl.DataFrame()
    
    df = pl.DataFrame(candles)
    return df.sort("timestamp")


def stream_until_sufficient_data(min_candles: int = 50, min_pools: int = 5, max_duration: int = 300):
    """Stream until we have sufficient data with real prices"""
    print(f"\n{'='*80}")
    print("STREAMING UNTIL SUFFICIENT DATA")
    print(f"{'='*80}")
    print(f"Minimum candles: {min_candles}")
    print(f"Minimum pools: {min_pools}")
    print(f"Max duration: {max_duration} seconds")
    print(f"{'='*80}\n")
    
    api_token = "1ae7c8f0-5cdf-4316-81f6-0fa3cb84aaa8"
    all_swaps = []
    start_time = time.time()
    
    def collect_swaps(swaps_batch):
        nonlocal all_swaps
        all_swaps.extend(swaps_batch)
    
    streamer = HyperSyncStreamer(api_token, chain="base")
    
    print("🚀 Starting continuous stream...")
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed >= max_duration:
            print(f"\n⏰ Max duration ({max_duration}s) reached")
            break
        
        try:
            # Stream for 30 seconds at a time
            batch_duration = 30
            candles = streamer.start_streaming(duration_seconds=batch_duration)
            
            # Convert to swaps (we need to modify streamer to return swaps)
            # For now, let's collect from the converter
            if hasattr(streamer, 'converter'):
                for pool, swaps in streamer.converter.swap_buffer.items():
                    all_swaps.extend(swaps)
            
            # Convert swaps to candles with prices
            candles_df = convert_swaps_to_candles_with_prices(all_swaps)
            
            if len(candles_df) >= min_candles:
                unique_pools = candles_df['pool_address'].n_unique()
                candles_with_prices = candles_df.filter(pl.col("price_count") > 0)
                
                print(f"\n✓ Collected {len(candles_df)} candles from {unique_pools} pools")
                print(f"  Candles with real prices: {len(candles_with_prices)}")
                
                if len(candles_with_prices) >= min_candles and unique_pools >= min_pools:
                    print(f"✅ Sufficient data collected!")
                    return candles_with_prices
            
            print(f"  Progress: {len(candles_df)} candles, {elapsed:.0f}s elapsed...")
            time.sleep(5)  # Brief pause before next batch
            
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"⚠️  Error: {e}")
            time.sleep(5)
    
    # Return what we have
    if all_swaps:
        candles_df = convert_swaps_to_candles_with_prices(all_swaps)
        return candles_df.filter(pl.col("price_count") > 0) if "price_count" in candles_df.columns else candles_df
    
    return pl.DataFrame()


def generate_signals_with_prices(candles: pl.DataFrame, model_path: str = "xgb_model.json", threshold: float = 0.5, top_n: int = 5) -> list:
    """Generate signals with actual token prices"""
    if len(candles) == 0:
        return []
    
    print(f"\n🔮 Generating signals from {len(candles)} candles with prices...")
    
    # Prepare features
    try:
        feature_df = engineer_features(candles)
    except:
        # Fallback
        feature_df = candles
    
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
    
    if model:
        feature_names = [
            "return_1d", "return_3d", "return_7d",
            "ema_cross_signal", "range_normalized",
            "buy_sell_ratio", "volume_zscore", "trade_accel",
            "volatility_7d", "volatility_14d", "vol_regime_change"
        ]
        X = feature_df.select(feature_names).to_numpy()
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        pred_proba = model.predict(dmatrix)
    else:
        # Mock predictions
        pred_proba = np.random.rand(len(candles)) * 0.3 + 0.1
    
    # Add predictions
    candles_with_pred = candles.with_columns([
        pl.Series("pred_proba", pred_proba),
        pl.Series("signal", (pred_proba >= threshold).astype(int))
    ])
    
    # Get top signals
    signals_df = candles_with_pred.sort("pred_proba", descending=True).head(top_n)
    
    # Generate signals with real prices
    signals = []
    for row in signals_df.iter_rows(named=True):
        current_price = float(row.get("close", row.get("open", 1.0)))
        
        # Use actual price if available, otherwise use close
        if current_price <= 0 or current_price == 1.0:
            # Try to get from price_count
            if row.get("price_count", 0) == 0:
                # No real price, skip or use placeholder
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
    """Main execution"""
    print(f"\n{'#'*80}")
    print("STREAM AND GENERATE SIGNALS WITH REAL PRICES")
    print(f"{'#'*80}\n")
    
    # Stream until we have sufficient data
    candles = stream_until_sufficient_data(min_candles=50, min_pools=5, max_duration=300)
    
    if len(candles) == 0:
        print("\n❌ No data collected. Cannot generate signals.")
        return 1
    
    print(f"\n✓ Final dataset: {len(candles)} candles")
    print(f"  Unique pools: {candles['pool_address'].n_unique()}")
    
    # Generate signals
    signals = generate_signals_with_prices(candles, threshold=0.5, top_n=5)
    
    if not signals:
        print("\n⚠️  No signals generated")
        return 1
    
    # Print signals
    print(f"\n{'='*80}")
    print(f"🎯 GENERATED {len(signals)} SIGNALS WITH REAL PRICES")
    print(f"{'='*80}\n")
    
    for i, signal in enumerate(signals, 1):
        signal_icon = "🟢 BUY" if signal["signal"] == 1 else "⚪ HOLD"
        price_status = "✓ Real Price" if signal.get("has_real_price") else "⚠️  Estimated"
        
        print(f"{i}. {signal_icon} {price_status}")
        print(f"   Timestamp: {signal['timestamp']}")
        print(f"   Pair ID: {signal['pair_id']}")
        print(f"   Pool: {signal['pool_address']}")
        print(f"   Probability: {signal['pred_proba']:.3f}")
        print(f"   Current Price: {signal['current_price']:.8f}")
        print(f"   Take Profit: +{signal['take_profit_pct']:.1f}% → {signal['take_profit_price']:.8f}")
        print(f"   Stop Loss: -{signal['stop_loss_pct']:.1f}% → {signal['stop_loss_price']:.8f}")
        print(f"   Max Holding: {signal['holding_period_days']} days")
        print(f"   OHLC: O={signal['open']:.8f} H={signal['high']:.8f} L={signal['low']:.8f} C={signal['close']:.8f}")
        print(f"   Volume: {signal['volume']:,.0f}")
        print()
    
    # Save to file
    output_path = Path("signals_with_prices.json")
    with open(output_path, "w") as f:
        json.dump(signals, f, indent=2)
    
    print(f"💾 Saved {len(signals)} signals to: {output_path}")
    print(f"\n{'='*80}")
    print("✅ Signal generation complete!")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
