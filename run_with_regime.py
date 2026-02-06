#!/usr/bin/env python3
"""
Run Trading System with Market Regime Analysis
===============================================

Wrapper script that:
1. Analyzes market regime using cached candles
2. Generates market_regime.json
3. Runs live signal generator with regime awareness

This ensures the live system always has up-to-date regime information.
"""

import subprocess
import sys
import time
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_regime_analyzer() -> bool:
    """Run market regime analyzer and return success status"""
    logger.info("="*80)
    logger.info("STEP 1: Analyzing Market Regime")
    logger.info("="*80)
    
    try:
        result = subprocess.run(
            [sys.executable, "market_regime_analyzer.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            logger.info("✅ Regime analysis complete")
            
            # Show key output lines
            for line in result.stdout.split('\n'):
                if any(keyword in line for keyword in ['Regime:', 'Confidence:', 'Risk Management:', 'EMERGENCY', 'REENTRY']):
                    logger.info(f"   {line.strip()}")
            
            # Check if regime file was created
            if Path("market_regime.json").exists():
                logger.info("✅ market_regime.json created")
                return True
            else:
                logger.error("❌ market_regime.json not found after analysis")
                return False
        else:
            logger.error(f"❌ Regime analyzer failed with exit code {result.returncode}")
            logger.error(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Regime analyzer timed out (30s)")
        return False
    except Exception as e:
        logger.error(f"❌ Error running regime analyzer: {e}")
        return False


def run_live_system():
    """Run live signal generator with regime awareness"""
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Starting Live Signal Generator")
    logger.info("="*80)
    logger.info("ℹ️  Press Ctrl+C to stop\n")
    
    try:
        # Run live system (will stream output in real-time)
        subprocess.run(
            [sys.executable, "live_signal_generator.py"],
            check=True
        )
    except KeyboardInterrupt:
        logger.info("\n⚠️  Received shutdown signal")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Live system exited with error code {e.returncode}")
        sys.exit(1)


def main():
    """Main entry point"""
    logger.info("="*80)
    logger.info("REGIME-AWARE TRADING SYSTEM")
    logger.info("="*80)
    logger.info("")
    
    # Check required files
    required_files = [
        "live_candles_cache.pkl",
        "xgb_model.json",
        "xgb_calibrator.pkl",
        "market_regime_analyzer.py",
        "live_signal_generator.py"
    ]
    
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        logger.error(f"❌ Missing required files: {', '.join(missing)}")
        sys.exit(1)
    
    logger.info("✅ All required files present\n")
    
    # Run regime analyzer first
    if not run_regime_analyzer():
        logger.error("\n❌ Cannot proceed without regime analysis")
        logger.info("Tip: Make sure live_candles_cache.pkl has sufficient data")
        sys.exit(1)
    
    # Small delay to ensure file is written
    time.sleep(1)
    
    # Run live system
    run_live_system()
    
    logger.info("\n✅ System shutdown complete")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n👋 Goodbye!")
        sys.exit(0)
