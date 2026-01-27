#!/usr/bin/env python3
"""
Run Full Pipeline
=================

Executes the complete on-chain signal generation pipeline:
1. Data preparation
2. Model training
3. Backtesting
4. Validation
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Completed in {elapsed:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n❌ Failed after {elapsed:.1f}s")
        print(f"Error code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_data_exists():
    """Check if required data files exist"""
    print(f"\n{'='*80}")
    print("📋 PRE-FLIGHT CHECK")
    print(f"{'='*80}\n")
    
    pair_universe = Path("./Files/pair-universe")
    candles = Path("./Files/candles-1d.parquet")
    
    checks = [
        ("Pair universe", pair_universe.exists()),
        ("Candles data", candles.exists()),
    ]
    
    all_good = True
    for name, exists in checks:
        status = "✅" if exists else "❌"
        print(f"  {status} {name}")
        if not exists:
            all_good = False
    
    if not all_good:
        print("\n⚠️  WARNING: Some data files are missing!")
        print("   Expected paths:")
        print(f"   - {pair_universe}")
        print(f"   - {candles}")
        print("\n   Please verify dataset locations and update paths in Config if needed.")
        
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("   Exiting...")
            sys.exit(1)
    else:
        print("\n✅ All data files found!")
    
    return all_good

def main():
    """Run full pipeline"""
    print(f"\n{'#'*80}")
    print("ON-CHAIN DEX SIGNAL GENERATION - FULL PIPELINE")
    print(f"{'#'*80}")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check data
    check_data_exists()
    
    
    # Step 1: Data preparation
    step1_success = run_command(
        f"python onchain_signal_system.py",
        "STEP 1: Data Preparation (Loading, Filtering, Labeling)"
    )
    
    if not step1_success:
        print("\n❌ Pipeline failed at Step 1 (Data Preparation)")
        print("   Check error messages above for details.")
        sys.exit(1)
    
    # Check if processed data was created
    processed_data = Path("./processed_data.parquet")
    if not processed_data.exists():
        print("\n❌ Processed data file not found!")
        print(f"   Expected: {processed_data}")
        sys.exit(1)
    
    print(f"\n✅ Processed data created: {processed_data}")
    print(f"   Size: {processed_data.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Step 2: Model training and backtesting
    step2_success = run_command(
        f"python model_training.py",
        "STEP 2: Model Training, Backtesting & Validation"
    )
    
    if not step2_success:
        print("\n❌ Pipeline failed at Step 2 (Model Training)")
        print("   Check error messages above for details.")
        sys.exit(1)
    
    # Final summary
    print(f"\n{'#'*80}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'#'*80}")
    
    print("\n📁 Generated Files:")
    files = [
        ("Processed Data", "./processed_data.parquet"),
        ("Trained Model", "./xgb_model.json"),
    ]
    
    for name, path in files:
        p = Path(path)
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            print(f"  ✅ {name}: {path} ({size_mb:.2f} MB)")
        else:
            print(f"  ⚠️  {name}: Not found")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print(f"{'='*80}")
    print("""
1. Review the final conclusion in the output above
   
2. If PROFITABLE:
   • Analyze feature importance
   • Test on additional out-of-sample data
   • Consider implementing live trading integration
   
3. If NOT PROFITABLE:
   • Review suggested improvements
   • Try different chains or timeframes
   • Consider alternative data sources
   
4. Read METHODOLOGY.md for detailed explanations
   
5. Customize Config parameters and re-run if needed
""")
    
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
