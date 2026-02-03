#!/usr/bin/env python3
"""Quick test to verify regime classifier training and signal generation"""

import sys
import os
from pathlib import Path

# Add market-regime-classifier to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'market-regime-classifier'))

from market_regime_classifier import MarketRegimeClassifier
from live_signal_generator import LiveConfig
import polars as pl

def test_regime_classifier():
    """Test that regime classifier can be trained and used"""
    print("="*80)
    print("TESTING REGIME CLASSIFIER INTEGRATION")
    print("="*80)
    
    # Check training data exists
    training_data_path = Path(LiveConfig.REGIME_TRAINING_DATA_PATH)
    print(f"\n📊 Training data path: {training_data_path}")
    print(f"   Exists: {training_data_path.exists()}")
    
    if training_data_path.exists():
        df = pl.read_parquet(str(training_data_path))
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {df.columns[:5]}...")
    else:
        print("❌ Training data not found!")
        return False
    
    # Check model path
    model_path = Path(LiveConfig.REGIME_MODEL_PATH)
    print(f"\n📁 Model path: {model_path}")
    print(f"   Currently exists: {model_path.exists()}")
    
    # Test training the classifier
    print("\n🔄 Training regime classifier...")
    try:
        classifier = MarketRegimeClassifier()
        classifier.train(
            data_path=str(training_data_path),
            test_size=0.2,
            sample_pairs=LiveConfig.REGIME_TRAIN_SAMPLE_PAIRS
        )
        
        # Save model
        classifier.save_model(str(model_path))
        print(f"✅ Model trained and saved to {model_path}")
        
        # Check if files were created
        metadata_path = model_path.parent / "market_regime_model_metadata.pkl"
        print(f"   Model JSON exists: {model_path.exists()}")
        print(f"   Metadata exists: {metadata_path.exists()}")
        
        # Test loading
        print("\n🔄 Testing model loading...")
        classifier2 = MarketRegimeClassifier(model_path=str(model_path))
        print("✅ Model loaded successfully!")
        
        return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_regime_classifier()
    sys.exit(0 if success else 1)
