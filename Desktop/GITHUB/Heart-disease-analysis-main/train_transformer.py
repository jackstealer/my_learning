"""
Prepare metadata for transformer model
This script analyzes the actual data and creates data-driven thresholds
"""

import sys
import os
sys.path.insert(0, 'ml_model')

from transformers_model import HeartDiseaseTransformerModel

def main():
    print("\n" + "="*70)
    print("HEART DISEASE AI - DATA-DRIVEN MODEL SETUP")
    print("Analyzing Heart_new2.csv to extract patterns and thresholds")
    print("="*70 + "\n")
    
    # Check if data file exists
    data_file = 'Heart_new2.csv'
    if not os.path.exists(data_file):
        print(f"❌ Error: {data_file} not found!")
        print("Please ensure the dataset is in the current directory.")
        return
    
    print(f"✅ Found dataset: {data_file}\n")
    
    # Initialize model
    model = HeartDiseaseTransformerModel()
    
    # Prepare data and save metadata
    print("Analyzing data and extracting statistical patterns...")
    X, y, df = model.prepare_training_data(data_file)
    
    # Save metadata
    model.save_metadata()
    
    # Show feature importance
    if 'feature_importance' in model.feature_stats:
        print("\n" + "="*70)
        print("TOP 10 MOST IMPORTANT FEATURES (by correlation)")
        print("="*70)
        importance = model.feature_stats['feature_importance']
        for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
            print(f"{i:2d}. {feature:<25} {score:.4f}")
    
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\nMetadata has been extracted from the actual dataset.")
    print("All thresholds and patterns are data-driven (no hardcoding).")
    print("\nTo start the application:")
    print("  python app.py")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
