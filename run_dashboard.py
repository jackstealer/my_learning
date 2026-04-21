"""
Helper script to run the Streamlit dashboard
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install them using:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        '12_air_quality.csv',
        'app.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def check_model():
    """Check if model is trained"""
    if not os.path.exists('aqi_prediction_model.pkl'):
        print("⚠️  Warning: Trained model not found!")
        print("   The prediction feature will not work until you train the model.")
        print("   Run: python air_quality_prediction.py")
        print("\n   You can still explore the dashboard's other features.")
        return False
    return True

def main():
    print("="*60)
    print("🌍 Air Quality Index Prediction Dashboard")
    print("="*60)
    print("\n🔍 Checking dependencies...")
    
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ All dependencies installed\n")
    
    print("🔍 Checking required files...")
    
    if not check_files():
        sys.exit(1)
    
    print("✅ All required files present\n")
    
    print("🔍 Checking trained model...")
    model_exists = check_model()
    
    if model_exists:
        print("✅ Model found\n")
    else:
        print()
    
    print("="*60)
    print("🚀 Starting Streamlit dashboard...")
    print("="*60)
    print("\n📌 The dashboard will open in your default browser")
    print("📌 Press Ctrl+C to stop the server\n")
    
    # Run streamlit
    try:
        subprocess.run(['streamlit', 'run', 'app.py'])
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped. Goodbye!")
    except FileNotFoundError:
        print("\n❌ Error: Streamlit not found!")
        print("   Install it using: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
