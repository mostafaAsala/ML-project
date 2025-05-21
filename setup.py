import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Basic requirements installed successfully!")
    except subprocess.CalledProcessError:
        print("Error installing basic requirements.")
        return False
    
    # Ask if user wants to install optional packages
    install_optional = input("Do you want to install optional packages for advanced models? (y/n): ").lower()
    if install_optional == 'y':
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "lightgbm"])
            print("XGBoost and LightGBM installed successfully!")
        except subprocess.CalledProcessError:
            print("Error installing XGBoost and LightGBM.")
        
        install_tf = input("Do you want to install TensorFlow for deep learning models? (y/n): ").lower()
        if install_tf == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
                print("TensorFlow installed successfully!")
            except subprocess.CalledProcessError:
                print("Error installing TensorFlow.")
    
    # Install spaCy English model
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("spaCy English model installed successfully!")
    except subprocess.CalledProcessError:
        print("Error installing spaCy English model.")
    
    print("\nSetup complete!")
    return True

if __name__ == "__main__":
    install_requirements()
