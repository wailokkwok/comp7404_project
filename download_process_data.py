import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml, fetch_california_housing
import requests
import os

# Create a data directory
DATA_DIR = './datasets'
os.makedirs(DATA_DIR, exist_ok=True)

print(f"All datasets will be saved to: {os.path.abspath(DATA_DIR)}\n")

# ============================================================================
# 1. ADULT-INCOME DATASET
# ============================================================================
def download_adult_income():
    """
    Download Adult Income (Census) dataset
    """
    print("Downloading Adult-Income dataset...")
    
    # Column names
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race', 'sex',
               'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    
    # Download training data
    train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    
    df_train = pd.read_csv(train_url, names=columns, skipinitialspace=True)
    df_test = pd.read_csv(test_url, names=columns, skipinitialspace=True, skiprows=1)
    
    # Save
    df_train.to_csv(f'{DATA_DIR}/adult_train.csv', index=False)
    df_test.to_csv(f'{DATA_DIR}/adult_test.csv', index=False)
    
    print(f"✓ Saved to {DATA_DIR}/adult_train.csv and adult_test.csv")
    print(f"  Train: {len(df_train)} samples, Test: {len(df_test)} samples\n")

# ============================================================================
# 2. TELCO-CHURN DATASET
# ============================================================================
def download_telco_churn():
    """
    Download Telco Customer Churn dataset
    """
    print("Downloading Telco-Churn dataset...")
    
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    df = pd.read_csv(url)
    df.to_csv(f'{DATA_DIR}/telco_churn.csv', index=False)
    
    print(f"✓ Saved to {DATA_DIR}/telco_churn.csv")
    print(f"  {len(df)} samples, {len(df.columns)} features\n")

# ============================================================================
# 3. CREDIT-FRAUD DATASET
# ============================================================================
def download_credit_fraud():
    """
    Credit Card Fraud Detection dataset (Kaggle)
    """
    print("Downloading Credit-Fraud dataset...")
    print("⚠ This dataset requires manual download from Kaggle:")
    print("  URL: https://www.kaggle.com/mlg-ulb/creditcardfraud")
    print("  Download 'creditcard.csv' and place it in the datasets folder")
    print(f"  Target location: {DATA_DIR}/creditcard.csv\n")
    
    # Check if already downloaded
    if os.path.exists(f'{DATA_DIR}/creditcard.csv'):
        df = pd.read_csv(f'{DATA_DIR}/creditcard.csv')
        print(f"✓ Found existing file: {len(df)} samples, {len(df.columns)} features\n")
    else:
        print("✗ File not found. Please download manually.\n")

# ============================================================================
# 4. HEALTHCARE DATASET
# ============================================================================
def download_healthcare():
    """
    Healthcare dataset is PRIVATE in the paper.
    Using public alternatives:
    - Option 1: Diabetes dataset (scikit-learn)
    - Option 2: Heart Disease (UCI)
    - Option 3: MIMIC-III (requires access)
    """
    print("Downloading Healthcare substitute...")
    print("⚠ Note: Original healthcare dataset in paper is private/proprietary")
    print("  Using UCI Heart Disease dataset as substitute\n")
    
    # Download Heart Disease dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    df = pd.read_csv(url, names=columns, na_values='?')
    df.to_csv(f'{DATA_DIR}/heart_disease.csv', index=False)
    
    print(f"✓ Saved to {DATA_DIR}/heart_disease.csv")
    print(f"  {len(df)} samples, {len(df.columns)} features\n")

# ============================================================================
# 5. CAL-HOUSING DATASET
# ============================================================================
def download_cal_housing():
    """
    California Housing dataset
    """
    print("Downloading Cal-Housing dataset...")
    
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    df.to_csv(f'{DATA_DIR}/cal_housing.csv', index=False)
    
    print(f"✓ Saved to {DATA_DIR}/cal_housing.csv")
    print(f"  {len(df)} samples, {len(df.columns)} features\n")

# ============================================================================
# 6. ELEVATORS DATASET
# ============================================================================
def download_elevators():
    """
    Elevators dataset from OpenML
    """
    print("Downloading Elevators dataset...")
    
    try:
        data = fetch_openml('elevators', version=1, parser='auto', as_frame=True)
        df = data.frame
        
        df.to_csv(f'{DATA_DIR}/elevators.csv', index=False)
        
        print(f"✓ Saved to {DATA_DIR}/elevators.csv")
        print(f"  {len(df)} samples, {len(df.columns)} features\n")
    except Exception as e:
        print(f"✗ Error downloading from OpenML: {e}")
        print("  Alternative: Download manually from OpenML.org\n")

# ============================================================================
# 7. POL DATASET
# ============================================================================
def download_pol():
    """
    Pol (PoleTele) dataset from OpenML
    """
    print("Downloading Pol dataset...")
    
    try:
        data = fetch_openml('pol', version=1, parser='auto', as_frame=True)
        df = data.frame
        
        df.to_csv(f'{DATA_DIR}/pol.csv', index=False)
        
        print(f"✓ Saved to {DATA_DIR}/pol.csv")
        print(f"  {len(df)} samples, {len(df.columns)} features\n")
    except Exception as e:
        print(f"✗ Error downloading from OpenML: {e}")
        print("  Alternative: Download manually from OpenML.org\n")

# ============================================================================
# 8. WINE-QUALITY DATASET
# ============================================================================
def download_wine_quality():
    """
    Wine Quality dataset (Red + White)
    """
    print("Downloading Wine-Quality dataset...")
    
    url_red = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    url_white = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    
    df_red = pd.read_csv(url_red, sep=';')
    df_red['type'] = 'red'
    
    df_white = pd.read_csv(url_white, sep=';')
    df_white['type'] = 'white'
    
    # Save separately
    df_red.to_csv(f'{DATA_DIR}/wine_quality_red.csv', index=False)
    df_white.to_csv(f'{DATA_DIR}/wine_quality_white.csv', index=False)
    
    # Save combined
    df_combined = pd.concat([df_red, df_white], ignore_index=True)
    df_combined.to_csv(f'{DATA_DIR}/wine_quality_combined.csv', index=False)
    
    print(f"✓ Saved to {DATA_DIR}/wine_quality_*.csv")
    print(f"  Red: {len(df_red)} samples, White: {len(df_white)} samples")
    print(f"  Combined: {len(df_combined)} samples, {len(df_combined.columns)} features\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("DOWNLOADING DATASETS FROM DP-EBM PAPER")
    print("="*70)
    print()
    
    download_functions = [
        download_adult_income,
        download_telco_churn,
        download_credit_fraud,
        download_healthcare,
        download_cal_housing,
        download_elevators,
        download_pol,
        download_wine_quality
    ]
    
    for download_func in download_functions:
        try:
            download_func()
        except Exception as e:
            print(f"✗ Error in {download_func.__name__}: {e}\n")
    
    print("="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"\nAll datasets saved to: {os.path.abspath(DATA_DIR)}")
    print("\nDataset files:")
    for filename in sorted(os.listdir(DATA_DIR)):
        filepath = os.path.join(DATA_DIR, filename)
        size_mb = os.path.getsize(filepath) / (1024*1024)
        print(f"  - {filename} ({size_mb:.2f} MB)")