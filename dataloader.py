# src/data_loader.py


import os                          
import pandas as pd
import numpy as np
from typing import Tuple           



MISSING_THRESHOLD = 0.30   
RANDOM_STATE      = 42     


def load_data(
    transaction_path: str,
    identity_path: str,
    sample_frac: float = 1.0,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
   

    
    _check_file_exists(transaction_path)
    _check_file_exists(identity_path)

    if not (0 < sample_frac <= 1.0):
        raise ValueError(
            f"sample_frac must be between 0 and 1. Got: {sample_frac}\n"
            f"Example: use 0.1 to load 10% of data for fast testing."
        )

    
    if verbose:
        print("[1/5] Loading transaction data...")
    df_trans = pd.read_csv(transaction_path, low_memory=False)

    if verbose:
        print(f"      → {df_trans.shape[0]:,} rows, {df_trans.shape[1]} columns")
        print("[2/5] Loading identity data...")
    df_id = pd.read_csv(identity_path, low_memory=False)

    if verbose:
        print(f"      → {df_id.shape[0]:,} rows, {df_id.shape[1]} columns")


# Merging (LEFT JOIN)



    if verbose:
        print("[3/5] Merging transaction + identity tables...")
    df = pd.merge(df_trans, df_id, on='TransactionID', how='left')
    if verbose:
        print(f"      → Merged shape: {df.shape}")

  
    if sample_frac < 1.0:
        if verbose:
            print(f"[4/5] Sampling {sample_frac*100:.0f}% of data (stratified)...")
        df = df.groupby('isFraud', group_keys=False).apply(
            lambda x: x.sample(frac=sample_frac, random_state=RANDOM_STATE)
        ).reset_index(drop=True)
        if verbose:
            print(f"      → Sampled shape: {df.shape}")
    else:
        if verbose:
            print("[4/5] Using full dataset (no sampling)")

    
    if verbose:
        print("[5/5] Selecting features...")
    X, y = _select_features(df, verbose=verbose)

    if verbose:
        fraud_n    = y.sum()
        fraud_pct  = fraud_n / len(y) * 100
        print(f"\n✓ Data ready: {len(X):,} rows × {X.shape[1]} features")
        print(f"  Fraud: {fraud_n:,} ({fraud_pct:.1f}%)  |  "
              f"Legit: {len(y)-fraud_n:,} ({100-fraud_pct:.1f}%)")

    return X, y



def _select_features(
    df: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    

    
    if 'isFraud' not in df.columns:
        raise KeyError(
            "Column 'isFraud' not found. "
            "Are you sure you loaded train_transaction.csv and not test?"
        )
    y = df['isFraud'].astype(int)

    
    id_cols_to_drop = ['TransactionID', 'isFraud']

   
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

   
    numeric_cols = [c for c in numeric_cols if c not in id_cols_to_drop]

   
    missing_rates = df[numeric_cols].isnull().mean()
    usable_cols   = missing_rates[missing_rates <= MISSING_THRESHOLD].index.tolist()

    dropped_cols = len(numeric_cols) - len(usable_cols)
    if verbose:
        print(f"      → Numeric candidates : {len(numeric_cols)}")
        print(f"      → Dropped (>{MISSING_THRESHOLD*100:.0f}% NaN): {dropped_cols}")
        print(f"      → Final feature count: {len(usable_cols)}")

   
    if len(usable_cols) == 0:
        raise RuntimeError(
            f"No usable features found after applying missing threshold "
            f"of {MISSING_THRESHOLD}. Try raising MISSING_THRESHOLD."
        )

    X = df[usable_cols].copy()
    return X, y



def _check_file_exists(path: str) -> None:
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n✗ File not found: {path}\n"
            f"  Download from: https://www.kaggle.com/c/ieee-fraud-detection/data\n"
            f"  Place in the /data/ folder of this project."
        )



if __name__ == '__main__':
    print("Running data_loader.py in standalone test mode...\n")

    
    X, y = load_data(
        transaction_path='data/train_transaction.csv',
        identity_path='data/train_identity.csv',
        sample_frac=0.1,   
        verbose=True
    )

    print(f"\nX shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print(f"X dtypes sample:\n{X.dtypes.head(10)}")
    print(f"\nFirst 3 rows of X:\n{X.head(3)}")
