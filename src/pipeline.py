# src/pipeline.py


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Tuple, Dict, Any



TEST_SIZE    = 0.20    
RANDOM_STATE = 42      


DEFAULT_THRESHOLD = 0.50



def build_pipeline(model_type: str = 'logistic') -> Pipeline:
  
    imputer = SimpleImputer(strategy='median')

    scaler = StandardScaler(copy=True)

    
    if model_type == 'logistic':
        
        classifier = LogisticRegression(
            class_weight='balanced',  
            max_iter=500,            
            random_state=RANDOM_STATE,
            solver='lbfgs',          
            n_jobs=-1                 
        )

    elif model_type == 'random_forest':
        classifier = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            max_depth=10,              
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Choose 'logistic' or 'random_forest'."
        )

    
    pipeline = Pipeline(steps=[
        ('imputer',    imputer),
        ('scaler',     scaler),
        ('classifier', classifier)
    ])

    return pipeline



def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'logistic',
    threshold: float = DEFAULT_THRESHOLD,
    verbose: bool = True
) -> Tuple[Pipeline, Dict[str, Any]]:
   

    
    if verbose:
        print(f"[1/3] Splitting data: {int((1-TEST_SIZE)*100)}% train / "
              f"{int(TEST_SIZE*100)}% test (stratified)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    if verbose:
        print(f"      → Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

    
    if verbose:
        print(f"[2/3] Training {model_type} pipeline...")
    pipeline = build_pipeline(model_type=model_type)
    pipeline.fit(X_train, y_train)   
    if verbose:
        print("      → Training complete")

    
    if verbose:
        print(f"[3/3] Generating predictions at threshold={threshold}...")

    y_prob = pipeline.predict_proba(X_test)[:, 1]   

    
    y_prob = np.clip(y_prob, 0.0, 1.0)
    y_pred = (y_prob >= threshold).astype(int)

    
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)     
    cm     = confusion_matrix(y_test, y_pred)

  
    tn, fp, fn, tp = cm.ravel()

    if verbose:
        print(f"\n── Results at threshold = {threshold} ──────────────")
        print(f"  Accuracy : {acc*100:.2f}%   ← remember: this number lies on imbalanced data")
        print(f"  AUC-ROC  : {auc:.4f}        ← this is the real signal")
        print(f"\n  Confusion Matrix:")
        print(f"  ┌──────────────┬──────────────┐")
        print(f"  │ TN: {tn:>8,} │ FP: {fp:>8,} │  ← Legit transactions")
        print(f"  ├──────────────┼──────────────┤")
        print(f"  │ FN: {fn:>8,} │ TP: {tp:>8,} │  ← Fraud transactions")
        print(f"  └──────────────┴──────────────┘")
        print(f"\n  Of {fn+tp:,} actual fraud cases:")
        print(f"    Caught  (TP): {tp:,} ({tp/(fn+tp)*100:.1f}%)")
        print(f"    Missed  (FN): {fn:,} ({fn/(fn+tp)*100:.1f}%)")
        print(f"\n  Of {tn+fp:,} actual legit cases:")
        print(f"    Correctly allowed (TN): {tn:,} ({tn/(tn+fp)*100:.1f}%)")
        print(f"    Wrongly blocked   (FP): {fp:,} ({fp/(tn+fp)*100:.1f}%)")

    
    results = {
        'y_test'    : y_test,
        'y_prob'    : y_prob,        
        'y_pred'    : y_pred,
        'threshold' : threshold,
        'accuracy'  : acc,
        'auc_roc'   : auc,
        'confusion_matrix': cm,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'X_train'   : X_train,
        'X_test'    : X_test,
        'y_train'   : y_train,
    }

    return pipeline, results



def dummy_baseline(y_test: pd.Series, verbose: bool = True) -> Dict[str, Any]:
   
    y_dummy = np.zeros(len(y_test), dtype=int)   
    acc = accuracy_score(y_test, y_dummy)
    cm  = confusion_matrix(y_test, y_dummy)
    tn, fp, fn, tp = cm.ravel()

    if verbose:
        print("\n── DUMMY CLASSIFIER (always predicts Legit) ────────────")
        print(f"  Accuracy : {acc*100:.2f}%  ← looks great, right?")
        print(f"  Fraud caught (TP): {tp}    ← ZERO. It caught nothing.")
        print(f"  Fraud missed (FN): {fn:,}  ← every fraud slipped through")
        print("  This is the Accuracy Trap.")

    return {'accuracy': acc, 'TP': tp, 'FN': fn, 'y_pred': y_dummy}



if __name__ == '__main__':
    print("pipeline.py standalone test — generating synthetic data...\n")

    
    rng = np.random.default_rng(42)
    n = 1000
    X_fake = pd.DataFrame(rng.standard_normal((n, 20)),
                          columns=[f'feat_{i}' for i in range(20)])
    y_fake = pd.Series((rng.random(n) < 0.035).astype(int))   # 3.5% fraud

    pipeline, results = train_and_evaluate(X_fake, y_fake, verbose=True)
    dummy_baseline(results['y_test'])
    print(f"\nPipeline steps: {[name for name, _ in pipeline.steps]}")
