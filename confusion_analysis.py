#confusion_analysis.py


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay
)
from typing import Dict, Tuple, List
import os



OUTPUT_DIR = 'outputs'    


COLOR_TP = '#2ecc71'      
COLOR_TN = '#3498db'     
COLOR_FP = '#f39c12'      
COLOR_FN = '#e74c3c'      



def prove_accuracy_trap(
    y_test: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    verbose: bool = True
) -> Dict:


  
    y_dummy = np.zeros(len(y_test), dtype=int)

  
    y_pred = (y_prob >= threshold).astype(int)

   
    cm_dummy = confusion_matrix(y_test, y_dummy)
    cm_real  = confusion_matrix(y_test, y_pred)

  
    tn_d, fp_d, fn_d, tp_d = cm_dummy.ravel()
    tn_r, fp_r, fn_r, tp_r = cm_real.ravel()

    acc_dummy = accuracy_score(y_test, y_dummy)
    acc_real  = accuracy_score(y_test, y_pred)

    total_fraud = y_test.sum()
    total_legit = len(y_test) - total_fraud

    if verbose:
        print("=" * 60)
        print("  THE ACCURACY TRAP — PROOF")
        print("=" * 60)
        print(f"\n  Dataset: {len(y_test):,} test transactions")
        print(f"  Actual fraud : {total_fraud:,} ({total_fraud/len(y_test)*100:.1f}%)")
        print(f"  Actual legit : {total_legit:,} ({total_legit/len(y_test)*100:.1f}%)")

        print(f"\n  ── DUMMY CLASSIFIER (no model, always says legit) ──")
        print(f"  Accuracy      : {acc_dummy*100:.1f}%  ← looks impressive")
        print(f"  Fraud caught  : {tp_d} out of {total_fraud:,}   ← ZERO. Complete failure.")
        print(f"  Fraud missed  : {fn_d:,}              ← every single one slipped through")

        print(f"\n  ── REAL MODEL (threshold = {threshold}) ─────────────")
        print(f"  Accuracy      : {acc_real*100:.1f}%   ← lower than dummy!")
        print(f"  Fraud caught  : {tp_r:,} out of {total_fraud:,} ({tp_r/total_fraud*100:.1f}%)")
        print(f"  Fraud missed  : {fn_r:,}              ← still some, but far fewer")
        print(f"  False alarms  : {fp_r:,}              ← real customers wrongly blocked")

        print(f"\n  CONCLUSION:")
        print(f"  The dummy model has HIGHER accuracy ({acc_dummy*100:.1f}% vs {acc_real*100:.1f}%)")
        print(f"  but catches ZERO fraud. Accuracy is lying.")
        print(f"  The real model has LOWER accuracy but is actually useful.")
        print("=" * 60)

    return {
        'dummy': {'accuracy': acc_dummy, 'TP': tp_d, 'FN': fn_d, 'cm': cm_dummy},
        'real':  {'accuracy': acc_real,  'TP': tp_r, 'FN': fn_r, 'FP': fp_r, 'TN': tn_r, 'cm': cm_real}
    }



def plot_confusion_matrix(
    y_test: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    save_path: str = None,
    title_suffix: str = ''
) -> plt.Figure:

   
    y_pred = (y_prob >= threshold).astype(int)

   
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total = len(y_test)
    fraud_total = y_test.sum()
    legit_total = total - fraud_total

  
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Confusion Matrix Analysis — Threshold = {threshold:.2f}{title_suffix}',
        fontsize=13, fontweight='bold', y=1.01
    )

    
    ax_heat = axes[0]

   
    matrix_display = np.array([[tn, fp], [fn, tp]])

   
    cell_colors = [
        [COLOR_TN, COLOR_FP],   
        [COLOR_FN, COLOR_TP]    
    ]

   
    sns.heatmap(
        matrix_display,
        ax=ax_heat,
        annot=False,        
        cmap='RdYlGn',      
        cbar=False,         
        linewidths=2,
        linecolor='white',
        vmin=0,
        vmax=max(tn, tp) * 1.2  
    )

    
    cell_data = [
        (0, 0, tn, 'TRUE NEGATIVE',  'Legit, correctly\npassed through', COLOR_TN),
        (0, 1, fp, 'FALSE POSITIVE', 'Legit, wrongly\nblocked (annoys customer)', COLOR_FP),
        (1, 0, fn, 'FALSE NEGATIVE', 'Fraud, wrongly\npassed through (DANGER)', COLOR_FN),
        (1, 1, tp, 'TRUE POSITIVE',  'Fraud, correctly\ncaught (success)', COLOR_TP),
    ]

    for row, col, count, label, desc, color in cell_data:
        pct = count / total * 100
        ax_heat.text(
            col + 0.5, row + 0.25, label,
            ha='center', va='center',
            fontsize=8, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7)
        )
        ax_heat.text(col + 0.5, row + 0.52, f'{count:,}',
                     ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        ax_heat.text(col + 0.5, row + 0.72, f'({pct:.1f}%)',
                     ha='center', va='center', fontsize=9, color='white')
        ax_heat.text(col + 0.5, row + 0.88, desc,
                     ha='center', va='center', fontsize=7, color='white', style='italic')

    ax_heat.set_xticklabels(['Predicted: LEGIT', 'Predicted: FRAUD'], fontsize=9)
    ax_heat.set_yticklabels(['Actual: LEGIT', 'Actual: FRAUD'], fontsize=9, rotation=0)
    ax_heat.set_title('Confusion Matrix', fontsize=11)

    
    ax_bar = axes[1]

    
    fraud_recall = tp / fraud_total if fraud_total > 0 else 0
    fraud_missed = fn / fraud_total if fraud_total > 0 else 0

    
    legit_pass  = tn / legit_total if legit_total > 0 else 0
    legit_block = fp / legit_total if legit_total > 0 else 0

    categories = ['Fraud\n(actual)', 'Legit\n(actual)']
    correct    = [fraud_recall * 100, legit_pass  * 100]
    wrong      = [fraud_missed * 100, legit_block * 100]

    x = np.arange(len(categories))
    width = 0.35

    bars_correct = ax_bar.bar(x - width/2, correct, width,
                               label='Correctly classified', color=[COLOR_TP, COLOR_TN], alpha=0.85)
    bars_wrong   = ax_bar.bar(x + width/2, wrong, width,
                               label='Misclassified', color=[COLOR_FN, COLOR_FP], alpha=0.85)

   
    for bar in list(bars_correct) + list(bars_wrong):
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(categories, fontsize=10)
    ax_bar.set_ylabel('Percentage of class (%)', fontsize=9)
    ax_bar.set_ylim(0, 115)
    ax_bar.set_title('Classification Rate by Class', fontsize=11)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis='y', alpha=0.3)

   
    metrics_text = (
        f"Fraud recall : {fraud_recall*100:.1f}%\n"
        f"False alarm  : {legit_block*100:.1f}%\n"
        f"Accuracy     : {(tp+tn)/total*100:.1f}%"
    )
    ax_bar.text(0.97, 0.95, metrics_text,
                transform=ax_bar.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved → {save_path}")

    return fig


#
def plot_threshold_sweep(
    y_test: pd.Series,
    y_prob: np.ndarray,
    n_thresholds: int = 100,
    save_path: str = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:


    
    thresholds = np.linspace(0.01, 0.99, n_thresholds)

    total_fraud = y_test.sum()
    total_legit = len(y_test) - total_fraud

    
    tp_rates = np.zeros(n_thresholds)   
    fp_rates = np.zeros(n_thresholds)   
    fn_rates = np.zeros(n_thresholds)   
    tn_rates = np.zeros(n_thresholds)   

    
    if total_fraud == 0:
        raise ValueError("No fraud cases in y_test — cannot compute fraud rates.")

    for i, t in enumerate(thresholds):
       
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

       
        tp_rates[i] = tp / total_fraud    
        fp_rates[i] = fp / total_legit    
        fn_rates[i] = fn / total_fraud    
        tn_rates[i] = tn / total_legit   

    # ── Plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Threshold Sweep — How Every Metric Changes with Threshold',
                 fontsize=12, fontweight='bold')

   
    ax1 = axes[0]
    ax1.plot(thresholds, tp_rates * 100, color=COLOR_TP, lw=2.5, label='Fraud Recall (TP rate)')
    ax1.plot(thresholds, fn_rates * 100, color=COLOR_FN, lw=2.5, linestyle='--', label='Miss Rate (FN rate)')

    
    idx_50 = np.argmin(np.abs(thresholds - 0.5))
    ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7, label='Default threshold (0.5)')
    ax1.plot(0.5, tp_rates[idx_50] * 100, 'o', color=COLOR_TP, ms=8)
    ax1.plot(0.5, fn_rates[idx_50] * 100, 'o', color=COLOR_FN, ms=8)

    ax1.set_xlabel('Decision Threshold', fontsize=10)
    ax1.set_ylabel('Rate (%)', fontsize=10)
    ax1.set_title('Fraud Detection Rates vs Threshold', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 105)

    
    ax1.annotate('← Lower threshold\ncatches more fraud\n(but more false alarms)',
                 xy=(0.15, tp_rates[15] * 100),
                 xytext=(0.05, 40), fontsize=7, color=COLOR_TP,
                 arrowprops=dict(arrowstyle='->', color=COLOR_TP))

    
    ax2 = axes[1]
    ax2.plot(thresholds, fp_rates * 100, color=COLOR_FP, lw=2.5, label='False Alarm Rate (FP rate)')
    ax2.plot(thresholds, tn_rates * 100, color=COLOR_TN, lw=2.5, linestyle='--', label='Correct Pass Rate (TN rate)')
    ax2.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7, label='Default threshold (0.5)')
    ax2.plot(0.5, fp_rates[idx_50] * 100, 'o', color=COLOR_FP, ms=8)
    ax2.plot(0.5, tn_rates[idx_50] * 100, 'o', color=COLOR_TN, ms=8)

    ax2.set_xlabel('Decision Threshold', fontsize=10)
    ax2.set_ylabel('Rate (%)', fontsize=10)
    ax2.set_title('Legit Customer Rates vs Threshold', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 105)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Threshold sweep saved → {save_path}")

    metrics = {
        'tp_rates': tp_rates,
        'fp_rates': fp_rates,
        'fn_rates': fn_rates,
        'tn_rates': tn_rates
    }

    return thresholds, metrics



if __name__ == '__main__':
    print("confusion_analysis.py — standalone test with synthetic data\n")

    rng = np.random.default_rng(42)
    n = 5000
    y_test_fake = pd.Series((rng.random(n) < 0.035).astype(int))

    
    y_prob_fake = np.where(
        y_test_fake == 1,
        rng.beta(5, 2, n),    
        rng.beta(2, 8, n)     
    )

    
    results = prove_accuracy_trap(y_test_fake, y_prob_fake, threshold=0.5)

    fig = plot_confusion_matrix(
        y_test_fake, y_prob_fake, threshold=0.5,
        save_path='outputs/confusion_matrix.png'
    )

    thresholds, metrics = plot_threshold_sweep(
        y_test_fake, y_prob_fake,
        save_path='outputs/threshold_sweep.png'
    )

    plt.show()
    print("\nAll outputs generated.")