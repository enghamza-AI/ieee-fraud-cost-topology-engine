
# stakeholder_report.py


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple
import os
import csv

from cost_surface import find_optimal_threshold    




DEFAULT_STAKEHOLDERS = {
    'Bank Risk Manager': {
        'fp_cost'   : 50,     
        'fn_cost'   : 1500,    
        'priority'  : 'Minimise fraud losses — false alarms are acceptable',
        'min_recall': 0.75,   
        'color'     : '#e74c3c'
    },
    'Merchant (E-commerce)': {
        'fp_cost'   : 250,    
        'fn_cost'   : 350,    
        'priority'  : 'Protect customer experience — revenue loss from blocks hurts equally',
        'min_recall': None,   
        'color'     : '#3498db'
    },
    'Financial Regulator': {
        'fp_cost'   : 10,     
        'fn_cost'   : 8000,    
        'priority'  : 'Maximum fraud recall — false alarms are irrelevant',
        'min_recall': 0.85,   
        'color'     : '#f39c12'
    }
}



def build_stakeholder_report(
    y_test: pd.Series,
    y_prob: np.ndarray,
    stakeholders: Dict = None,
    verbose: bool = True
) -> pd.DataFrame:
  

    if stakeholders is None:
        stakeholders = DEFAULT_STAKEHOLDERS

   
    if len(y_test) != len(y_prob):
        raise ValueError("y_test and y_prob must have the same length.")

    total_fraud = int(y_test.sum())
    total_legit = int(len(y_test) - total_fraud)

    if total_fraud == 0:
        raise ValueError("No fraud cases in y_test. Report is meaningless.")

    rows = []   

    for name, config in stakeholders.items():
        fp_cost    = config['fp_cost']
        fn_cost    = config['fn_cost']
        min_recall = config.get('min_recall', None)

        
        opt_thresh, min_total_cost = find_optimal_threshold(
            y_test, y_prob, fp_cost, fn_cost, n_thresholds=200
        )

        
        if min_recall is not None:
            opt_thresh = _enforce_recall_constraint(
                y_test, y_prob, opt_thresh, min_recall
            )
           
            y_pred_c = (y_prob >= opt_thresh).astype(int)
            fp_c = int(np.sum((y_pred_c == 1) & (np.array(y_test) == 0)))
            fn_c = int(np.sum((y_pred_c == 0) & (np.array(y_test) == 1)))
            min_total_cost = fp_c * fp_cost + fn_c * fn_cost

        
        y_pred = (y_prob >= opt_thresh).astype(int)
        y_arr  = np.array(y_test)

        tp = int(np.sum((y_pred == 1) & (y_arr == 1)))   
        fp = int(np.sum((y_pred == 1) & (y_arr == 0)))   
        fn = int(np.sum((y_pred == 0) & (y_arr == 1)))  
        tn = int(np.sum((y_pred == 0) & (y_arr == 0)))  

        
        recall    = tp / total_fraud if total_fraud > 0 else 0   
        fpr       = fp / total_legit if total_legit > 0 else 0   
        precision = tp / (tp + fp)  if (tp + fp) > 0  else 0    
        accuracy  = (tp + tn) / len(y_test)

        
        y_pred_default = (y_prob >= 0.5).astype(int)
        fp_def = int(np.sum((y_pred_default == 1) & (y_arr == 0)))
        fn_def = int(np.sum((y_pred_default == 0) & (y_arr == 1)))
        cost_at_default = fp_def * fp_cost + fn_def * fn_cost

        cost_saving = cost_at_default - min_total_cost   

       
        if min_recall is not None:
            constraint_met = recall >= min_recall
            constraint_str = f"{'✓' if constraint_met else '✗'} {recall*100:.1f}% ≥ {min_recall*100:.0f}% required"
        else:
            constraint_str = "None"

        rows.append({
            'Stakeholder'         : name,
            'Priority'            : config['priority'],
            'FP Cost ($)'         : fp_cost,
            'FN Cost ($)'         : fn_cost,
            'Optimal Threshold'   : round(opt_thresh, 3),
            'Fraud Caught (TP)'   : tp,
            'Fraud Missed (FN)'   : fn,
            'False Alarms (FP)'   : fp,
            'Correct Pass (TN)'   : tn,
            'Recall (%)'          : round(recall * 100, 1),
            'False Alarm Rate (%)': round(fpr * 100, 1),
            'Precision (%)'       : round(precision * 100, 1),
            'Accuracy (%)'        : round(accuracy * 100, 1),
            'Total Cost ($)'      : round(min_total_cost, 0),
            'Cost at Threshold=0.5($)': round(cost_at_default, 0),
            'Cost Saving ($)'     : round(cost_saving, 0),
            'Recall Constraint'   : constraint_str,
            '_color'              : config['color'],    
        })

    df = pd.DataFrame(rows)

    if verbose:
        _print_report(df, total_fraud, total_legit)

    return df



def _print_report(df: pd.DataFrame, total_fraud: int, total_legit: int) -> None:
  

    print("\n" + "═" * 72)
    print("  STAKEHOLDER THRESHOLD REPORT")
    print(f"  Test set: {total_fraud + total_legit:,} transactions  |  "
          f"Fraud: {total_fraud:,} ({total_fraud/(total_fraud+total_legit)*100:.1f}%)  |  "
          f"Legit: {total_legit:,}")
    print("═" * 72)

    for _, row in df.iterrows():
        saving_str = (f"saves ${row['Cost Saving ($)']:,.0f}"
                      if row['Cost Saving ($)'] > 0
                      else f"costs ${abs(row['Cost Saving ($)']):,.0f} more")

        print(f"\n  {'─' * 68}")
        print(f"  STAKEHOLDER : {row['Stakeholder']}")
        print(f"  Priority    : {row['Priority']}")
        print(f"  {'─' * 68}")
        print(f"  Cost structure  : FP = ${row['FP Cost ($)']:,}  |  FN = ${row['FN Cost ($)']:,}")
        print(f"  Optimal threshold: {row['Optimal Threshold']:.3f}   (vs default 0.500)")
        print(f"  {'─' * 30}")
        print(f"  Fraud caught  (TP): {row['Fraud Caught (TP)']:>8,}   Recall: {row['Recall (%)']:.1f}%")
        print(f"  Fraud missed  (FN): {row['Fraud Missed (FN)']:>8,}")
        print(f"  False alarms  (FP): {row['False Alarms (FP)']:>8,}   False alarm rate: {row['False Alarm Rate (%)']:.1f}%")
        print(f"  Correct pass  (TN): {row['Correct Pass (TN)']:>8,}")
        print(f"  {'─' * 30}")
        print(f"  Total cost at optimal : ${row['Total Cost ($)']:>12,.0f}")
        print(f"  Total cost at 0.5     : ${row['Cost at Threshold=0.5($)']:>12,.0f}")
        print(f"  Optimisation          : {saving_str}")
        print(f"  Recall constraint     : {row['Recall Constraint']}")

    print(f"\n{'═' * 72}")
    print(f"  KEY TAKEAWAY:")
    thresholds = df['Optimal Threshold'].values
    names      = df['Stakeholder'].values
    print(f"  Same model. Same probabilities. Three different thresholds:")
    for name, t in zip(names, thresholds):
        print(f"    {name}: {t:.3f}")
    print(f"  The threshold is a business decision, not a model parameter.")
    print("═" * 72 + "\n")



def _enforce_recall_constraint(
    y_test: pd.Series,
    y_prob: np.ndarray,
    starting_threshold: float,
    min_recall: float
) -> float:
 
    candidate_thresholds = np.linspace(starting_threshold, 0.01, 200)

    for t in candidate_thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (np.array(y_test) == 1))
        recall = tp / y_test.sum()

        if recall >= min_recall:
            return float(t)

   
    import warnings
    warnings.warn(
        f"Could not achieve min_recall={min_recall:.0%} even at threshold=0.01. "
        f"Returning 0.01. Consider retraining with class_weight='balanced'.",
        RuntimeWarning
    )
    return 0.01



def save_report_csv(df: pd.DataFrame, save_path: str = 'outputs/stakeholder_report.csv') -> None:
  
    export_cols = [c for c in df.columns if not c.startswith('_')]
    export_df   = df[export_cols]

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    export_df.to_csv(save_path, index=False)
    print(f"Report CSV saved → {save_path}")


def plot_stakeholder_comparison(
    df: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
   

    colors    = df['_color'].tolist()
    names     = [n.split(' ')[0] for n in df['Stakeholder'].tolist()]   

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('Stakeholder Comparison — Same Model, Different Operating Points',
                 fontsize=12, fontweight='bold')

    metrics = [
        (axes[0, 0], 'Optimal Threshold',    'Optimal Decision Threshold',    'Threshold value', None),
        (axes[0, 1], 'Recall (%)',            'Fraud Recall (%)',              '% of fraud caught', '%'),
        (axes[1, 0], 'False Alarm Rate (%)',  'False Alarm Rate (%)',          '% of legit blocked', '%'),
        (axes[1, 1], 'Total Cost ($)',        'Minimum Total Cost ($)',        'Business loss ($)', '$'),
    ]

    for ax, col, title, ylabel, fmt in metrics:
        values = df[col].tolist()
        bars   = ax.bar(names, values, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)

       
        for bar, val in zip(bars, values):
            label = (f'${val:,.0f}' if fmt == '$'
                     else f'{val:.1f}%' if fmt == '%'
                     else f'{val:.3f}')
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(values) * 0.01,
                    label, ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(title, fontsize=10, pad=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(axis='y', alpha=0.3)

       
        if col == 'Optimal Threshold':
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            ax.text(2.4, 0.51, 'default 0.5', fontsize=7, color='gray', va='bottom')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison chart saved → {save_path}")

    return fig



if __name__ == '__main__':
    print("stakeholder_report.py — standalone test with synthetic data\n")

    rng = np.random.default_rng(42)
    n = 5000

    y_test_fake = pd.Series((rng.random(n) < 0.035).astype(int))
    y_prob_fake = np.where(
        y_test_fake == 1,
        rng.beta(5, 2, n),
        rng.beta(2, 8, n)
    )

    
    report_df = build_stakeholder_report(y_test_fake, y_prob_fake, verbose=True)

    
    save_report_csv(report_df, save_path='outputs/stakeholder_report.csv')

    
    fig = plot_stakeholder_comparison(
        report_df,
        save_path='outputs/stakeholder_comparison.png'
    )

    plt.show()
    print("Done.")