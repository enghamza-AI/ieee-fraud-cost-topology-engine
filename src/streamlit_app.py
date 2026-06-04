
# streamlit_app.py
# LLM ASSISST!

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.datasets import make_classification
import io
import os
import sys
import warnings
warnings.filterwarnings('ignore')   


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, '.')


try:
    from confusion_analysis import prove_accuracy_trap, plot_confusion_matrix, plot_threshold_sweep
    from cost_surface import compute_cost_grid, plot_cost_surface_static, find_optimal_threshold
    from stakeholder_report import build_stakeholder_report, DEFAULT_STAKEHOLDERS
    MODULES_LOADED = True
except ImportError:
    MODULES_LOADED = False
   



st.set_page_config(
    page_title="Fraud Cost Intelligence | Topaz",
    page_icon="💎",
    layout="wide",                  
    initial_sidebar_state="expanded"
)



st.markdown("""
<style>
/* ── Import fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Root variables ───────────────────────────────────────── */
:root {
    --bg-primary   : #0d0f14;
    --bg-secondary : #141720;
    --bg-card      : #1a1e2a;
    --accent       : #f5a623;       /* amber — financial terminal */
    --accent-dim   : #b37a1a;
    --danger       : #e74c3c;
    --success      : #2ecc71;
    --warning      : #f39c12;
    --info         : #3498db;
    --text-primary : #e8eaf0;
    --text-secondary: #8892a4;
    --border       : #252a38;
    --border-accent: #f5a62340;
}

/* ── Global reset ─────────────────────────────────────────── */
.stApp {
    background-color: var(--bg-primary);
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-primary);
}

/* ── Hide Streamlit branding ──────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Metric cards ─────────────────────────────────────────── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.metric-card.danger  { border-left-color: var(--danger); }
.metric-card.success { border-left-color: var(--success); }
.metric-card.info    { border-left-color: var(--info); }

.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 4px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1;
}
.metric-sub {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 4px;
}

/* ── Section headers ──────────────────────────────────────── */
.section-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.25rem;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}
.section-sub {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
}

/* ── Insight boxes ────────────────────────────────────────── */
.insight-box {
    background: var(--bg-card);
    border: 1px solid var(--border-accent);
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-size: 13px;
    line-height: 1.7;
    color: var(--text-primary);
}
.insight-box strong { color: var(--accent); }

/* ── Stakeholder cards ────────────────────────────────────── */
.stakeholder-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.stakeholder-card:hover { border-color: var(--accent); }
.stakeholder-name {
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 6px;
}
.stakeholder-threshold {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: var(--accent);
}
.stakeholder-meta {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 4px;
}

/* ── Table styling ────────────────────────────────────────── */
.stDataFrame { border: 1px solid var(--border) !important; }

/* ── Slider labels ────────────────────────────────────────── */
.stSlider label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Tab styling ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
    color: var(--text-secondary);
    padding: 0.6rem 1.2rem;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}

/* ── Hero header ──────────────────────────────────────────── */
.hero-wrap {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 4px;
    padding: 1.75rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '◈';
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 80px;
    color: var(--accent);
    opacity: 0.06;
}
.hero-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.4rem;
}
.hero-title {
    font-size: 26px;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 13px;
    color: var(--text-secondary);
}

/* ── Threshold badge ──────────────────────────────────────── */
.threshold-badge {
    display: inline-block;
    background: var(--accent);
    color: #000;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 2px;
}

/* ── matplotlib dark background ──────────────────────────── */
/* Set globally so all charts match the dark theme */
</style>
""", unsafe_allow_html=True)


plt.rcParams.update({
    'figure.facecolor'  : '#141720',
    'axes.facecolor'    : '#1a1e2a',
    'axes.edgecolor'    : '#252a38',
    'axes.labelcolor'   : '#8892a4',
    'text.color'        : '#e8eaf0',
    'xtick.color'       : '#8892a4',
    'ytick.color'       : '#8892a4',
    'grid.color'        : '#252a38',
    'grid.alpha'        : 0.5,
    'axes.grid'         : True,
    'font.family'       : 'monospace',
    'axes.titlecolor'   : '#e8eaf0',
    'figure.dpi'        : 120,
})




@st.cache_data(show_spinner=False)
def generate_demo_data(n_samples: int = 8000, fraud_rate: float = 0.035, seed: int = 42):
  
    rng = np.random.default_rng(seed)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    
    fraud_features = rng.standard_normal((n_fraud, 15))
    fraud_features[:, 0] += 2.5    
    fraud_features[:, 1] -= 1.5   
    fraud_features[:, 3] += 1.8    

    
    legit_features = rng.standard_normal((n_legit, 15))

    X = pd.DataFrame(
        np.vstack([fraud_features, legit_features]),
        columns=[f'V{i}' for i in range(1, 16)]
    )
    y = pd.Series(
        np.concatenate([np.ones(n_fraud), np.zeros(n_legit)]).astype(int)
    )

    
    idx = rng.permutation(len(y))
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def train_model(X_hash: str, _X: pd.DataFrame, _y: pd.Series):
  
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, stratify=_y, random_state=42
    )

    pipeline = Pipeline([
        ('imputer',    SimpleImputer(strategy='median')),
        ('scaler',     StandardScaler()),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=500,
            random_state=42,
            solver='lbfgs',
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_prob = np.clip(y_prob, 0.0, 1.0)

    return y_test.reset_index(drop=True), y_prob


def compute_cm_at_threshold(y_test, y_prob, threshold):
    
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tp, fp, fn, tn


def find_opt_threshold_inline(y_test, y_prob, fp_cost, fn_cost, n=200):
    
    thresholds  = np.linspace(0.01, 0.99, n)
    costs       = np.zeros(n)
    y_arr       = np.array(y_test)

    for i, t in enumerate(thresholds):
        y_pred  = (y_prob >= t).astype(int)
        fp      = np.sum((y_pred == 1) & (y_arr == 0))
        fn      = np.sum((y_pred == 0) & (y_arr == 1))
        costs[i] = fp * fp_cost + fn * fn_cost

    best = np.argmin(costs)
    return float(thresholds[best]), float(costs[best])



with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1.5rem'>
        <div style='font-family:IBM Plex Mono,monospace;font-size:10px;
                    letter-spacing:3px;text-transform:uppercase;
                    color:#f5a623;margin-bottom:4px'>TOPAZ</div>
        <div style='font-size:16px;font-weight:600;color:#e8eaf0'>
            Fraud Cost Intelligence
        </div>
        <div style='font-size:11px;color:#8892a4;margin-top:3px'>
            Stage 2 · Week 1 · enghamza-AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

  
    st.markdown('<div class="section-tag">Data Source</div>', unsafe_allow_html=True)

    data_mode = st.radio(
        "Choose data",
        ["🔬 Demo (synthetic)", "📁 Upload CSV"],
        help="Demo mode uses synthetic fraud-like data. Upload mode accepts "
             "a CSV with numeric features and a binary 'label' column (0/1)."
    )

    uploaded_file = None
    if data_mode == "📁 Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload predictions CSV",
            type=['csv'],
            help="CSV must have: numeric feature columns + a 'label' column (0=legit, 1=fraud). "
                 "Or a 'y_true' + 'y_prob' column for pre-computed probabilities."
        )

    
    if data_mode == "🔬 Demo (synthetic)":
        demo_size = st.select_slider(
            "Demo dataset size",
            options=[2000, 5000, 8000, 15000],
            value=8000,
            help="Larger = more realistic metrics. Smaller = faster."
        )
    else:
        demo_size = 8000

    st.divider()

    
    st.markdown('<div class="section-tag">Cost Structure</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#8892a4;margin-bottom:0.75rem">'
        'Set costs to find YOUR optimal threshold</div>',
        unsafe_allow_html=True
    )

    fp_cost_slider = st.slider(
        "FP Cost — False Alarm ($)",
        min_value=10, max_value=1000, value=100, step=10,
        help="Cost of blocking one real (legit) customer. "
             "Includes lost sale, dispute handling, churn risk."
    )

    fn_cost_slider = st.slider(
        "FN Cost — Missed Fraud ($)",
        min_value=50, max_value=10000, value=1500, step=50,
        help="Cost of letting one fraudulent transaction through. "
             "Includes fraud loss, chargeback fee, regulatory risk."
    )

    # Live cost ratio display
    ratio = fn_cost_slider / fp_cost_slider
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;'
        f'color:#8892a4;margin-top:0.5rem">'
        f'FN/FP ratio: <span style="color:#f5a623">{ratio:.1f}×</span> — '
        f'{"FN-heavy: lower threshold" if ratio > 5 else "Balanced" if ratio > 2 else "FP-heavy: higher threshold"}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.divider()

    
    st.markdown("""
    <div style='font-size:11px;color:#8892a4;line-height:2'>
        <a href='https://github.com/enghamza-AI' target='_blank'
           style='color:#f5a623;text-decoration:none'>
            ↗ GitHub: enghamza-AI
        </a><br>
        <a href='https://huggingface.co/spaces/enghamza-AI'
           target='_blank' style='color:#f5a623;text-decoration:none'>
            ↗ HuggingFace Spaces
        </a><br>
        
    </div>
    """, unsafe_allow_html=True)



if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)

        
        if 'label' in df_upload.columns:
            y_raw = df_upload['label'].astype(int)
            X_raw = df_upload.drop(columns=['label'])
            X_raw = X_raw.select_dtypes(include=[np.number])
            data_hash = str(len(df_upload)) + str(df_upload.columns.tolist())
            y_test, y_prob = train_model(data_hash, X_raw, y_raw)
            data_label = f"Uploaded — {len(df_upload):,} rows"

        
        elif 'y_true' in df_upload.columns and 'y_prob' in df_upload.columns:
            y_test = df_upload['y_true'].astype(int).reset_index(drop=True)
            y_prob = df_upload['y_prob'].values
            y_prob = np.clip(y_prob, 0.0, 1.0)
            data_label = f"Uploaded probabilities — {len(df_upload):,} rows"

        else:
            st.sidebar.error(
                "CSV must have a 'label' column (0/1) "
                "OR 'y_true' + 'y_prob' columns."
            )
            st.stop()    

    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        st.stop()

else:
    
    with st.spinner("Generating demo data and training model..."):
        X_demo, y_demo = generate_demo_data(n_samples=demo_size)
        data_hash = f"demo_{demo_size}"
        y_test, y_prob = train_model(data_hash, X_demo, y_demo)
    data_label = f"Synthetic demo — {demo_size:,} rows (3.5% fraud)"




total_fraud   = int(y_test.sum())
total_legit   = int(len(y_test) - total_fraud)
total         = len(y_test)
fraud_pct     = total_fraud / total * 100

auc_score     = roc_auc_score(y_test, y_prob)


opt_thresh, opt_cost = find_opt_threshold_inline(
    y_test, y_prob, fp_cost_slider, fn_cost_slider
)


tp_opt, fp_opt, fn_opt, tn_opt = compute_cm_at_threshold(y_test, y_prob, opt_thresh)


tp_50, fp_50, fn_50, tn_50 = compute_cm_at_threshold(y_test, y_prob, 0.5)


acc_dummy = (total_legit) / total   



st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-tag">◈ Stage 2 · Week 1 · Decision Intelligence</div>
    <div class="hero-title">Fraud Cost Intelligence System</div>
    <div class="hero-sub">
        {data_label} &nbsp;·&nbsp;
        AUC-ROC: <strong style="color:#f5a623">{auc_score:.4f}</strong> &nbsp;·&nbsp;
        Fraud rate: <strong style="color:#f5a623">{fraud_pct:.1f}%</strong> &nbsp;·&nbsp;
        Optimal threshold: <span class="threshold-badge">{opt_thresh:.3f}</span>
    </div>
</div>
""", unsafe_allow_html=True)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card success">
        <div class="metric-label">AUC-ROC</div>
        <div class="metric-value">{auc_score:.4f}</div>
        <div class="metric-sub">Real discriminative power</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card danger">
        <div class="metric-label">Dummy Accuracy</div>
        <div class="metric-value">{acc_dummy*100:.1f}%</div>
        <div class="metric-sub">Catches 0 fraud — the trap</div>
    </div>""", unsafe_allow_html=True)

with col3:
    recall_opt = tp_opt / total_fraud if total_fraud > 0 else 0
    st.markdown(f"""
    <div class="metric-card info">
        <div class="metric-label">Fraud Recall (optimal)</div>
        <div class="metric-value">{recall_opt*100:.1f}%</div>
        <div class="metric-sub">At threshold {opt_thresh:.3f}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    cost_default = fp_50 * fp_cost_slider + fn_50 * fn_cost_slider
    saving = cost_default - opt_cost
    saving_str = f"+${saving:,.0f}" if saving > 0 else f"-${abs(saving):,.0f}"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Cost Saving vs 0.5</div>
        <div class="metric-value">{saving_str}</div>
        <div class="metric-sub">From threshold optimisation</div>
    </div>""", unsafe_allow_html=True)



tab1, tab2, tab3, tab4 = st.tabs([
    "01 · ACCURACY TRAP",
    "02 · CONFUSION MATRIX",
    "03 · COST SURFACE",
    "04 · STAKEHOLDER REPORT"
])



with tab1:
    st.markdown("""
    <div class="section-tag">Proof</div>
    <div class="section-title">The Accuracy Trap</div>
    <div class="section-sub">
        Why 96%+ accuracy on fraud data is meaningless — and how the confusion matrix exposes it.
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("""
        <div class="insight-box">
        A model that predicts <strong>"legit"</strong> for every single transaction —
        no training, no features, no intelligence — achieves the accuracy you see below.
        It catches <strong>zero fraud</strong>. Yet by the accuracy metric, it would be
        considered an excellent model.<br><br>
        This is the <strong>Accuracy Trap</strong>: on imbalanced datasets, accuracy
        rewards predicting the majority class. The confusion matrix shows the truth.
        </div>
        """, unsafe_allow_html=True)

       
        real_acc = accuracy_score(y_test, (y_prob >= 0.5).astype(int))
        recall_50 = tp_50 / total_fraud if total_fraud > 0 else 0

        st.markdown("**Dummy Classifier — always says Legit**")
        st.markdown(f"""
        <div class="metric-card danger">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{acc_dummy*100:.1f}%</div>
            <div class="metric-sub">Fraud caught: 0 of {total_fraud:,}
            &nbsp;|&nbsp; Fraud missed: {total_fraud:,}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Real Model — threshold = 0.5**")
        st.markdown(f"""
        <div class="metric-card success">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{real_acc*100:.1f}%</div>
            <div class="metric-sub">Fraud caught: {tp_50:,} ({recall_50*100:.1f}%)
            &nbsp;|&nbsp; Fraud missed: {fn_50:,}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-box">
        The real model has <strong>lower accuracy</strong> than the dummy.
        But it catches <strong>{tp_50:,} fraud cases</strong> the dummy misses entirely.
        <br><br>Accuracy went down. Usefulness went up enormously.
        </div>""", unsafe_allow_html=True)

    with col_right:
     
        fig, ax = plt.subplots(figsize=(6, 4))

        categories = ['Accuracy\n(%)', 'Fraud Recall\n(%)', 'Fraud Caught\n(count)']
        dummy_vals = [acc_dummy * 100, 0, 0]
        real_vals  = [real_acc * 100, recall_50 * 100, tp_50 / total_fraud * 100]

        x = np.arange(len(categories))
        w = 0.35

        b1 = ax.bar(x - w/2, dummy_vals, w, label='Dummy (always legit)',
                    color='#e74c3c', alpha=0.8, linewidth=0)
        b2 = ax.bar(x + w/2, real_vals,  w, label='Real model (threshold=0.5)',
                    color='#2ecc71', alpha=0.8, linewidth=0)

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                        f'{h:.1f}%', ha='center', va='bottom',
                        fontsize=8, color='#e8eaf0', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 115)
        ax.set_title('Dummy vs Real Model', fontsize=10, pad=10)
        ax.legend(fontsize=8)
        ax.set_ylabel('Value (%)', fontsize=8)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    
    st.markdown("---")
    st.markdown("**How every metric changes as you move the threshold**")

    thresholds_sweep = np.linspace(0.01, 0.99, 100)
    tp_r = np.zeros(100)
    fp_r = np.zeros(100)
    fn_r = np.zeros(100)
    tn_r = np.zeros(100)
    y_arr = np.array(y_test)

    for i, t in enumerate(thresholds_sweep):
        y_p = (y_prob >= t).astype(int)
        tp_r[i] = np.sum((y_p == 1) & (y_arr == 1)) / total_fraud
        fp_r[i] = np.sum((y_p == 1) & (y_arr == 0)) / total_legit
        fn_r[i] = np.sum((y_p == 0) & (y_arr == 1)) / total_fraud
        tn_r[i] = np.sum((y_p == 0) & (y_arr == 0)) / total_legit

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 3.5))

    ax_l = axes2[0]
    ax_l.plot(thresholds_sweep, tp_r * 100, color='#2ecc71', lw=2, label='Fraud Recall (TP%)')
    ax_l.plot(thresholds_sweep, fn_r * 100, color='#e74c3c', lw=2, ls='--', label='Miss Rate (FN%)')
    ax_l.axvline(x=0.5,          color='#8892a4', ls=':', alpha=0.6, lw=1, label='Default 0.5')
    ax_l.axvline(x=opt_thresh,   color='#f5a623', ls='-', alpha=0.8, lw=1.5, label=f'Optimal {opt_thresh:.2f}')
    ax_l.set_xlabel('Threshold')
    ax_l.set_ylabel('%')
    ax_l.set_title('Fraud Detection vs Threshold')
    ax_l.legend(fontsize=8)
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(0, 110)

    ax_r = axes2[1]
    ax_r.plot(thresholds_sweep, fp_r * 100, color='#f39c12', lw=2, label='False Alarm Rate (FP%)')
    ax_r.plot(thresholds_sweep, tn_r * 100, color='#3498db', lw=2, ls='--', label='Correct Pass (TN%)')
    ax_r.axvline(x=0.5,          color='#8892a4', ls=':', alpha=0.6, lw=1)
    ax_r.axvline(x=opt_thresh,   color='#f5a623', ls='-', alpha=0.8, lw=1.5)
    ax_r.set_xlabel('Threshold')
    ax_r.set_ylabel('%')
    ax_r.set_title('Legit Customer Rates vs Threshold')
    ax_r.legend(fontsize=8)
    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(0, 110)

    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)



with tab2:
    st.markdown("""
    <div class="section-tag">Live</div>
    <div class="section-title">Confusion Matrix</div>
    <div class="section-sub">
        Adjust cost sliders in the sidebar — threshold and matrix update instantly.
    </div>
    """, unsafe_allow_html=True)

    
    use_optimal = st.checkbox("Use cost-optimal threshold (recommended)", value=True)

    if use_optimal:
        active_threshold = opt_thresh
    else:
        active_threshold = st.slider(
            "Manual threshold",
            min_value=0.01, max_value=0.99,
            value=float(round(opt_thresh, 2)), step=0.01
        )

    tp_a, fp_a, fn_a, tn_a = compute_cm_at_threshold(y_test, y_prob, active_threshold)
    recall_a    = tp_a / total_fraud
    fpr_a       = fp_a / total_legit
    precision_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0
    acc_a       = (tp_a + tn_a) / total
    cost_a      = fp_a * fp_cost_slider + fn_a * fn_cost_slider

    col_cm, col_stats = st.columns([1.2, 1])

    with col_cm:
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))

        cm_data = np.array([[tn_a, fp_a], [fn_a, tp_a]])

       
        cell_vals  = [tn_a, fp_a, fn_a, tp_a]
        cell_clrs  = ['#1a3a2a', '#2a2010', '#2a1010', '#0f2a1a']   # bg colours
        cell_tclrs = ['#2ecc71', '#f39c12', '#e74c3c', '#2ecc71']   # text/accent colours
        cell_lbls  = ['TRUE NEGATIVE', 'FALSE POSITIVE', 'FALSE NEGATIVE', 'TRUE POSITIVE']
        cell_descs = [
            'Legit → correctly\npassed through ✓',
            'Legit → wrongly\nblocked (false alarm)',
            'Fraud → wrongly\npassed through ⚠',
            'Fraud → correctly\ncaught ✓'
        ]

        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for (r, c), val, bg, tc, lbl, desc in zip(
            positions, cell_vals, cell_clrs, cell_tclrs, cell_lbls, cell_descs
        ):
            ax_cm.add_patch(plt.Rectangle((c, 1-r), 1, 1, facecolor=bg,
                            edgecolor='#252a38', linewidth=2))
            ax_cm.text(c+0.5, 1-r+0.78, lbl, ha='center', va='center',
                       fontsize=7.5, color=tc, fontweight='bold',
                       fontfamily='monospace')
            ax_cm.text(c+0.5, 1-r+0.52, f'{val:,}', ha='center', va='center',
                       fontsize=18, color='#e8eaf0', fontweight='bold',
                       fontfamily='monospace')
            ax_cm.text(c+0.5, 1-r+0.30, f'{val/total*100:.1f}% of all',
                       ha='center', va='center', fontsize=8, color='#8892a4')
            ax_cm.text(c+0.5, 1-r+0.12, desc, ha='center', va='center',
                       fontsize=7, color='#8892a4', style='italic')

        ax_cm.set_xlim(0, 2)
        ax_cm.set_ylim(0, 2)
        ax_cm.set_xticks([0.5, 1.5])
        ax_cm.set_xticklabels(['Predicted:\nLEGIT', 'Predicted:\nFRAUD'], fontsize=9)
        ax_cm.set_yticks([0.5, 1.5])
        ax_cm.set_yticklabels(['Actual:\nFRAUD', 'Actual:\nLEGIT'], fontsize=9)
        ax_cm.tick_params(length=0)
        ax_cm.set_title(
            f'Confusion Matrix — Threshold: {active_threshold:.3f}',
            fontsize=10, pad=10
        )
        ax_cm.grid(False)

        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    with col_stats:
        st.markdown(f"""
        <div class="metric-card success">
            <div class="metric-label">Fraud Recall</div>
            <div class="metric-value">{recall_a*100:.1f}%</div>
            <div class="metric-sub">{tp_a:,} of {total_fraud:,} fraud caught</div>
        </div>
        <div class="metric-card danger">
            <div class="metric-label">Fraud Missed (FN)</div>
            <div class="metric-value">{fn_a:,}</div>
            <div class="metric-sub">Slipped through undetected</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">False Alarm Rate</div>
            <div class="metric-value">{fpr_a*100:.1f}%</div>
            <div class="metric-sub">{fp_a:,} legit customers blocked</div>
        </div>
        <div class="metric-card info">
            <div class="metric-label">Total Business Cost</div>
            <div class="metric-value">${cost_a:,.0f}</div>
            <div class="metric-sub">At FP=${fp_cost_slider} · FN=${fn_cost_slider}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{precision_a*100:.1f}%</div>
            <div class="metric-sub">Of fraud flags, % correct</div>
        </div>
        """, unsafe_allow_html=True)



with tab3:
    st.markdown("""
    <div class="section-tag">3D Analysis</div>
    <div class="section-title">Cost Surface</div>
    <div class="section-sub">
        For every combination of FP and FN cost — where is the minimum total loss?
        Colour = optimal decision threshold at that point.
    </div>
    """, unsafe_allow_html=True)

    
    @st.cache_data(show_spinner=False)
    def get_cost_grid(y_hash, _y_test, _y_prob):
        
        thresholds   = np.linspace(0.01, 0.99, 50)
        y_arr        = np.array(_y_test)
        n_legit_     = int((y_arr == 0).sum())
        n_fraud_     = int((y_arr == 1).sum())

        fp_cs = np.linspace(10, 800,  25)
        fn_cs = np.linspace(50, 8000, 25)

        fp_counts_ = np.array([
            np.sum(((y_prob >= t).astype(int) == 1) & (y_arr == 0))
            for t in thresholds
        ], dtype=float)
        fn_counts_ = np.array([
            np.sum(((y_prob >= t).astype(int) == 0) & (y_arr == 1))
            for t in thresholds
        ], dtype=float)

        min_cost   = np.zeros((25, 25))
        opt_thresh_ = np.zeros((25, 25))

        for i, fpc in enumerate(fp_cs):
            for j, fnc in enumerate(fn_cs):
                costs   = fp_counts_ * fpc + fn_counts_ * fnc
                best    = np.argmin(costs)
                min_cost[i, j]    = costs[best]
                opt_thresh_[i, j] = thresholds[best]

        FP_m, FN_m = np.meshgrid(fp_cs, fn_cs)
        return fp_cs, fn_cs, min_cost.T, opt_thresh_.T, FP_m, FN_m

    with st.spinner("Computing cost surface..."):
        fp_axis, fn_axis, cost_grid, thresh_grid, FP_m, FN_m = get_cost_grid(
            data_hash, y_test, y_prob
        )

    # Stakeholder presets
    STAKEHOLDERS_DEF = {
        'Bank'      : (50,  1500, '#e74c3c'),
        'Merchant'  : (250, 350,  '#3498db'),
        'Regulator' : (10,  5000, '#f39c12'),
        'Custom'    : (fp_cost_slider, fn_cost_slider, '#f5a623'),
    }

    col_surf, col_legend = st.columns([2, 1])

    with col_surf:
        view_az = st.slider("Rotate surface", 0, 360, 225, 5, key="az_surf")

        fig_3d = plt.figure(figsize=(9, 6))
        ax3d   = fig_3d.add_subplot(111, projection='3d')

        surf = ax3d.plot_surface(
            FP_m, FN_m, cost_grid,
            facecolors=plt.cm.RdYlGn_r(thresh_grid),
            alpha=0.88, rstride=1, cstride=1,
            linewidth=0, antialiased=True
        )

        
        for sname, (fpc, fnc, scolor) in STAKEHOLDERS_DEF.items():
            i = np.argmin(np.abs(fp_axis - fpc))
            j = np.argmin(np.abs(fn_axis - fnc))
            z = cost_grid[j, i]
            ax3d.scatter([fpc], [fnc], [z * 1.05],
                         color=scolor, s=80, zorder=10,
                         depthshade=False, edgecolors='white', linewidth=1)
            ax3d.text(fpc, fnc, z * 1.12, sname,
                      color=scolor, fontsize=7, fontweight='bold')

        ax3d.set_xlabel('FP Cost ($)', fontsize=8, labelpad=6)
        ax3d.set_ylabel('FN Cost ($)', fontsize=8, labelpad=6)
        ax3d.set_zlabel('Min Loss ($)', fontsize=8, labelpad=6)
        ax3d.set_title('Cost Surface — colour = optimal threshold',
                       fontsize=9, pad=10)
        ax3d.view_init(elev=28, azim=view_az)

        mappable = plt.cm.ScalarMappable(
            cmap='RdYlGn_r', norm=plt.Normalize(0, 1)
        )
        mappable.set_array([])
        cbar = fig_3d.colorbar(mappable, ax=ax3d, shrink=0.45, pad=0.08)
        cbar.set_label('Optimal Threshold', fontsize=7)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels(['0.0\naggressive', '0.5', '1.0\nconservative'])

        plt.tight_layout()
        st.pyplot(fig_3d, use_container_width=True)
        plt.close(fig_3d)

    with col_legend:
        st.markdown("""
        <div class="insight-box" style="font-size:12px">
        <strong>How to read this surface:</strong><br><br>
        Each point = one cost structure scenario.<br><br>
        <strong>Height (Z)</strong> = minimum achievable total loss at that cost structure.<br><br>
        <strong>Colour</strong> = the threshold that achieves that minimum.
        Green = aggressive (low threshold, catches more fraud).
        Red = conservative (high threshold, fewer false alarms).<br><br>
        <strong>The shape</strong> tells you: as FN cost rises relative to FP cost,
        the optimal threshold drops — the system becomes more aggressive
        at flagging fraud, accepting more false alarms to avoid
        missing real fraud.
        </div>
        """, unsafe_allow_html=True)

        for sname, (fpc, fnc, scolor) in STAKEHOLDERS_DEF.items():
            t_opt, c_opt = find_opt_threshold_inline(y_test, y_prob, fpc, fnc)
            st.markdown(f"""
            <div class="stakeholder-card" style="border-left:3px solid {scolor}">
                <div class="stakeholder-name" style="color:{scolor}">{sname}</div>
                <div class="stakeholder-threshold">{t_opt:.3f}</div>
                <div class="stakeholder-meta">
                    FP=${fpc} · FN=${fnc}<br>
                    Min cost: ${c_opt:,.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)



with tab4:
    st.markdown("""
    <div class="section-tag">Report</div>
    <div class="section-title">Stakeholder Threshold Report</div>
    <div class="section-sub">
        Same model · same probabilities · three different operating points.
        This is what decision intelligence means.
    </div>
    """, unsafe_allow_html=True)

    # Build full report table
    stakeholder_defs = {
        '🏦 Bank Risk Manager': {
            'fp_cost': 50, 'fn_cost': 1500,
            'priority': 'Minimise fraud losses',
            'min_recall': 0.75, 'color': '#e74c3c'
        },
        '🛒 Merchant': {
            'fp_cost': 250, 'fn_cost': 350,
            'priority': 'Protect customer experience',
            'min_recall': None, 'color': '#3498db'
        },
        '⚖️ Regulator': {
            'fp_cost': 10, 'fn_cost': 5000,
            'priority': 'Maximum fraud recall (legal minimum)',
            'min_recall': 0.85, 'color': '#f39c12'
        },
        '⚙️ Custom': {
            'fp_cost': fp_cost_slider, 'fn_cost': fn_cost_slider,
            'priority': 'Your sidebar cost structure',
            'min_recall': None, 'color': '#f5a623'
        },
    }

    report_rows = []
    y_arr = np.array(y_test)

    for name, cfg in stakeholder_defs.items():
        t_opt, cost_opt = find_opt_threshold_inline(
            y_test, y_prob, cfg['fp_cost'], cfg['fn_cost']
        )

        # Enforce recall constraint if applicable
        if cfg['min_recall']:
            # Lower threshold until recall requirement met
            for t_try in np.linspace(t_opt, 0.01, 200):
                y_p_try = (y_prob >= t_try).astype(int)
                tp_try  = np.sum((y_p_try == 1) & (y_arr == 1))
                if tp_try / total_fraud >= cfg['min_recall']:
                    t_opt = t_try
                    break

        y_pred_s = (y_prob >= t_opt).astype(int)
        tp_s = int(np.sum((y_pred_s == 1) & (y_arr == 1)))
        fp_s = int(np.sum((y_pred_s == 1) & (y_arr == 0)))
        fn_s = int(np.sum((y_pred_s == 0) & (y_arr == 1)))
        tn_s = int(np.sum((y_pred_s == 0) & (y_arr == 0)))

        recall_s    = tp_s / total_fraud
        fpr_s       = fp_s / total_legit
        precision_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0
        cost_s      = fp_s * cfg['fp_cost'] + fn_s * cfg['fn_cost']

        # Default 0.5 cost for comparison
        fp_def_ = int(np.sum(((y_prob >= 0.5).astype(int) == 1) & (y_arr == 0)))
        fn_def_ = int(np.sum(((y_prob >= 0.5).astype(int) == 0) & (y_arr == 1)))
        cost_def_ = fp_def_ * cfg['fp_cost'] + fn_def_ * cfg['fn_cost']

        report_rows.append({
            'Stakeholder'        : name,
            'Optimal Threshold'  : round(t_opt, 3),
            'Fraud Recall'       : f"{recall_s*100:.1f}%",
            'False Alarm Rate'   : f"{fpr_s*100:.1f}%",
            'Precision'          : f"{precision_s*100:.1f}%",
            'TP (caught)'        : f"{tp_s:,}",
            'FN (missed)'        : f"{fn_s:,}",
            'FP (false alarm)'   : f"{fp_s:,}",
            'Total Cost ($)'     : f"${cost_s:,.0f}",
            'Cost at 0.5 ($)'    : f"${cost_def_:,.0f}",
            'Saving ($)'         : f"${cost_def_-cost_s:,.0f}",
            '_color'             : cfg['color'],
        })

   
    card_cols = st.columns(4)
    for col, row in zip(card_cols, report_rows):
        with col:
            st.markdown(f"""
            <div class="stakeholder-card" style="border-left:3px solid {row['_color']}">
                <div style="font-size:11px;font-weight:600;color:{row['_color']};
                            margin-bottom:4px">{row['Stakeholder']}</div>
                <div class="stakeholder-threshold">{row['Optimal Threshold']}</div>
                <div class="stakeholder-meta">
                    Recall: {row['Fraud Recall']}<br>
                    False alarms: {row['False Alarm Rate']}<br>
                    Cost: {row['Total Cost ($)']}<br>
                    Saving: <strong style="color:{row['_color']}">{row['Saving ($)']}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    
    st.markdown("**Full comparison table**")
    display_cols = [c for c in report_rows[0].keys() if not c.startswith('_')]
    df_report = pd.DataFrame(report_rows)[display_cols]
    st.dataframe(df_report, use_container_width=True, hide_index=True)

    # Download button
    csv_bytes = df_report.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇ Download report as CSV",
        data=csv_bytes,
        file_name="stakeholder_report.csv",
        mime="text/csv",
        help="Download the full stakeholder report as a CSV file."
    )

   
    thresholds_display = [r['Optimal Threshold'] for r in report_rows]
    names_display      = [r['Stakeholder'].split(' ')[1] for r in report_rows]

    st.markdown(f"""
    <div class="insight-box">
    <strong>The central insight of this project:</strong><br><br>
    The model didn't change. The probabilities didn't change.
    What changed is the <strong>cost structure</strong> — and that alone
    moved the optimal threshold from
    <span class="threshold-badge">{min(thresholds_display):.3f}</span>
    to
    <span class="threshold-badge">{max(thresholds_display):.3f}</span>
    across stakeholders.<br><br>
    Choosing a threshold is not a technical decision.
    It is a <strong>business decision</strong> — and it should be made explicitly,
    with full awareness of what each wrong prediction costs.
    </div>
    """, unsafe_allow_html=True)



st.markdown("---")
st.markdown(f"""
<div style='text-align:center;padding:1rem 0;
            font-family:IBM Plex Mono,monospace;font-size:10px;
            color:#8892a4;letter-spacing:2px'>
    TOPAZ · FRAUD COST INTELLIGENCE &nbsp;·&nbsp;
    STAGE 2 WEEK 1 · DIAMOND AI ROADMAP &nbsp;·&nbsp;
    <a href='https://github.com/enghamza-AI' target='_blank'
       style='color:#f5a623;text-decoration:none'>ENGHAMZA-AI</a>
    &nbsp;·&nbsp;
    <a href='https://huggingface.co/spaces/enghamza-AI/topaz' target='_blank'
       style='color:#f5a623;text-decoration:none'>HUGGINGFACE</a>
</div>
""", unsafe_allow_html=True)
