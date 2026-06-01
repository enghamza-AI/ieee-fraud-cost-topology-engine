# Fraud Cost Intelligence System

> *"The threshold that saves money depends on whose money you're saving."*

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Topaz%20Space-blue)](https://huggingface.co/spaces/enghamza-AI/topaz)
[![GitHub](https://img.shields.io/badge/GitHub-enghamza--AI-black?logo=github)](https://github.com/enghamza-AI)
[![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikitlearn)](https://scikit-learn.org)
[![Stage](https://img.shields.io/badge/Roadmap-Stage%202%20%E2%80%94%20Week%201-purple)]()

---

## The Problem

Every fraud detection model outputs a number between 0 and 1.  
You pick a threshold — say `0.5` — and call everything above it "fraud."

That threshold is arbitrary. And it costs someone money.

Three stakeholders walk into a room. They all use the same model. They all want a different threshold — and they are **all correct**:

| Stakeholder | Priority | What a wrong prediction costs them |
|---|---|---|
| 🏦 Bank Risk Manager | Catch every fraud, even at the cost of false alarms | $200–$50,000 per missed fraud case |
| 🛒 Merchant (e-commerce) | Never block a real customer | $50–$500 per blocked legitimate sale + reputation |
| ⚖️ Financial Regulator | Minimum 80% fraud recall — by law | Regulatory fine per uncaught fraud incident |

**This project answers:** what is the mathematically optimal decision threshold for each stakeholder, and what does it cost everyone else?

---

## Dataset

**IEEE-CIS Fraud Detection** — [Kaggle Competition 2019](https://www.kaggle.com/c/ieee-fraud-detection)

| Property | Value |
|---|---|
| Source | Vesta Corporation (real anonymised e-commerce data) |
| Transactions | 590,540 rows |
| Fraud rate | ~3.5% (20,663 fraud / 569,877 legit) |
| Features | 433 (transaction + identity tables) |
| Label | `isFraud` — binary (0 = legit, 1 = fraud) |

> **Why this dataset?**  
> The 3.5% fraud rate is the point. A model that predicts "legit" for every single transaction achieves **96.5% accuracy** while catching zero fraud. This is the accuracy trap — and this project proves it.

---

## What This Project Does

### 1. Proves the Accuracy Trap
Trains a baseline classifier. Shows that accuracy is a dangerously misleading metric on imbalanced data. A dummy classifier that predicts "legit" every time achieves 96.5% accuracy — this project shows that number next to a confusion matrix that reveals it caught 0 fraud cases.

### 2. Builds a Full Confusion Matrix Analysis
Decomposes every prediction into TP / TN / FP / FN. Explains the real-world consequence of each cell in the context of fraud:

```
                  Predicted: FRAUD    Predicted: LEGIT
Actual: FRAUD   [ True Positive  ]  [ False Negative ]   ← missed fraud
Actual: LEGIT   [ False Positive ]  [ True Negative  ]   ← blocked customer
```

### 3. Generates a 3D Animated Cost Surface
For every combination of `FP_cost` and `FN_cost`, computes the total business loss across all decision thresholds and finds the minimum. The result is a 3D surface — animated and rotating — showing exactly how the cost landscape shifts depending on what mistakes you consider expensive.

### 4. Produces a Stakeholder Threshold Report
A clean table: for each of the three stakeholders, the optimal threshold, the resulting TP/FP/FN/TN counts, and the projected financial impact.

### 5. Ships an Interactive Streamlit App
Sliders for FP cost and FN cost. Live confusion matrix. Live 3D cost surface. Any user can drag a slider and watch the optimal threshold move in real time.

---

## Architecture

```
fraud-cost-intelligence/
│
├── data/
│   └── .gitkeep                  # dataset downloaded locally, not committed
│
├── notebooks/
│   └── 01_exploration.ipynb      # EDA, class imbalance analysis
│
├── src/
│   ├── data_loader.py            # loads, merges, samples IEEE-CIS tables
│   ├── pipeline.py               # sklearn pipeline: impute → scale → classify
│   ├── confusion_analysis.py     # confusion matrix + accuracy trap proof
│   ├── cost_surface.py           # 3D cost surface computation + animation
│   └── stakeholder_report.py     # per-stakeholder optimal threshold finder
│
├── app/
│   └── streamlit_app.py          # interactive Streamlit frontend
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── cost_surface.gif          # animated 3D surface
│   └── stakeholder_report.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technical Stack

| Tool | Purpose |
|---|---|
| `pandas` | Data loading, merging, feature selection |
| `scikit-learn` | Pipeline, imputer, scaler, LogisticRegression, RandomForest |
| `matplotlib` | Confusion matrix heatmap, 3D surface, animation |
| `numpy` | Threshold sweep, cost computation grid |
| `streamlit` | Interactive web app |
| `scipy.optimize` | (Week 4 extension) formal threshold optimisation |

---

## Key Results

> *(Populated after implementation — placeholder structure below)*

| Metric | Dummy Classifier | Trained Model (threshold=0.5) | Optimal (Bank) | Optimal (Merchant) | Optimal (Regulator) |
|---|---|---|---|---|---|
| Accuracy | 96.5% | — | — | — | — |
| Fraud Recall | 0% | — | — | — | — |
| False Positive Rate | 0% | — | — | — | — |
| Optimal Threshold | N/A | 0.5 | — | — | — |
| Projected Loss ($) | — | — | — | — | — |

---

## The Core Insight

Accuracy hides the truth on imbalanced data.

A 96.5% accurate fraud detector that catches 0% of fraud is not a model — it is a liability. The confusion matrix exposes what accuracy buries. The cost surface shows that there is no single correct threshold — only the correct threshold *for a given stakeholder's cost structure*.

This is the decision intelligence problem. Not "is the model good?" but "what decision should the model drive, and for whom?"

---

## Live Demo

🚀 **[Launch on HuggingFace Spaces — Topaz](https://huggingface.co/spaces/enghamza-AI/topaz)**

The app allows you to:
- Upload your own prediction CSV (probability scores + true labels)
- Set FP cost and FN cost with sliders
- See the confusion matrix update live
- Get the optimal threshold for your cost structure
- View the full 3D cost surface for your dataset

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/enghamza-AI/fraud-cost-intelligence
cd fraud-cost-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Go to: https://www.kaggle.com/c/ieee-fraud-detection/data
# Place train_transaction.csv and train_identity.csv inside /data/

# 4. Run the Streamlit app
streamlit run app/streamlit_app.py
```

---

## What I Learned

This project was built as part of my **Diamond AI Roadmap** — a structured self-study programme targeting AI systems engineering roles at research labs and AI-first startups (Anthropic, xAI, OpenAI, Perplexity, YC-backed companies).

### Concepts I went deep on

**Confusion Matrix**
Before this project I thought accuracy was the right way to measure a model. The confusion matrix showed me that accuracy is a one-number summary that hides four very different outcomes. On a dataset with 96.5% negative class, a model can achieve near-perfect accuracy by being completely blind to the minority class. The confusion matrix makes this visible immediately.

**The Accuracy Trap**
The dummy classifier experiment was the moment this clicked. Predicting "legit" for every transaction — no model, no training, no features — achieves 96.5% accuracy. Any model I build must beat a dummy classifier by a meaningful margin, not just in accuracy, but specifically in catching the minority class.

**False Positive vs False Negative Cost Asymmetry**
These are not equal mistakes. In fraud: a false negative (missed fraud) costs the bank hundreds to thousands of dollars. A false positive (blocked real customer) costs the merchant a sale and trust. The decision of which mistake to tolerate is a *business decision*, not a model decision. The model gives probabilities. The human (or the system designer) sets the threshold that encodes which mistake they prefer.

**3D Cost Surface**
When I swept FP cost and FN cost as independent variables and plotted total business loss across every threshold combination, the surface revealed something non-obvious: the optimal threshold is not a fixed number. It is a function of the cost ratio. As FN_cost increases relative to FP_cost, the optimal threshold drops — the model becomes more aggressive at flagging fraud, accepting more false alarms to avoid missing real fraud. Visualising this as a 3D surface made the mathematics intuitive.

**Decision Threshold as a Design Parameter**
The most important shift in thinking from Stage 1: the threshold is not 0.5 by default. It is a parameter you tune based on who is making the decision and what they can afford to get wrong. This is what "decision intelligence" means.

### Skills I practiced
- Loading and merging large multi-table datasets (590k rows) in pandas without memory crashes
- Building sklearn pipelines with imputation, scaling, and a classifier as one object
- Computing confusion matrices at multiple thresholds programmatically
- Building and animating 3D matplotlib surfaces
- Designing Streamlit apps for non-technical users

---

## About Me

**Hamza** — BSAI student, 5th semester, based in China.  
Self-studying AI systems engineering on top of my degree, targeting research lab and AI-first startup roles by graduation (early 2028).

- 🤗 HuggingFace: [huggingface.co/spaces/enghamza-AI](https://huggingface.co/spaces/enghamza-AI)  
- 💻 GitHub: [github.com/enghamza-AI](https://github.com/enghamza-AI)

> *This project is Stage 2, Week 1 of the Diamond AI Roadmap — a project-based curriculum covering evaluation mastery, decision intelligence, systems design, and production ML.*

---

## Roadmap Position

```
Stage 1 — Signal & Pattern Recognition       ✅ Complete
Stage 2 — Decision Intelligence              🔄 In Progress
  Week 1 — Confusion Matrix & Cost Surface   ← YOU ARE HERE
  Week 2 — Precision, Recall, F1
  Week 3 — Class Imbalance Tournament
  Week 4 — Cost Matrices & Pareto Frontier
Stage 3 — Systems Design                     ⏳ Upcoming
Stage 4 — Production ML                      ⏳ Upcoming
```

---

*Built with curiosity. Documented for the next engineer who wonders why their 99% accurate model is useless.*
