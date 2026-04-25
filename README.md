# WiDS Datathon 2026 — Wildfire Survival Prediction

<div align="center">
  <img src="extracted_visuals/WiDS-Global-Datathon-2026.jpg" alt="WiDS 2026 Datathon Banner" width="100%">
</div>

<br>

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-scikit--survival-green)](https://scikit-survival.readthedocs.io/)
[![Score](https://img.shields.io/badge/C--index-0.9737-orange)](#validation)
[![Validation](https://img.shields.io/badge/Validation-Stratified_K--Fold-purple)](#validation)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

**Houston City College — ITAI 2377 Data Science in Artificial Intelligence — Spring 2026**

**Team:** DeMarcus Crump · Chloe Tu · Akinbobola Akinpelu · Aima Ayaz

---

## Overview

This repository contains our full solution for the **WiDS 2026 Global Datathon** — a Kaggle competition hosted by Women in Data Science. The task: given sensor and environmental data from an active wildfire, predict the probability it reaches a designated community evacuation zone within **12, 24, 48, or 72 hours**.

With only **221 fire trajectories** in the training set and roughly **31% positive events**, standard classification approaches were unsuitable. Deep learning and complex gradient boosters would memorize the training data rather than generalize. Our solution centers on **Kinematic Survival Analysis** — a framework built for exactly this kind of small, time-to-event, right-censored dataset.

---

## Approach

### Why Survival Analysis

Survival analysis treats the problem as "when will it happen?" rather than "will it happen?" This matters for two reasons:

1. **Time-sensitivity** — the competition asks for probabilities across four time horizons, not a single binary outcome.
2. **Right-censored data** — fires contained before reaching the zone (69% of training data) are not failures; they carry meaningful information about what slows a fire down. Survival analysis uses these observations rather than discarding them.

### Physics-Based Feature Engineering

Rather than feeding raw sensor columns directly into a model, we engineered three features grounded in fire behavior:

| Feature | Description |
|---|---|
| `danger_index` | Closing speed × trajectory alignment — a composite fire threat score |
| `time_to_impact_est` | Physics-based ETA: distance to zone ÷ closing speed |
| `growth_impact` | Area growth rate × trajectory alignment |
| `dist_accel_impact` | Distance acceleration relative to closing speed |

These engineered features consistently outperformed the raw sensor columns in ablation testing.

### Model Architecture

An ensemble of two survival models, combined via weighted averaging:

- **Random Survival Forest (RSF)** — captures nonlinear relationships; robust on small datasets
- **Gradient Boosting Survival Analysis (GBSA)** — sequential error correction via boosting
- **Bayesian hyperparameter tuning** via Optuna (50 trials per model)
- **Monotonic probability bounds** enforced post-prediction to ensure P(12h) ≤ P(24h) ≤ P(48h) ≤ P(72h)

---

## Validation

Honest evaluation was a core priority. Competition leaderboards can be gamed by overfitting to the public test set — we built validation from the ground up instead.

- **Stratified K-Fold (k=5):** Each fold maintains the real class distribution (~31% events), preventing folds with artificially easy or hard splits.
- **Out-of-fold scoring:** The reported score of **0.9737 (C-index)** is an average across 5 held-out test sets the model never saw during training — not a leaderboard number.
- **Safe survival wrapper:** A custom `safe_surv_val` function (ε = 1e-4) prevents numerical collapse when test event times exceed the training time horizon.

The concordance index (C-index) measures rank-ordering accuracy: in 97.37% of pairwise fire comparisons, the model correctly predicted which fire would arrive first.

---

## Project Structure

```text
├── data/
│   └── raw/                    # Competition CSVs (not tracked — see .gitignore)
├── src/
│   └── feature_engineering.py  # Physics-based feature pipeline
├── outputs/
│   └── final_submission.csv    # Final Kaggle submission
├── extracted_visuals/          # Figures and charts from the notebook
├── docs/
│   ├── Interview_Presentation.html/.pdf      # Stakeholder presentation deck
│   ├── Final_Reflection_Report.pdf           # Team reflection document
│   └── Team_Presentation_Guidebook.html/.pdf # Presentation reference guide
├── wids_2026_wildfire_survival_notebook.ipynb  # Full modeling pipeline
├── requirements.txt
└── README.md
```

---

## Getting Started

**1. Add competition data**

Place `train.csv` and `test.csv` inside `data/raw/`.

**2. Set up the environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Run the notebook**

Open `wids_2026_wildfire_survival_notebook.ipynb` in Jupyter. The notebook runs end-to-end: loads and engineers features via `src/feature_engineering.py`, runs EDA, builds Stratified K-Fold cross-validation, tunes and trains both survival models, and outputs final probability predictions to `outputs/`.

---

## Documents

| Document | Description |
|---|---|
| `docs/Interview_Presentation.pdf` | Stakeholder-facing presentation covering the problem, approach, and results |
| `docs/Final_Reflection_Report.pdf` | Team reflection on the project process and learnings |
| `docs/Team_Presentation_Guidebook.pdf` | Q&A reference guide with glossary and analogy explanations |

---

*WiDS Global Datathon 2026 · Houston City College · ITAI 2377 · Spring 2026*
