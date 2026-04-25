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

A comprehensive breakdown of all files and directories in this repository.

```text
wildfire-survival-prediction-main/
│
├── Data Science Instuctions.txt        # Initial project instructions and guidelines
├── LICENSE                             # MIT License for this repository
├── README.md                           # Main documentation file (this document)
├── requirements.txt                    # Python package dependencies required to run the code
│
├── docs/                               # Project documentation and presentation materials
│   ├── Final_Reflection_Report.pdf     # Team reflection on the project process and learnings
│   ├── Interview_Presentation.pdf      # Stakeholder-facing presentation covering problem, approach, and results
│   └── Team_Presentation_Guidebook.pdf # Q&A reference guide with glossary and analogy explanations
│
├── extracted_visuals/                  # Figures, diagrams, and screenshots used in docs and notebooks
│   ├── WIDS_registration_screenshots/  # Proof of registration for the WiDS Datathon
│   │   ├── Email_Registered_WIDS_Datathon_2026.png
│   │   ├── End_of_registration_WIDS_Datathon.png
│   │   └── WIDS_Datathon_Global_Challenge_2026_Registration.png
│   ├── Kaggle_*.png                    # Screenshots detailing the Kaggle competition overview and datasets
│   ├── actionable_predictions.png      # Visualization of prediction outcomes
│   ├── eda_kinematic_validation.png    # Exploratory Data Analysis kinematics plots
│   ├── model_committee.png             # Ensemble model architecture diagram
│   ├── permutation_feature_importance.png # Feature importance plot
│   ├── technical_*.png                 # Technical diagrams explaining feature engineering and cross-validation
│   └── WiDS-Global-Datathon-2026.jpg   # WiDS 2026 Datathon Banner
│
├── notebooks/                          # Jupyter Notebooks for EDA and modeling
│   └── wids_2026_wildfire_survival_notebook.ipynb # Full end-to-end modeling pipeline
│
├── outputs/                            # Generated predictions and model outputs
│   └── final_submission.csv            # Final Kaggle submission file containing probability predictions
│
└── src/                                # Source code for data processing and feature engineering
    └── feature_engineering.py          # Physics-based feature pipeline logic
```

---

## Getting Started

Follow these step-by-step instructions to replicate our environment and run the full modeling pipeline from scratch.

### 1. Clone the Repository

First, clone this repository to your local machine and navigate into the project directory:

```bash
git clone https://github.com/your-username/wildfire-survival-prediction.git
cd wildfire-survival-prediction
```

### 2. Download the Competition Data

Since the raw dataset is large, we do not track it in this repository. You must download it directly from Kaggle.
1. Sign up for the WiDS Datathon 2026 Kaggle Competition.
2. Download `train.csv` and `test.csv` from the Data tab.
3. Create a `data/raw/` directory at the root of this project and place the CSV files inside.

```bash
mkdir -p data/raw
# Place train.csv and test.csv here
```

### 3. Set Up the Python Environment

We strongly recommend using a virtual environment to avoid dependency conflicts. This project uses Python 3.9+.

**For Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**For macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Execute the Pipeline

With the data in place and the environment activated, you can now run the core modeling notebook.

1. Start Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
2. Navigate to the `notebooks/` directory and open `wids_2026_wildfire_survival_notebook.ipynb`.
3. Select "Run All" from the "Cell" menu.

The notebook executes the following steps end-to-end:
- **Data Loading:** Imports the Kaggle datasets.
- **Feature Engineering:** Calls the functions defined in `src/feature_engineering.py` to calculate physics-based metrics (`danger_index`, `time_to_impact_est`, etc.).
- **Exploratory Data Analysis (EDA):** Generates visualizations of the data distribution.
- **Validation Setup:** Configures Stratified K-Fold cross-validation.
- **Hyperparameter Tuning:** Uses Optuna to find the optimal settings for the Random Survival Forest and Gradient Boosting models.
- **Model Training & Prediction:** Trains the ensemble and enforces monotonic probability bounds.
- **Output:** Saves the test predictions to `outputs/final_submission.csv` for Kaggle evaluation.

---

## Documentation Guide

For a deeper dive into our methodology, results, and team reflections, please refer to the documents in the `docs/` folder:

| Document | Purpose & Description |
|---|---|
| `Interview_Presentation.pdf` | **Stakeholder Presentation**: A high-level overview of the wildfire survival prediction problem, our kinematic modeling approach, and the actionable business impact of our results. |
| `Team_Presentation_Guidebook.pdf` | **Q&A Reference Guide**: A supplementary document containing a glossary of terms, technical deep dives, and race car analogies used to explain complex hyperparameters to non-technical audiences. |
| `Final_Reflection_Report.pdf` | **Project Retrospective**: A comprehensive team reflection on the data science process, challenges overcome, what we learned, and how we plan to apply these skills in future projects. |

---

*WiDS Global Datathon 2026 · Houston City College · ITAI 2377 · Spring 2026*
