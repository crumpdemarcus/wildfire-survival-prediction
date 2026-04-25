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

---

## Detailed Project Structure & Contents

This section provides a comprehensive breakdown of all directories and files in this repository, including detailed explanations of their contents and functionality.

### 1. Documentation (`docs/`)

This folder contains all of the project's formal documentation — including the team's stakeholder presentation, a Q&A reference guide, and a final reflection report.

**Folder Structure:**
```text
docs/
├── Final_Reflection_Report.pdf
├── Interview_Presentation.pdf
├── Project_Progress_Screenshots.pdf
└── Team_Presentation_Guidebook.pdf
```

**File Details:**

- **`Interview_Presentation.pdf`**: A stakeholder-facing slide deck used to communicate the project findings in an interview or business context. Covers:
  - The problem background — why wildfire survival prediction matters for communities
  - The dataset and competition context (WiDS Datathon 2026 on Kaggle)
  - Our modeling strategy: why Survival Analysis was chosen over classification
  - Physics-informed feature engineering (Danger Index, Trajectory Alignment, etc.)
  - Model architecture: the ensemble of Random Survival Forest + Gradient Boosting Survival Analysis
  - Validation methodology and results (0.9737 C-index)
  - Actionable takeaways and real-world implications for emergency management

- **`Team_Presentation_Guidebook.pdf`**: A reference guide prepared to support Q&A sessions during presentations. Includes:
  - A glossary of all technical terms used in the project (e.g., C-index, Survival Analysis, Kaplan-Meier, Brier Score)
  - A plain-language "race car analogy" that explains hyperparameters as tunable dials
  - Deep dives into the physics-based features and why each was engineered
  - Explanations of why standard deep learning and gradient boosters were ruled out for this small dataset

- **`Final_Reflection_Report.pdf`**: A comprehensive team retrospective summarizing the full project experience. Includes:
  - What each team member learned about data science and machine learning
  - Challenges encountered during the competition (e.g., small dataset, class imbalance, right-censored survival data)
  - How the project boosted each member's confidence in professional data science
  - Long-term takeaways about the role of survival analysis in real-world domains
  - Reflections on teamwork, workflow, and process

- **`Project_Progress_Screenshots.pdf`**: A structured document created as part of the course submission requirements. It contains annotated screenshots documenting the full project journey, including WiDS Datathon registration proofs, Kaggle competition setup and data exploration, public leaderboard submission history, and final private leaderboard score.

### 2. Extracted Visuals (`extracted_visuals/`)

This directory stores all visual assets for the project — including competition registration proofs, Kaggle dataset screenshots, public and private leaderboard score captures, and all technical diagrams and EDA charts generated during model development.

**Folder Structure:**
```text
extracted_visuals/
│
├── Kaggle_registration_Screenshots/          # Kaggle competition evidence & leaderboard results
│   ├── Kaggle_CLI_Download.png               # Kaggle CLI command used to download the dataset
│   ├── Kaggle_Data_Description.png           # Competition data column descriptions
│   ├── Kaggle_Dataset_metaData.png           # Full metadata view of the competition dataset
│   ├── Kaggle_Dataset_sample.png             # Sample rows from the dataset
│   ├── Kaggle_Dataset_test.png               # Test dataset view (95 fire trajectories)
│   ├── Kaggle_Dataset_train.png              # Training dataset view (221 fire trajectories)
│   ├── Kaggle_Overview.png                   # Competition overview page on Kaggle
│   ├── Private Score.png                     # Final private leaderboard score after competition close
│   ├── Public_Scores_Ranking.png              # Final public leaderboard score after competition close
│   ├── Submissions.png                       # Public leaderboard submission history
│   └── Team Registration .png               # Kaggle team registration confirmation
│
├── WIDS_registration_screenshots/            # WiDS Datathon official registration proofs
│   ├── Email_Registered_WIDS_Datathon_2026.png     # Email confirmation of successful registration
│   ├── End_of_registration_WIDS_Datathon.png       # Final screen shown after completing registration
│   └── WIDS_Datathon_Global_Challenge_2026_Registration.png  # WiDS registration form/page
│
├── WiDS-Global-Datathon-2026.jpg             # Official WiDS 2026 Datathon banner (used in README)
├── actionable_predictions.png               # Final visualization of the model's probability predictions
├── eda_kinematic_validation.png             # EDA plots: Kaplan-Meier, closing speed vs. outcome, trajectory alignment
├── model_committee.png                      # Ensemble model architecture diagram (RSF + GBSA)
├── permutation_feature_importance.png       # Permutation importance chart across all features
├── repo_setup.png                           # Repository and environment setup screenshot
├── technical_cv_leakage.png                 # Diagram showing how stratified K-fold prevents data leakage
├── technical_feature_engineering.png        # Diagram explaining the physics feature engineering pipeline
├── technical_hpo.png                        # Bayesian hyperparameter optimization (Optuna) results
└── technical_monotonic.png                  # Illustration of monotonic probability constraints
```

**File Descriptions:**
- **Kaggle Registration Screenshots**: Visuals documenting the competition setup, rules, datasets, and final placements. Includes `Kaggle_Overview.png`, `Team Registration .png`, `Kaggle_CLI_Download.png`, `Kaggle_Data_Description.png`, `Kaggle_Dataset_metaData.png`, `Kaggle_Dataset_sample.png`, `Kaggle_Dataset_train.png`, `Kaggle_Dataset_test.png`, `Submissions.png`, `Public Score.png`, and `Private Score.png`.
- **WiDS Registration Screenshots**: Evidence of official participation, including `WIDS_Datathon_Global_Challenge_2026_Registration.png`, `End_of_registration_WIDS_Datathon.png`, and `Email_Registered_WIDS_Datathon_2026.png`.
- **Technical Diagrams & EDA Plots**: Project workflow diagrams and visualizations created to explain our methodology, including `eda_kinematic_validation.png`, `model_committee.png`, `permutation_feature_importance.png`, `actionable_predictions.png`, `repo_setup.png`, `technical_feature_engineering.png`, `technical_cv_leakage.png`, `technical_hpo.png`, and `technical_monotonic.png`.

### 3. Notebooks (`notebooks/`)

This directory contains the primary Jupyter Notebook for the WiDS 2026 Global Datathon project. The notebook is a complete, end-to-end machine learning pipeline — from raw data loading through final submission generation — and is designed to be fully reproducible.

**Folder Structure:**
```text
notebooks/
└── wids_2026_wildfire_survival_notebook.ipynb    # Full end-to-end modeling pipeline
```

**Notebook Overview:**
The problem is framed as a **Survival Analysis** task due to the time-to-event nature of the target and the ~69% right-censored data. With only **N=221 training samples** and ~31% positive events, the methodology strictly avoids overfitting by using Physics-Informed Feature Engineering, right-sized tree-based survival models, and rigorous Stratified K-Fold cross-validation (evaluating Concordance Index and Integrated Brier Score). The notebook runs through 13 structured sections:

1. **Environment Setup & Pipeline Initialization**: Imports libraries, sets global constants (e.g., `RANDOM_SEED = 42`, `HORIZONS`), and verifies the environment.
2. **Physics-Informed Feature Engineering**: Calls `load_and_engineer_features()` from `src/feature_engineering.py` to create custom features (`danger_index`, `time_to_impact_est`, `growth_impact`, `dist_accel_impact`).
3. **Exploratory Data Analysis (EDA) & Kinematic Validation**: Validates the physics features mathematically using Kaplan-Meier curves, closing speed vs. outcome boxplots, and trajectory alignment scatter plots.
4. **Custom Evaluation Mechanics (Brier Score)**: Defines evaluation functions (`safe_surv_val`, `surv_funcs_to_cum_probs`, `compute_wbs`, `compute_hybrid`) to handle edge cases safely.
5. **Stratified Cross-Validation Strategy**: Configures a 5-fold Stratified K-Fold to maintain the 31% positive event rate across folds to ensure realistic CV metrics.
6. **Modeling: Tree-Based Survival Architectures**: Evaluates Random Survival Forest (RSF), Gradient Boosting Survival Analysis (GBSA), and Extra Survival Trees (EST). (EST is ultimately excluded).
7. **Global Interpretability & Feature Importance**: Uses Permutation Feature Importance to confirm the predictive dominance of the engineered physics features.
8. **Hyperparameter Optimization via Optuna**: Conducts 50 Bayesian optimization trials per model (RSF and GBSA) using Optuna to tune parameters like `n_estimators`, `max_depth`, and `min_samples_leaf`.
9. **Final Blending & Submission**: Trains final models on the full training set, predicts for the 95 test fires, enforces monotonic constraints (P(12h) ≤ P(24h) ≤ P(48h) ≤ P(72h)), and saves `outputs/final_submission.csv`.
10. **Final Results Summary**: Reports mean C-index (0.9737), mean WBS (< 0.03), and mean Hybrid Score (> 0.96).
11. **Discussion & Methodological Reflections**: Justifies the methodological decisions and acknowledges limitations.
12. **Conclusion**: Summarizes the contribution and modeling results.
13. **References**: Lists all academic and technical references.

### 4. Source Code (`src/`)

This directory contains all Python source modules used by the Jupyter notebook. By separating data processing logic from the notebook, the code is cleaner, more maintainable, and fully testable.

**Folder Structure:**
```text
src/
└── feature_engineering.py    # Physics-based feature extraction and data loading pipeline
```

**File Details: `feature_engineering.py`**
The core ETL pipeline script imported into the Jupyter notebook at Step 2 via `from feature_engineering import load_and_engineer_features`.

**Function: `load_and_engineer_features(train_path, test_path)`**
- **Purpose:** Loads the raw Kaggle CSVs, prepares the survival analysis target (a structured array with `event` and `time_to_hit_hours` for `scikit-survival`), drops targets from features, handles missing values using the training set median to prevent leakage, and engineers the four key physics-based features.
- **Physics Feature Engineering:**
  - `danger_index`: `closing_speed_m_per_h × alignment_cos`
  - `time_to_impact_est`: `dist_min_ci_0_5h ÷ closing_speed_m_per_h`
  - `growth_impact`: `area_growth_rate_ha_per_h × alignment_cos`
  - `dist_accel_impact`: `dist_accel_m_per_h2 ÷ closing_speed_m_per_h`
- **Design Decisions:** No external data was used per competition rules. The script ensures the exact same transformations are applied identically to train and test sets.

### 5. Outputs (`outputs/`)

This directory stores all generated predictions and model output files produced by the Jupyter notebook pipeline.

**Folder Structure:**
```text
outputs/
└── final_submission.csv    # Kaggle submission file with predicted probabilities for 95 test fires
```

**File Details: `final_submission.csv`**
The final Kaggle submission file generated by blending the predictions of the RSF and GBSA models in `notebooks/wids_2026_wildfire_survival_notebook.ipynb`.
- **Format:** Contains 5 columns: `event_id`, `prob_12h`, `prob_24h`, `prob_48h`, and `prob_72h`.
- **Key Properties:** Contains 95 test records. The probabilities are strictly monotonic for every fire (e.g., `prob_12h ≤ prob_24h ≤ prob_48h ≤ prob_72h`), enforced post-prediction. They are derived from survival function estimates extracted at the 4 time horizons, not raw classification scores.

---

*WiDS Global Datathon 2026 · Houston City College · ITAI 2377 · Spring 2026*
