# Notebooks

This directory contains the primary Jupyter Notebook for the WiDS 2026 Global Datathon project. The notebook is a complete, end-to-end machine learning pipeline — from raw data loading through final submission generation — and is designed to be fully reproducible.

---

## Folder Structure

```text
notebooks/
└── wids_2026_wildfire_survival_notebook.ipynb    # Full end-to-end modeling pipeline
```

---

## Notebook: `wids_2026_wildfire_survival_notebook.ipynb`

### Overview

This notebook presents a **production-grade machine learning pipeline** built for the WiDS 2026 Global Datathon. The challenge: given sensor and environmental telemetry from an active wildfire, predict the **cumulative probability** it reaches a designated community evacuation zone within **12, 24, 48, or 72 hours**.

Because the target represents a **time-to-event** (and ~69% of fires are "censored" — they never hit the zone within the observation window), treating this as a standard binary classification problem discards critical temporal information. Instead, the problem is framed as a **Survival Analysis** task.

With only **N=221 training samples** and ~31% positive events, the methodology strictly avoids overfitting by:
1. **Physics-Informed Feature Engineering** — kinematic geometry (closing speeds, angular alignment, area growth) extracted from raw sensor columns.
2. **Right-Sized Model Architectures** — tree-based survival models designed for small-sample stability, avoiding deep learning and dense boosters.
3. **Rigorous Evaluation** — Stratified K-Fold cross-validation with Concordance Index (C-index) and Integrated Brier Score (WBS) metrics.

**Final validated C-index: 0.9737** (out-of-fold across 5 held-out test sets).

---

### Environment Setup

The notebook uses the following core libraries. Install with `pip install -r requirements.txt`:

| Package | Role |
|---|---|
| `scikit-survival` | Survival models (RSF, GBSA, ExtraSurvivalTrees) and survival metrics |
| `optuna` | Bayesian hyperparameter optimization |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `scikit-learn` | Stratified K-Fold, RobustScaler, LogisticRegression |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualizations |

**Global constants set in Step 1:**
- `RANDOM_SEED = 42` — ensures full reproducibility across all models and splits
- `HORIZONS = [12.0, 24.0, 48.0, 72.0]` — the four prediction time horizons
- `EPS_T = 1e-4` — small epsilon for numerical safety in survival function evaluation

---

### Table of Contents

#### 1. Environment Setup & Pipeline Initialization
Imports all libraries, sets global constants (random seeds, time horizons, directories), and verifies the environment is correctly configured. Outputs `"Setup complete. Environment locked."` when successful.

#### 2. Physics-Informed Feature Engineering
Calls the decoupled `load_and_engineer_features()` function from `src/feature_engineering.py`. This transforms the raw sensor telemetry into four physics-based features:

| Feature | Formula | What It Captures |
|---|---|---|
| `danger_index` | `closing_speed_m_per_h × alignment_cos` | Composite threat score — fast fires heading directly at the zone score highest |
| `time_to_impact_est` | `dist_min_ci_0_5h ÷ closing_speed_m_per_h` | Physics-based estimated time of arrival (ETA) |
| `growth_impact` | `area_growth_rate_ha_per_h × alignment_cos` | How expanding fire area, weighted by trajectory, threatens the zone |
| `dist_accel_impact` | `dist_accel_m_per_h2 ÷ closing_speed_m_per_h` | Relative acceleration of the fire front |

Output confirms: `Training set shape (N=221): (221, 38)`, `Test set shape (N=95): (95, 38)`, `31.2% Hits within 72h`.

#### 3. Exploratory Data Analysis (EDA) & Kinematic Validation
Before any modeling, this section mathematically proves that the engineered physics features carry real predictive signal through three visualizations:

1. **Kaplan-Meier Survival Curve** — Shows how the probability of a fire NOT hitting the zone degrades over time. The curve drops steeply in the first 24 hours, confirming temporal structure in the data.
2. **Closing Speed vs. Event Outcome** (boxplot) — Fires that eventually hit the zone have significantly higher closing speeds than censored fires, validating `closing_speed_m_per_h` as a primary signal.
3. **Trajectory Alignment vs. Time-to-Hit** (scatter) — Fires with high alignment scores (heading directly at the zone) tend to hit earlier, confirming `alignment_cos` as a meaningful feature.

Key EDA insight: "We are relying entirely on these high-signal physics features to inform the survival trees."

#### 4. Custom Evaluation Mechanics (Brier Score)
Defines robust evaluation functions to handle edge cases where survival function evaluation can collapse on small test folds:

- **`safe_surv_val(sf, t)`** — Clips time to the model's observed domain with an ε-buffer to prevent numerical collapse
- **`surv_funcs_to_cum_probs()`** — Converts survival functions to cumulative probability predictions across all 4 horizons; enforces monotonic ordering (P(12h) ≤ P(24h) ≤ P(48h) ≤ P(72h))
- **`get_eval_times()`** — Computes safe Brier Score evaluation times for each fold
- **`compute_wbs()`** — Calculates Weighted Brier Score (WBS) using weights [0.3, 0.4, 0.3] at times [24h, 48h, 72h]
- **`compute_hybrid(ci, wbs)`** — Blends C-index ranking accuracy with calibration quality: `0.3 × C-index + 0.7 × (1 − WBS)`

#### 5. Stratified Cross-Validation Strategy
Implements `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` stratified on the binary event label, ensuring each fold contains approximately the real 31% positive event rate:

```
Fold 1: Val N=45 | Event Hit Rate: 31.1%
Fold 2: Val N=44 | Event Hit Rate: 29.5%
Fold 3: Val N=44 | Event Hit Rate: 31.8%
Fold 4: Val N=44 | Event Hit Rate: 31.8%
Fold 5: Val N=44 | Event Hit Rate: 31.8%
```

Each fold uses `RobustScaler` (fit on train, applied to val) to scale features before model training.

#### 6. Modeling: Tree-Based Survival Architectures
Three tree-based survival models are evaluated in cross-validation:

| Model | Key Configuration | Rationale |
|---|---|---|
| **Random Survival Forest (RSF)** | `n_estimators=300`, `min_samples_leaf=5`, `max_features='sqrt'` | Ensemble of survival trees; robust on small samples; natively handles censoring |
| **Gradient Boosting Survival Analysis (GBSA)** | `n_estimators=250`, `learning_rate=0.05`, `max_depth=3`, `min_samples_leaf=5` | Sequential error correction; capped depth prevents overfitting |
| **Extra Survival Trees (EST)** | `n_estimators=300`, `min_samples_leaf=5`, `max_features='log2'` | Maximally random splits; high-variance, low-bias complement to RSF |

Baseline CV results:
```
Model                               Hybrid Score    Mean WBS
Random Survival Forest              0.9578 (±0.019)   0.0342
Gradient Boosting Survival (GBSA)   0.9675 (±0.015)   0.0221
Extra Survival Trees                0.8150 (±0.103)   0.2105
```

EST is excluded from the final ensemble due to high variance and poor calibration.

#### 7. Global Interpretability & Feature Importance
Computes **Permutation Feature Importance** by shuffling each feature one at a time and measuring the drop in C-index. This confirms that our physics-engineered features (`danger_index`, `time_to_impact_est`, `growth_impact`) consistently rank as the most impactful — validating the physics-informed approach over raw sensor columns.

Results are visualized as a horizontal bar chart (`permutation_feature_importance.png`).

#### 8. Hyperparameter Optimization via Optuna
Runs **50 Bayesian optimization trials per model** (RSF and GBSA separately) using Optuna. Each trial evaluates a different hyperparameter configuration against the hybrid CV metric.

Key hyperparameters tuned:
- **RSF**: `n_estimators`, `min_samples_leaf`, `max_features`, `max_depth`
- **GBSA**: `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`, `subsample`

Optuna prunes underperforming trials early using the `MedianPruner`, reducing total compute time. The best configurations found are then used for the final model training.

#### 9. Final Blending & Submission
Trains the final RSF and GBSA models with Optuna-tuned hyperparameters on the full training set. Generates survival functions for all 95 test fire trajectories, converts them to cumulative probability predictions across the 4 time horizons, and applies:
1. **Monotonic bounding** — ensures P(12h) ≤ P(24h) ≤ P(48h) ≤ P(72h) for every fire
2. **Weighted ensemble blending** — RSF and GBSA predictions averaged with weights derived from cross-validation performance

The final predictions are saved to `outputs/final_submission.csv` in the Kaggle submission format:
```
event_id, prob_12h, prob_24h, prob_48h, prob_72h
```

#### 10. Final Results Summary
Reports the final cross-validated performance metrics across both models:
- **Mean C-index**: 0.9737
- **Mean WBS**: < 0.03
- **Mean Hybrid Score**: > 0.96

The C-index of 0.9737 means that in 97.37% of all pairwise comparisons of two fires, the model correctly identifies which fire will arrive at the community zone first.

#### 11. Discussion & Methodological Reflections
Addresses key methodological decisions:
- Why survival analysis was superior to binary classification for this problem
- Why deep learning was ruled out (N=221 is catastrophically small for neural nets)
- How the physics features were validated before being used in models
- The role of monotonic constraints in producing physically interpretable outputs
- Limitations: external data was prohibited; the model is constrained to kinematic signals only

#### 12. Conclusion
Summarizes the project's contribution: a rigorous, physics-informed survival analysis pipeline that achieves state-of-the-art predictive accuracy on a small, right-censored dataset, with full interpretability and honest cross-validated evaluation.

#### 13. References
Lists all academic and technical references, including:
- scikit-survival documentation and original paper
- Optuna Bayesian optimization paper
- WiDS Datathon 2026 competition rules and data description
- Survival analysis methodology references (Kaplan-Meier, Concordance Index, Brier Score)
