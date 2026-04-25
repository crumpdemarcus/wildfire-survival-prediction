# Outputs

This directory stores all generated predictions and model output files produced by the Jupyter notebook pipeline.

---

## Folder Structure

```text
outputs/
└── final_submission.csv    # Kaggle submission file with predicted probabilities for 95 test fires
```

---

## File Details

### `final_submission.csv`

The final Kaggle submission file generated at the end of the modeling pipeline in `notebooks/wids_2026_wildfire_survival_notebook.ipynb`.

**Format:**

| Column | Type | Description |
|---|---|---|
| `event_id` | Integer | Unique identifier for each fire trajectory in the test set |
| `prob_12h` | Float (0–1) | Predicted cumulative probability the fire reaches the zone within 12 hours |
| `prob_24h` | Float (0–1) | Predicted cumulative probability the fire reaches the zone within 24 hours |
| `prob_48h` | Float (0–1) | Predicted cumulative probability the fire reaches the zone within 48 hours |
| `prob_72h` | Float (0–1) | Predicted cumulative probability the fire reaches the zone within 72 hours |

**Key properties:**
- **Monotonic**: For every fire, `prob_12h ≤ prob_24h ≤ prob_48h ≤ prob_72h` — enforced post-prediction via `np.maximum.accumulate()`
- **95 test records**: One row per test fire trajectory (fire event IDs in the Kaggle test set)
- **Survival analysis-based**: Probabilities are derived from survival function estimates, not raw classification scores

**Sample rows:**

```
event_id,       prob_12h,  prob_24h,  prob_48h,  prob_72h
10662602,       0.0149,    0.0312,    0.0367,    0.0807
13353600,       0.5973,    0.9038,    0.9265,    0.9506
35311039,       0.9188,    0.9378,    0.9433,    0.9475
```

**How it was generated:**
1. The Optuna-tuned Random Survival Forest (RSF) and Gradient Boosting Survival Analysis (GBSA) models were each trained on the full training set (N=221).
2. Survival functions were predicted for each of the 95 test fires.
3. Cumulative probabilities at each of the four time horizons were extracted from the survival functions using the `safe_surv_val()` utility with ε-clipping to prevent edge cases.
4. Monotonic constraints were enforced.
5. The RSF and GBSA predictions were blended via weighted averaging based on their cross-validation performance.
6. The final output was written to this CSV file and submitted to Kaggle.
