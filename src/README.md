# Source Code

This directory contains all Python source modules used by the Jupyter notebook. By separating data processing logic from the notebook, the code is cleaner, more maintainable, and fully testable.

---

## Folder Structure

```text
src/
└── feature_engineering.py    # Physics-based feature extraction and data loading pipeline
```

---

## File Details

### `feature_engineering.py`

The core ETL (Extract, Transform, Load) pipeline for the project. This script is imported directly into the Jupyter notebook at Step 2 via:

```python
from feature_engineering import load_and_engineer_features
```

---

#### Function: `load_and_engineer_features(train_path, test_path)`

**Purpose:** Loads the raw Kaggle CSVs, prepares the survival analysis target, and engineers four physics-based features.

**Parameters:**
| Parameter | Default | Description |
|---|---|---|
| `train_path` | `'data/raw/train.csv'` | Path to the training dataset |
| `test_path` | `'data/raw/test.csv'` | Path to the test dataset |

**Returns:**
| Variable | Type | Description |
|---|---|---|
| `X_train` | `pd.DataFrame` | Feature matrix for training (221 rows × 38 features) |
| `y_train_struct` | `np.ndarray` (structured) | Survival target array with `('event', bool)` and `('time_to_hit_hours', float)` fields, required by `scikit-survival` |
| `X_test` | `pd.DataFrame` | Feature matrix for test (95 rows × 38 features) |
| `test_ids` | `pd.Series` | Fire event IDs for the test set (used in submission file) |
| `y_train` | `pd.DataFrame` | Raw target DataFrame (used for stratification and EDA) |

---

#### Processing Steps

1. **Load data** — Reads `train.csv` and `test.csv` using `pd.read_csv()`

2. **Prepare survival target** — Extracts `event` (bool) and `time_to_hit_hours` (float) and converts to a `numpy` structured array for `scikit-survival`:
   ```python
   dtype=[('event', '?'), ('time_to_hit_hours', '<f8')]
   ```

3. **Drop identifiers/targets** — Removes `event_id`, `event`, and `time_to_hit_hours` from feature matrices

4. **Handle missing values** — Fills NaN values with the training set median (median is robust to outliers). Importantly, test set NaNs are filled with the **training** median to prevent data leakage:
   ```python
   X_test = X_test.fillna(X_train.median())
   ```

5. **Physics Feature Engineering** — Four new columns are computed from existing sensor columns:

| Feature | Formula | Physical Meaning |
|---|---|---|
| `danger_index` | `closing_speed_m_per_h × alignment_cos` | High when a fast fire is also headed directly at the zone — the primary composite threat score |
| `time_to_impact_est` | `dist_min_ci_0_5h ÷ closing_speed_m_per_h` | Naive physics-based ETA for when the fire front will arrive. Safe division: speed is clipped to a minimum of 0.1 m/h |
| `growth_impact` | `area_growth_rate_ha_per_h × alignment_cos` | How much the fire's lateral expansion, weighted by its trajectory, threatens the zone |
| `dist_accel_impact` | `dist_accel_m_per_h2 ÷ closing_speed_m_per_h` | Relative acceleration — does the fire front speed up relative to its current closing speed? |

**Why these features?** In ablation testing, these physics-engineered features consistently outperformed raw sensor columns by providing direct kinematic signals that the tree-based survival models can split on effectively. The features encode domain knowledge about fire behavior that the raw columns do not express directly.

---

#### Design Decisions
- **No external data or API calls** — the competition rules prohibit external data; all features are derived purely from the provided columns
- **Separate module** — keeping ETL logic out of the notebook keeps the notebook readable and allows the feature pipeline to be independently tested or reused
- **Test/train consistency** — the exact same transformations are applied to both train and test in a single loop, ensuring no inconsistency between what the model was trained on and what it predicts on
