# Extracted Visuals

This directory stores all visual assets for the project — including competition registration proofs, Kaggle dataset screenshots, public and private leaderboard score captures, and all technical diagrams and EDA charts generated during model development. These images are used across the project documentation, the `README.md`, and the `Project_Progress_Screenshots.md`.

---

## Folder Structure

```text
extracted_visuals/
│
├── kaggle_registration_screenshots/          # Kaggle competition evidence & leaderboard results
│   ├── Kaggle_CLI_Download.png               # Kaggle CLI command used to download the dataset
│   ├── Kaggle_Data_Description.png           # Competition data column descriptions
│   ├── Kaggle_Dataset_metaData.png           # Full metadata view of the competition dataset
│   ├── Kaggle_Dataset_sample.png             # Sample rows from the dataset
│   ├── Kaggle_Dataset_test.png               # Test dataset view (95 fire trajectories)
│   ├── Kaggle_Dataset_train.png              # Training dataset view (221 fire trajectories)
│   ├── Kaggle_Overview.png                   # Competition overview page on Kaggle
│   ├── Private Score.png                     # Final private leaderboard score after competition close
│   ├── Public_Scores_Ranking.png              # Final public leaderboard score after competition close
│   ├── Submissions_page.png                  # Public leaderboard submission history
│   └── Team_Registration.png                 # Kaggle team registration confirmation
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

---

## File Descriptions

### Kaggle Registration Screenshots (`kaggle_registration_screenshots/`)

| File | Description |
|---|---|
| `Kaggle_Overview.png` | The main WiDS Global Datathon 2026 competition page on Kaggle, showing the competition description, theme ("Predicting Wildfire Impact: From Infrastructure to Equity"), and timeline. |
| `Team_Registration.png` | Confirmation of our team's official registration in the Kaggle competition. Shows team name, members, and competition acceptance. |
| `Kaggle_CLI_Download.png` | Screenshot of the Kaggle CLI command used to programmatically download the `train.csv` and `test.csv` dataset files. |
| `Kaggle_Data_Description.png` | The data dictionary page from Kaggle, describing each feature column (e.g., `closing_speed_m_per_h`, `alignment_cos`, `area_growth_rate_ha_per_h`) and their data types. |
| `Kaggle_Dataset_metaData.png` | Full metadata view of the competition dataset — file names, sizes, and formats. |
| `Kaggle_Dataset_sample.png` | A sample of rows from the dataset shown in Kaggle's data preview, illustrating the structure of the sensor telemetry. |
| `Kaggle_Dataset_train.png` | The training dataset as shown in the Kaggle interface. Contains 221 fire trajectory records, each with sensor readings and a binary event label. |
| `Kaggle_Dataset_test.png` | The test dataset as shown in the Kaggle interface. Contains 95 fire trajectories for which predictions must be submitted. |
| `Submissions_page.png` | The public leaderboard submission history, showing each submission's public score (C-index) and timestamp. |
| `Private Score.png` | The final private leaderboard score revealed after the competition ended, computed on the hidden portion of the test data. |
| `Public_Scores_Ranking.png` | The final public leaderboard score revealed after the competition ended, full ranking. |

---

### WiDS Registration Screenshots (`WIDS_registration_screenshots/`)

| File | Description |
|---|---|
| `WIDS_Datathon_Global_Challenge_2026_Registration.png` | The official WiDS Datathon Global Challenge 2026 registration form, showing the Airtable-based sign-up page used to formally enroll as participants. |
| `End_of_registration_WIDS_Datathon.png` | The confirmation screen shown at the end of the WiDS registration process, indicating successful enrollment. |
| `Email_Registered_WIDS_Datathon_2026.png` | The official email confirmation received from WiDS Worldwide confirming successful registration for the 2026 Datathon. |

---

### Technical Diagrams & EDA Plots

| File | Description |
|---|---|
| `WiDS-Global-Datathon-2026.jpg` | Official WiDS 2026 Datathon banner image. Used as the header image in the main `README.md`. |
| `eda_kinematic_validation.png` | Three-panel EDA chart: (1) Kaplan-Meier survival curve showing probability degradation over time, (2) closing speed boxplot comparing hits vs. censored fires, (3) scatter plot of trajectory alignment vs. time-to-hit. |
| `model_committee.png` | High-level architecture diagram illustrating the ensemble model — how RSF and GBSA predictions are weighted and blended, then subjected to monotonic constraints. |
| `permutation_feature_importance.png` | Bar chart of permutation feature importance scores. Shows which features (especially the physics-engineered ones) most impact the model's C-index when shuffled. |
| `actionable_predictions.png` | Final visualization of predicted cumulative probabilities for the 95 test fires across all four time horizons (12h, 24h, 48h, 72h). |
| `repo_setup.png` | Screenshot of the repository structure and local environment setup. |
| `technical_feature_engineering.png` | Diagram explaining how raw sensor columns are transformed into physics-based features (Danger Index, Trajectory Alignment, Acceleration Metrics). |
| `technical_cv_leakage.png` | Diagram illustrating how Stratified K-Fold cross-validation is configured to preserve class proportions and prevent data leakage across folds. |
| `technical_hpo.png` | Screenshot or diagram showing Optuna's Bayesian hyperparameter optimization process and the best parameter configurations found. |
| `technical_monotonic.png` | Diagram illustrating the monotonic probability constraint: P(12h) ≤ P(24h) ≤ P(48h) ≤ P(72h), enforced post-prediction to ensure physically valid outputs. |
