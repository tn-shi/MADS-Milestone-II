# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Economic forecasting research project for predicting recessions using multi-domain economic data across 9 nations (G7: USA, Canada, UK, France, Germany, Italy, Japan + Australia, South Korea) from 1970-2020.

- **Supervised Learning Task**: Multi-domain recession forecasting (predicting recession onset 12 months ahead)
- **Unsupervised Learning Task**: Economic regime identification (clustering economic states)

## Development Environment

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# FRED API key required - set environment variable
# Register at https://fredaccount.stlouisfed.org/login/secure/
set FRED_API_KEY=<your_key>  # Windows
```

## Running the Project

**Pipeline order: Dataset -> EDA -> Preprocessing -> Baseline -> FeatureSelection -> Models -> Evaluation**

### Notebooks

1. **Dataset** (`Milestone_II_Dataset.ipynb`): Data gathering
   - Fetches economic data from FRED API
   - Applies publication lag adjustments and yield curve features
   - Creates forecast target (`pre_recession`) and country one-hot encoding
   - Exports to `data/milestone_ii_dataset.csv` and `.pkl`

2. **EDA** (`Milestone_II_EDA.ipynb`): Exploratory analysis of raw indicators
   - Missing values analysis and target distribution
   - Raw indicator correlations and time series patterns
   - Pre-recession vs normal period comparisons
   - Yield curve inversion analysis

3. **Preprocessing** (`Milestone_II_Preprocessing.ipynb`): All feature engineering
   - Log differences, first differences, amplitude deviations (config-driven)
   - GDP QoQ growth, technical recession, SAHM Rule, rolling statistics
   - Real interest rates, unemployment acceleration
   - Lagged features (1/3/6/12-month lags)
   - Exports to `data/milestone_ii_preprocessed.csv` and `.pkl`
   - Also exports NaN-free variants: `data/milestone_ii_preprocessed_complete.pkl` and `data/milestone_ii_dataset_complete.pkl` (trimmed to earliest NaN-free tail across all countries)

4. **Baseline** (`Milestone_II_Baseline.ipynb`): XGBoost baseline with default hyperparameters
   - Time-series cross-validation with expanding window
   - Per-country classification metrics and feature importance

5. **Feature Selection** (`Milestone_II_FeatureSelection.ipynb`): Three-stage feature reduction
   - Correlation removal (|r| > 0.90 threshold, model-specific tiebreaker: XGBoost gain for incomplete data, LogReg |coef| for complete data)
   - Permutation importance (5-fold CV, keeps features with positive ROC-AUC drop)
   - LASSO L1 (LogisticRegression sweep across C values)
   - Pipeline runs independently on both incomplete and complete datasets
   - Consensus of permutation + LASSO -> `data/selected_features.pkl` (from incomplete data, for XGBoost)
   - Consensus of permutation + LASSO -> `data/selected_features_complete.pkl` (from complete data, for LSTM / LogReg)

6. **Supervised Models** (in `supervised/`):
   - `XGBoost_Model.ipynb` — Two-stage random search + Optuna tuning, threshold optimization
   - `LSTM_Model.ipynb` — Sequence-based LSTM with Optuna tuning (50 trials)
   - `LogisticRegression_Model.ipynb` — Logistic Regression with Optuna tuning (200 trials)

7. **Unsupervised Models** (in `unsupervised/`):
   - `K-Means_model.ipynb` — K-Means clustering (k=4), silhouette analysis, regime profiling, recession overlay
   - `HiddenMarkov_model.ipynb` — Gaussian HMM regime identification, BIC/AIC state selection, transition matrix, stationary distribution, expected durations

8. **Evaluation / Analysis** (in `supervised/` and `unsupervised/`):
   - `supervised/eval_ablation_analysis - XGBoost.ipynb` — Feature group and leave-one-out ablation (measures unique feature contribution vs redundancy)
   - `supervised/eval_sensitivity_analysis - XGBoost.ipynb` — OAT perturbation, partial dependence, threshold/HP/data-noise sensitivity
   - `supervised/eval_failure_analysis - XGBoost.ipynb` — FP/FN patterns, per-country error rates, episode detection rates, lead-time analysis
   - `supervised/eval_feature_importance_analysis - XGBoost.ipynb` — Gain, SHAP, permutation importance comparison; category contributions; stability across folds
   - `unsupervised/sensitivity_analysis_hmm.ipynb` — HMM sensitivity to number of states, covariance type, initialization

   All evaluation notebooks load `data/xgboost_model_config.pkl` for tuned hyperparameters and re-run CV internally.

### Data Flow

```
Dataset notebook                    EDA notebook
   |                                   |
   +---> milestone_ii_dataset.pkl -----+
   |     (raw indicators + yield curve)
   v
Preprocessing notebook
   |
   +---> milestone_ii_preprocessed.pkl -------> Baseline / FeatureSelection
   +---> milestone_ii_preprocessed_complete.pkl --> FeatureSelection + Unsupervised
   +---> milestone_ii_dataset_complete.pkl

FeatureSelection notebook (runs pipeline on both datasets)
   |
   +---> selected_features.pkl -----------------> XGBoost_Model + Eval notebooks
   |     (from incomplete data)
   +---> selected_features_complete.pkl --------> LSTM_Model / LogReg_Model
         (from complete data)                     + XGBoost_Model (complete data section)
                                                  + Unsupervised models

Supervised Model notebooks
   |
   +---> xgboost_model_config.pkl ---------------> All eval_* notebooks
   +---> xgboost_complete_model_config.pkl
   +---> lstm_model_config.pkl
   +---> logistic_regression_model_config.pkl
```

**NaN-free data**: LSTM and Logistic Regression require `_complete` datasets (NaN-free). XGBoost can use either but also re-runs on `_complete` data for cross-model comparison. Unsupervised models use `_complete` data.

## Architecture

### Shared Configuration (`config.py`)

Contains all shared configuration used across notebooks:
- `FeatureOp` type alias
- `SeriesConfig` dataclass for FRED series and feature engineering configuration
- `COUNTRY_CODES` dict (ISO-2 and ISO-3 mappings for 9 countries)
- `SERIES_CONFIG` dict (17 indicator configurations driving the pipeline)
- `LOCAL_EPU_CONFIG` dict (local EPU file paths)
- `build_series_id()` and `build_series_dict()` functions

```python
@dataclass(frozen=True)
class SeriesConfig:
    prefix: str           # Characters before country code in FRED series ID
    suffix: str           # Characters after country code
    use_iso3: bool        # Use 3-letter (True) or 2-letter (False) country codes
    is_global: bool       # If True, series is not country-specific (e.g., VIX)
    agg_method: str|None  # Aggregation method for frequency conversion (e.g., "avg")
    suffix_overrides: dict  # Country-specific suffix exceptions
    iso_overrides: dict     # Country-specific ISO code exceptions
    custom_id: dict         # Country-specific custom series IDs bypassing pattern construction
    feature_ops: list     # Operations: "log_diff", "first_diff", "amplitude_deviation", "rolling_stats"
    lagged: bool          # Add 1/3/6/12-month lagged features
    publication_lag: int  # Months to shift data forward (prevents look-ahead bias)
```

Add new indicators by creating a `SeriesConfig` entry in `SERIES_CONFIG` dict in `config.py`.

### Shared Utility Modules

**`supervised/model_utils.py`** — Evaluation/visualization functions shared by all three supervised model notebooks:
- `compute_fold_metrics()`, `compute_country_metrics()` — Per-fold and per-country classification metrics
- `sweep_thresholds()`, `plot_threshold_curves()` — Decision threshold optimization
- `plot_roc_pr_curves()`, `plot_confusion_matrices()`, `plot_optuna_study()` — Standard visualizations
- `build_model_comparison()` — Naive vs baseline vs tuned model comparison table
- `export_model_config()` — Save model config pickle for downstream eval notebooks

**`unsupervised/model_utils.py`** — Clustering/profiling functions shared by K-Means and HMM notebooks:
- `load_clustering_data()` — Load preprocessed data, select features, scale with StandardScaler
- `print_cluster_summary()` — Silhouette, Calinski-Harabasz, Davies-Bouldin metrics
- `plot_silhouette_analysis()`, `plot_center_heatmaps()`, `plot_feature_boxplots()` — Cluster profiling
- `plot_regime_composition_over_time()`, `plot_country_regime_analysis()` — Temporal/geographic analysis
- `plot_pca_analysis()`, `plot_tsne()` — Dimensionality reduction visualizations
- `plot_recession_overlay()` — Post-hoc comparison of regimes vs OECD recession dates
- `plot_stability_ari()` — Adjusted Rand Index stability across random seeds

### Dataset Pipeline

Sequential processing stages in the Dataset notebook:
1. `build_series_dict()` — Generate FRED series IDs from patterns (via `config.py`)
2. `get_fred_data()` — Fetch and combine into MultiIndex DataFrame (date, country)
3. `engineer_features()` — Apply publication lags and yield curve features
4. `create_forecast_target()` — Generate binary `pre_recession` label (12-month horizon)
5. `one_hot_encode_countries()` — Create country dummy variables

### Preprocessing Pipeline

Sequential processing stages in the Preprocessing notebook:
1. `engineer_complex_features()` — Config-driven FE (log_diff, first_diff, amplitude_deviation), GDP growth, SAHM Rule, rolling stats, real rates, unemployment acceleration
2. `add_lagged_features()` — Create lagged versions for indicators with `lagged=True`
3. Complete data export — Find NaN-free tail, export `_complete` variants

### Model Training Conventions

All three supervised models follow the same pattern:
- **Data cutoff**: `2019-01-01` (excludes 2019 pre-recession labels due to COVID contamination)
- **CV**: `TimeSeriesSplit(n_splits=5)` with expanding window
- **Validation split**: `VAL_FRAC=0.2` from each training fold for early stopping + threshold tuning
- **Class imbalance**: `scale_pos_weight` (XGBoost) or `class_weight='balanced'` (LogReg)
- **Threshold tuning**: Sweep 0.05–0.95 on validation predictions, maximize F1; report metrics on test folds
- **Per-fold scaling**: StandardScaler fit on train, transform test (LSTM and LogReg)
- **Feature importance**: Averaged across CV folds (not from a single model)
- **Exports**: Model config pickle to `data/` (e.g., `xgboost_model_config.pkl`)

### Unsupervised Model Conventions

Both unsupervised models follow the same pattern:
- **Data**: `milestone_ii_preprocessed_complete.pkl` + `selected_features_complete.pkl`
- **Data cutoff**: `2019-01-01` (same as supervised)
- **Feature filtering**: Excludes `country_*` dummies from clustering features
- **Scaling**: StandardScaler on selected features, NaN rows dropped
- **Regime column**: `regime` column added to `df_cluster` for all downstream analysis
- **Validation**: Silhouette, Calinski-Harabasz, Davies-Bouldin internal metrics
- **Post-hoc**: Recession overlay analysis using `oecd_rec` column (not used during fitting)
- **Stability**: ARI scores across multiple random seeds

### Evaluation Notebooks

All four XGBoost evaluation notebooks (`eval_*`) follow the same pattern:
- Load `data/xgboost_model_config.pkl` for tuned hyperparameters, threshold, and feature list
- Load `data/milestone_ii_preprocessed.pkl` for features
- Re-train models across 5 time-series CV folds using the saved config
- Run analyses on test fold predictions
- These are downstream of the XGBoost model notebook and do not modify any data files

### Special Cases

- **Japan/Australia EPU**: FRED data discontinued/unavailable; loaded from local Excel files via `load_local_epu()`
- **Global indicators** (VIX, oil, copper, gold): Same series ID used for all countries
- **Quarterly data** (real_gdp): Upsampled to monthly with forward fill

## Key Data Structures

- **MultiIndex DataFrame**: `(date, country)` index enables group-by-country operations
- **COUNTRY_CODES**: ISO-2 and ISO-3 mappings for 9 countries used in FRED series construction (in `config.py`)
- **SERIES_CONFIG**: Dict of 17 indicator configurations driving the entire pipeline (in `config.py`)
- **Model config pickles**: Dicts with `best_params`, `best_threshold`, `selected_features`, `cv_metrics`, and model-specific fields (e.g., `max_rounds`, `early_stop` for XGBoost)
