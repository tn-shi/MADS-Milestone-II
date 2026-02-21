"""Shared evaluation, visualization, and export utilities for supervised model notebooks."""

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_fold_metrics(tuned_pred_df, n_splits, best_threshold):
    """Compute per-fold metrics using optimal threshold predictions.

    Applies ``y_pred_opt`` column in-place, then computes accuracy, precision,
    recall, F1, ROC-AUC, and PR-AUC for each CV fold.

    Returns a DataFrame with one row per fold.
    """
    tuned_pred_df['y_pred_opt'] = (
        tuned_pred_df['y_proba'] >= best_threshold
    ).astype(int)

    final_results = []
    for fold_idx in range(n_splits):
        fold_data = tuned_pred_df[tuned_pred_df['fold'] == fold_idx + 1]
        roc = roc_auc_score(fold_data['y_true'], fold_data['y_proba'])
        prec_arr, rec_arr, _ = precision_recall_curve(
            fold_data['y_true'], fold_data['y_proba']
        )
        pr = auc(rec_arr, prec_arr)
        f1 = f1_score(fold_data['y_true'], fold_data['y_pred_opt'], zero_division=0)
        prec = precision_score(fold_data['y_true'], fold_data['y_pred_opt'], zero_division=0)
        rec = recall_score(fold_data['y_true'], fold_data['y_pred_opt'], zero_division=0)
        acc = accuracy_score(fold_data['y_true'], fold_data['y_pred_opt'])

        final_results.append({
            'fold': fold_idx + 1, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'roc_auc': roc, 'pr_auc': pr,
        })

    final_df = pd.DataFrame(final_results)

    print('Per-Fold Metrics (tuned HP + optimal threshold)')
    print('=' * 85)
    print(final_df.to_string(index=False, float_format='{:.3f}'.format))
    print('\nMean ± Std:')
    for col in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
        print(f'  {col:12s}: {final_df[col].mean():.3f} ± {final_df[col].std():.3f}')

    return final_df


def compute_country_metrics(tuned_pred_df):
    """Compute per-country classification metrics with optimal threshold.

    Expects ``y_pred_opt`` column to already exist (set by
    :func:`compute_fold_metrics`).  Returns a DataFrame indexed by country.
    """
    country_metrics = []
    for country in sorted(tuned_pred_df['country'].unique()):
        mask = tuned_pred_df['country'] == country
        yt = tuned_pred_df.loc[mask, 'y_true']
        yp = tuned_pred_df.loc[mask, 'y_pred_opt']
        yprob = tuned_pred_df.loc[mask, 'y_proba']

        if yt.nunique() > 1:
            roc_val = roc_auc_score(yt, yprob)
            prec_curve, rec_curve, _ = precision_recall_curve(yt, yprob)
            pr_val = auc(rec_curve, prec_curve)
        else:
            roc_val = np.nan
            pr_val = np.nan

        country_metrics.append({
            'Country': country,
            'Accuracy': accuracy_score(yt, yp),
            'Precision': precision_score(yt, yp, zero_division=0),
            'Recall': recall_score(yt, yp, zero_division=0),
            'F1': f1_score(yt, yp, zero_division=0),
            'ROC-AUC': roc_val,
            'PR-AUC': pr_val,
            'Support (pos)': int(yt.sum()),
            'Total': len(yt),
        })

    country_df = pd.DataFrame(country_metrics).set_index('Country')

    print('Per-Country Classification Metrics (Tuned Model, Aggregated Across All CV Folds)')
    print('=' * 100)
    print(country_df.to_string(float_format='{:.3f}'.format))

    # Overall metrics
    yt_all = tuned_pred_df['y_true']
    yp_all = tuned_pred_df['y_pred_opt']
    yprob_all = tuned_pred_df['y_proba']
    prec_all, rec_all, _ = precision_recall_curve(yt_all, yprob_all)
    print(f'\nOverall:  Accuracy={accuracy_score(yt_all, yp_all):.3f}  '
          f'Precision={precision_score(yt_all, yp_all, zero_division=0):.3f}  '
          f'Recall={recall_score(yt_all, yp_all, zero_division=0):.3f}  '
          f'F1={f1_score(yt_all, yp_all, zero_division=0):.3f}  '
          f'ROC-AUC={roc_auc_score(yt_all, yprob_all):.3f}  '
          f'PR-AUC={auc(rec_all, prec_all):.3f}')

    return country_df


# ── Threshold helpers ─────────────────────────────────────────────────────────

def sweep_thresholds(pred_df, n_splits):
    """Sweep decision thresholds and compute per-fold F1 / precision / recall.

    Also computes naive baselines (predict-all-positive F1, class prevalence).

    Returns ``(thresholds, mean_f1, mean_prec, mean_rec, naive_f1,
    mean_prevalence)`` where the metric arrays are averaged across folds.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_arr = np.zeros((n_splits, len(thresholds)))
    prec_arr = np.zeros((n_splits, len(thresholds)))
    rec_arr = np.zeros((n_splits, len(thresholds)))

    for fold_idx in range(n_splits):
        fold_data = pred_df[pred_df['fold'] == fold_idx + 1]
        for j, thresh in enumerate(thresholds):
            y_pred_t = (fold_data['y_proba'] >= thresh).astype(int)
            f1_arr[fold_idx, j] = f1_score(
                fold_data['y_true'], y_pred_t, zero_division=0)
            prec_arr[fold_idx, j] = precision_score(
                fold_data['y_true'], y_pred_t, zero_division=0)
            rec_arr[fold_idx, j] = recall_score(
                fold_data['y_true'], y_pred_t, zero_division=0)

    # Naive baselines (computed per fold, then averaged)
    all_pos_f1, prevalences = [], []
    for fold_idx in range(n_splits):
        fold_data = pred_df[pred_df['fold'] == fold_idx + 1]
        yt = fold_data['y_true']
        all_pos_f1.append(f1_score(yt, np.ones(len(yt)), zero_division=0))
        prevalences.append(yt.mean())
    naive_f1 = float(np.mean(all_pos_f1))
    mean_prevalence = float(np.mean(prevalences))

    return (thresholds, f1_arr.mean(axis=0), prec_arr.mean(axis=0),
            rec_arr.mean(axis=0), naive_f1, mean_prevalence)


def plot_threshold_curves(thresholds, mean_f1, mean_prec, mean_rec,
                          best_threshold, naive_f1, *, ax=None, title=''):
    """Plot F1 / precision / recall vs decision threshold on *ax*.

    If *ax* is ``None`` a new figure is created and ``plt.show()`` is called.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(thresholds, mean_f1, label='F1 Score', linewidth=2, color='steelblue')
    ax.plot(thresholds, mean_prec, label='Precision', linewidth=1.5,
            color='darkorange', linestyle='--')
    ax.plot(thresholds, mean_rec, label='Recall', linewidth=1.5,
            color='green', linestyle='--')
    ax.axvline(best_threshold, color='red', linestyle=':', linewidth=1.5,
               label=f'Best threshold = {best_threshold:.2f}')
    ax.axhline(naive_f1, color='purple', linestyle='--', linewidth=1, alpha=0.5,
               label=f'Predict-all-positive F1 ({naive_f1:.2f})')
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5,
               label='Default (0.50)')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score')
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title)

    if standalone:
        plt.tight_layout()
        plt.show()


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_roc_pr_curves(tuned_pred_df, n_splits, model_name):
    """Plot per-fold ROC and Precision-Recall curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for fold_idx in range(n_splits):
        fold_data = tuned_pred_df[tuned_pred_df['fold'] == fold_idx + 1]
        yt = fold_data['y_true']
        yp = fold_data['y_proba']

        # ROC curve
        fpr, tpr, _ = roc_curve(yt, yp)
        roc_val = roc_auc_score(yt, yp)
        axes[0].plot(fpr, tpr,
                     label=f'Fold {fold_idx+1} (AUC={roc_val:.3f})', alpha=0.7)

        # PR curve
        prec_arr, rec_arr, _ = precision_recall_curve(yt, yp)
        pr_val = auc(rec_arr, prec_arr)
        axes[1].plot(rec_arr, prec_arr,
                     label=f'Fold {fold_idx+1} (AUC={pr_val:.3f})', alpha=0.7)

    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves by Fold')
    axes[0].legend(fontsize=9)

    positive_rate = tuned_pred_df['y_true'].mean()
    axes[1].axhline(positive_rate, color='k', linestyle='--', alpha=0.3,
                    label=f'Baseline ({positive_rate:.2f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curves by Fold')
    axes[1].legend(fontsize=9)

    plt.suptitle(f'Tuned {model_name}: ROC and PR Curves', fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(tuned_pred_df, best_threshold):
    """Plot side-by-side confusion matrices for default (0.50) and optimal threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm_default = confusion_matrix(tuned_pred_df['y_true'], tuned_pred_df['y_pred'])
    ConfusionMatrixDisplay(cm_default, display_labels=['Normal', 'Pre-Recession']).plot(
        ax=axes[0], cmap='Blues')
    axes[0].set_title('Default Threshold (0.50)')

    cm_optimal = confusion_matrix(tuned_pred_df['y_true'], tuned_pred_df['y_pred_opt'])
    ConfusionMatrixDisplay(cm_optimal, display_labels=['Normal', 'Pre-Recession']).plot(
        ax=axes[1], cmap='Blues')
    axes[1].set_title(f'Optimal Threshold ({best_threshold:.2f})')

    plt.suptitle('Confusion Matrices: Default vs Optimal Threshold', fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_optuna_study(study):
    """Plot Optuna optimization history and hyperparameter importance."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Optimization history
    trial_numbers = [t.number for t in study.trials]
    trial_values = [t.value for t in study.trials]
    best_so_far = np.maximum.accumulate(trial_values)

    axes[0].scatter(trial_numbers, trial_values, alpha=0.3, s=15,
                    c='steelblue', label='Trial PR-AUC')
    axes[0].plot(trial_numbers, best_so_far, color='red', linewidth=2,
                 label='Best so far')
    axes[0].set_xlabel('Trial Number')
    axes[0].set_ylabel('Mean CV PR-AUC')
    axes[0].set_title('Optuna Optimization History')
    axes[0].legend()

    # Hyperparameter importance
    importances = optuna.importance.get_param_importances(study)
    params_sorted = list(importances.keys())
    values_sorted = list(importances.values())

    axes[1].barh(params_sorted[::-1], values_sorted[::-1],
                 color='steelblue', alpha=0.8)
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Optuna: Hyperparameter Importance')

    plt.tight_layout()
    plt.show()


# ── Model comparison ──────────────────────────────────────────────────────────

def build_model_comparison(baseline_df, baseline_pred_df,
                           tuned_df, tuned_pred_df,
                           final_df, best_threshold,
                           naive_f1, mean_prevalence,
                           model_name):
    """Build a comparison table of naive, baseline, and tuned models.

    Prints the table, improvement deltas, and naive-baseline checks.
    Returns the comparison DataFrame.
    """
    naive_pos = {
        'Model': 'Naive: Predict All Positive',
        'ROC-AUC': 0.5,
        'PR-AUC': mean_prevalence,
        'F1': naive_f1,
        'Accuracy': tuned_pred_df['y_true'].mean(),
        'Precision': tuned_pred_df['y_true'].mean(),
        'Recall': 1.0,
    }
    naive_neg = {
        'Model': 'Naive: Predict All Negative',
        'ROC-AUC': 0.5,
        'PR-AUC': mean_prevalence,
        'F1': 0.0,
        'Accuracy': 1 - tuned_pred_df['y_true'].mean(),
        'Precision': 0.0,
        'Recall': 0.0,
    }

    baseline_overall = {
        'Model': 'Baseline (default HP, t=0.50)',
        'ROC-AUC': baseline_df['roc_auc'].mean(),
        'PR-AUC': baseline_df['pr_auc'].mean(),
        'F1': baseline_df['f1'].mean(),
        'Accuracy': accuracy_score(baseline_pred_df['y_true'], baseline_pred_df['y_pred']),
        'Precision': precision_score(baseline_pred_df['y_true'], baseline_pred_df['y_pred'], zero_division=0),
        'Recall': recall_score(baseline_pred_df['y_true'], baseline_pred_df['y_pred'], zero_division=0),
    }

    tuned_default_t = {
        'Model': 'Tuned HP (t=0.50)',
        'ROC-AUC': tuned_df['roc_auc'].mean(),
        'PR-AUC': tuned_df['pr_auc'].mean(),
        'F1': tuned_df['f1'].mean(),
        'Accuracy': accuracy_score(tuned_pred_df['y_true'], tuned_pred_df['y_pred']),
        'Precision': precision_score(tuned_pred_df['y_true'], tuned_pred_df['y_pred'], zero_division=0),
        'Recall': recall_score(tuned_pred_df['y_true'], tuned_pred_df['y_pred'], zero_division=0),
    }

    tuned_opt_label = f'Tuned HP + Threshold (t={best_threshold:.2f})'
    tuned_opt_t = {
        'Model': tuned_opt_label,
        'ROC-AUC': final_df['roc_auc'].mean(),
        'PR-AUC': final_df['pr_auc'].mean(),
        'F1': final_df['f1'].mean(),
        'Accuracy': accuracy_score(tuned_pred_df['y_true'], tuned_pred_df['y_pred_opt']),
        'Precision': precision_score(tuned_pred_df['y_true'], tuned_pred_df['y_pred_opt'], zero_division=0),
        'Recall': recall_score(tuned_pred_df['y_true'], tuned_pred_df['y_pred_opt'], zero_division=0),
    }

    comparison = pd.DataFrame(
        [naive_neg, naive_pos, baseline_overall, tuned_default_t, tuned_opt_t]
    ).set_index('Model')

    print(f'{model_name} Model Comparison')
    print('=' * 100)
    print(comparison.to_string(float_format='{:.3f}'.format))

    # Improvement deltas
    print(f'\nImprovement (Tuned HP + Threshold vs Baseline):')
    for col in ['ROC-AUC', 'PR-AUC', 'F1', 'Precision', 'Recall']:
        delta = comparison.loc[tuned_opt_label, col] - comparison.loc[baseline_overall['Model'], col]
        print(f'  {col:12s}: {delta:+.3f}')

    print('\nBeats naive baselines?')
    final_f1 = final_df['f1'].mean()
    final_roc = final_df['roc_auc'].mean()
    final_pr = final_df['pr_auc'].mean()
    print(f'  F1 ({final_f1:.3f}) > predict-all-positive ({naive_f1:.3f}): {final_f1 > naive_f1}')
    print(f'  ROC-AUC ({final_roc:.3f}) > random (0.500): {final_roc > 0.5}')
    print(f'  PR-AUC ({final_pr:.3f}) > prevalence ({mean_prevalence:.3f}): {final_pr > mean_prevalence}')

    return comparison


# ── Export ────────────────────────────────────────────────────────────────────

def export_model_config(best_params, best_threshold, selected_features,
                        final_df, filepath, **extra_fields):
    """Save model configuration pickle for downstream use.

    Standard fields are always included.  Model-specific extras (e.g.
    ``max_rounds``, ``early_stop``, ``dataset``) are passed as keyword
    arguments.
    """
    model_config = {
        'best_params': best_params,
        'best_threshold': float(best_threshold),
        'selected_features': selected_features,
        'random_state': 42,
        'n_cv_splits': 5,
        'data_cutoff': '2019-01-01',
        'cv_metrics': {
            'roc_auc_mean': float(final_df['roc_auc'].mean()),
            'roc_auc_std': float(final_df['roc_auc'].std()),
            'pr_auc_mean': float(final_df['pr_auc'].mean()),
            'pr_auc_std': float(final_df['pr_auc'].std()),
            'f1_mean': float(final_df['f1'].mean()),
            'f1_std': float(final_df['f1'].std()),
        },
    }
    model_config.update(extra_fields)

    with open(filepath, 'wb') as f:
        pickle.dump(model_config, f)

    print(f'Exported to: {filepath.replace("../", "")}')
    print(f'\nModel Configuration:')
    print(f'  Features:    {len(selected_features)}')
    print(f'  Threshold:   {best_threshold:.2f}')
    print(f'  Random state: 42')
    if 'max_rounds' in extra_fields:
        print(f'  Early stopping: {extra_fields["early_stop"]} rounds '
              f'(max {extra_fields["max_rounds"]})')
    print(f'\n  Hyperparameters:')
    for param, val in sorted(best_params.items()):
        print(f'    {param}: {val:.4f}' if isinstance(val, float) else f'    {param}: {val}')
    print(f'\n  CV Performance:')
    print(f'    ROC-AUC: {final_df["roc_auc"].mean():.3f} ± {final_df["roc_auc"].std():.3f}')
    print(f'    PR-AUC:  {final_df["pr_auc"].mean():.3f} ± {final_df["pr_auc"].std():.3f}')
    print(f'    F1:      {final_df["f1"].mean():.3f} ± {final_df["f1"].std():.3f}')
