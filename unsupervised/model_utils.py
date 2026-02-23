"""Shared clustering, profiling, and visualization utilities for unsupervised model notebooks."""

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score, calinski_harabasz_score,
)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_clustering_data(data_path, features_path, cutoff_date='2019-01-01'):
    """Load preprocessed data, select clustering features, and scale.

    Parameters
    ----------
    data_path : str
        Path to the preprocessed pickle (e.g. milestone_ii_preprocessed_complete.pkl).
    features_path : str
        Path to selected_features pickle.
    cutoff_date : str
        Exclude observations on or after this date (default '2019-01-01').

    Returns
    -------
    df_cluster : DataFrame
        Filtered DataFrame with date/country columns intact.
    cluster_features : list[str]
        Feature names used for clustering (excludes country dummies).
    X_scaled : ndarray
        StandardScaler-transformed feature matrix.
    scaler : StandardScaler
        Fitted scaler instance (for inverse_transform of centers).
    """
    df = pd.read_pickle(data_path)
    if 'date' not in df.columns:
        df = df.reset_index()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] < cutoff_date].copy().sort_values(['date', 'country']).reset_index(drop=True)

    with open(features_path, 'rb') as f:
        selected_features = pickle.load(f)

    cluster_features = [feat for feat in selected_features if not feat.startswith('country_')]
    available = [feat for feat in cluster_features if feat in df.columns]
    missing = [feat for feat in cluster_features if feat not in df.columns]
    if missing:
        print(f"Warning — missing features (excluded): {missing}")
    cluster_features = available

    X = df[cluster_features].copy()
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    df_cluster = df.loc[X.index].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Dataset shape: {df_cluster.shape}")
    print(f"Date range: {df_cluster['date'].min():%Y-%m} to {df_cluster['date'].max():%Y-%m}")
    print(f"Countries ({df_cluster['country'].nunique()}): {sorted(df_cluster['country'].unique())}")
    print(f"\nClustering features ({len(cluster_features)}):")
    for i, feat in enumerate(cluster_features, 1):
        print(f"  {i:2d}. {feat}")
    print(f"\nFeature matrix: {X_scaled.shape[0]} observations × {X_scaled.shape[1]} features")
    print(f"NaN rows dropped: {(~valid_mask).sum()}")

    return df_cluster, cluster_features, X_scaled, scaler


# ── Internal Validation ──────────────────────────────────────────────────────

def print_cluster_summary(X_scaled, labels, n_clusters, model_name, *, inertia=None):
    """Print internal validation metrics and cluster sizes.

    Parameters
    ----------
    X_scaled : ndarray
        Scaled feature matrix.
    labels : array-like
        Cluster/regime labels for each observation.
    n_clusters : int
        Number of clusters.
    model_name : str
        Display name (e.g. 'K-Means', 'HMM').
    inertia : float, optional
        Within-cluster sum of squares (K-Means only).
    """
    labels = np.asarray(labels)
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    print(f"{model_name} fitted with {n_clusters} clusters/states")
    print(f"\nInternal Validation Metrics:")
    print(f"  Silhouette Score:      {sil:.4f}")
    print(f"  Calinski-Harabasz:     {ch:.4f}")
    print(f"  Davies-Bouldin:        {db:.4f}")
    if inertia is not None:
        print(f"  Inertia:               {inertia:.2f}")

    print(f"\nCluster Sizes:")
    total = len(labels)
    for c in range(n_clusters):
        count = (labels == c).sum()
        pct = count / total * 100
        print(f"  Regime {c}: {count:>5d} observations ({pct:.1f}%)")


# ── Silhouette Analysis ─────────────────────────────────────────────────────

def plot_silhouette_analysis(X_scaled, labels, n_clusters):
    """Plot silhouette diagram with per-cluster mean lines and summary stats.

    Parameters
    ----------
    X_scaled : ndarray
        Scaled feature matrix.
    labels : array-like
        Cluster/regime labels.
    n_clusters : int
        Number of clusters.
    """
    labels = np.asarray(labels)
    sample_sil_values = silhouette_samples(X_scaled, labels)
    mean_sil = silhouette_score(X_scaled, labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10

    for i in range(n_clusters):
        ith_values = sample_sil_values[labels == i]
        ith_values.sort()
        size_i = ith_values.shape[0]
        y_upper = y_lower + size_i

        color = plt.cm.tab10(i / max(n_clusters, 1))
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_i, f'Regime {i}',
                fontweight='bold', fontsize=9)

        cluster_mean = ith_values.mean()
        ax.plot([cluster_mean, cluster_mean], [y_lower, y_upper],
                color='black', linestyle=':', linewidth=1.5, alpha=0.9)
        ax.text(cluster_mean + 0.008, y_upper - 0.15 * size_i,
                f'{cluster_mean:.3f}', fontsize=8, color='black',
                fontweight='bold', va='center')

        y_lower = y_upper + 10

    ax.axvline(x=mean_sil, color='red', linestyle='--', linewidth=2,
               label=f'Overall mean: {mean_sil:.3f}')
    ax.set_title('Silhouette Plot by Cluster', fontsize=14, fontweight='bold')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Observations (sorted within cluster)')
    ax.set_yticks([])
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

    print("Per-Cluster Silhouette Statistics:")
    for i in range(n_clusters):
        vals = sample_sil_values[labels == i]
        print(f"  Regime {i}: mean={vals.mean():.3f}, min={vals.min():.3f}, "
              f"max={vals.max():.3f}, % negative={(vals < 0).mean()*100:.1f}%")


# ── Cluster Profiling ────────────────────────────────────────────────────────

def plot_center_heatmaps(centers_original, centers_scaled, feature_names, n_clusters):
    """Plot side-by-side heatmaps of cluster centers (original + standardized).

    Parameters
    ----------
    centers_original : ndarray, shape (n_clusters, n_features)
        Cluster centers in original (inverse-transformed) scale.
    centers_scaled : ndarray, shape (n_clusters, n_features)
        Cluster centers in standardized scale.
    feature_names : list[str]
        Feature names for axis labels.
    n_clusters : int
        Number of clusters.
    """
    regime_labels = [f'Regime {i}' for i in range(n_clusters)]

    centers_df = pd.DataFrame(centers_original, columns=feature_names, index=regime_labels)
    centers_scaled_df = pd.DataFrame(centers_scaled, columns=feature_names, index=regime_labels)

    fig, axes = plt.subplots(1, 2, figsize=(20, max(4, n_clusters * 1.5)))

    sns.heatmap(centers_df, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
                linewidths=0.5, ax=axes[0])
    axes[0].set_title('Cluster Centers (Original Scale)', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('')

    sns.heatmap(centers_scaled_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                linewidths=0.5, ax=axes[1], vmin=-2, vmax=2)
    axes[1].set_title('Cluster Centers (Standardized — Deviation from Mean)',
                      fontweight='bold', fontsize=12)
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.show()

    print("Regime Profiles (Original Scale):")
    print(centers_df.T.to_string())


def plot_feature_boxplots(df_cluster, feature_names, n_clusters, *, regime_col='regime'):
    """Plot per-feature box plots grouped by regime.

    Parameters
    ----------
    df_cluster : DataFrame
        Data with regime labels and feature columns.
    feature_names : list[str]
        Features to plot.
    n_clusters : int
        Number of clusters/regimes.
    regime_col : str
        Column name for regime labels (default 'regime').
    """
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(feature_names):
        data_to_plot = [df_cluster.loc[df_cluster[regime_col] == r, feat].values
                        for r in range(n_clusters)]
        bp = axes[i].boxplot(data_to_plot, patch_artist=True, widths=0.6)
        for j, patch in enumerate(bp['boxes']):
            patch.set_facecolor(plt.cm.tab10(j / max(n_clusters, 1)))
            patch.set_alpha(0.7)
        axes[i].set_title(feat, fontweight='bold', fontsize=10)
        axes[i].set_xticklabels([f'R{r}' for r in range(n_clusters)])
        axes[i].grid(True, alpha=0.3, axis='y')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Feature Distributions by Regime', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ── Temporal Analysis ────────────────────────────────────────────────────────

def plot_regime_composition_over_time(df_cluster, n_clusters, *, regime_col='regime'):
    """Plot stacked area chart of regime proportions over time.

    Parameters
    ----------
    df_cluster : DataFrame
        Data with 'date' and regime columns.
    n_clusters : int
        Number of clusters/regimes.
    regime_col : str
        Column name for regime labels (default 'regime').
    """
    temporal = (df_cluster.groupby('date')[regime_col]
                .value_counts().unstack(fill_value=0))
    temporal = temporal.div(temporal.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = [plt.cm.tab10(i) for i in range(n_clusters)]
    temporal.plot.area(ax=ax, stacked=True, alpha=0.8,
                       color=[colors[c] for c in temporal.columns])
    ax.set_title('Economic Regime Composition Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Proportion of Countries')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles[:len(temporal.columns)],
              [f'Regime {i}' for i in temporal.columns],
              title='Regime', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def plot_country_regime_analysis(df_cluster, n_clusters, *, regime_col='regime'):
    """Plot per-country regime heatmap and stacked proportion bar chart.

    Parameters
    ----------
    df_cluster : DataFrame
        Data with 'date', 'country', and regime columns.
    n_clusters : int
        Number of clusters/regimes.
    regime_col : str
        Column name for regime labels (default 'regime').
    """
    # Per-country regime timeline heatmap
    regime_matrix = df_cluster.pivot_table(index='country', columns='date',
                                            values=regime_col, aggfunc='first')

    fig, ax = plt.subplots(figsize=(18, 5))
    cmap = plt.colormaps['tab10'].resampled(n_clusters)
    im = ax.imshow(regime_matrix.values, aspect='auto', cmap=cmap,
                   interpolation='nearest', vmin=-0.5, vmax=n_clusters - 0.5)

    n_dates = len(regime_matrix.columns)
    tick_step = max(1, n_dates // 10)
    ax.set_xticks(range(0, n_dates, tick_step))
    ax.set_xticklabels([regime_matrix.columns[i].strftime('%Y-%m')
                         for i in range(0, n_dates, tick_step)], rotation=45, ha='right')
    ax.set_yticks(range(len(regime_matrix.index)))
    ax.set_yticklabels(regime_matrix.index)

    cbar = plt.colorbar(im, ax=ax, ticks=range(n_clusters))
    cbar.set_label('Regime')
    ax.set_title('Economic Regime by Country Over Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Per-country regime proportions
    country_regime = (df_cluster.groupby('country')[regime_col]
                      .value_counts(normalize=True).unstack(fill_value=0))
    country_regime = country_regime[sorted(country_regime.columns)]

    fig, ax = plt.subplots(figsize=(10, 5))
    country_regime.plot.barh(stacked=True, ax=ax, cmap='tab10',
                              edgecolor='black', linewidth=0.5)
    ax.set_title('Regime Proportions by Country', fontsize=14, fontweight='bold')
    ax.set_xlabel('Proportion of Time in Each Regime')
    ax.legend(title='Regime', labels=[f'Regime {i}' for i in sorted(country_regime.columns)],
              bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# ── Dimensionality Reduction ────────────────────────────────────────────────

def plot_pca_analysis(X_scaled, labels, feature_names, n_clusters, *, centers_scaled=None):
    """Plot PCA scatter (colored by regime) and loadings bar chart.

    Parameters
    ----------
    X_scaled : ndarray
        Scaled feature matrix.
    labels : array-like
        Cluster/regime labels.
    feature_names : list[str]
        Feature names for loadings plot.
    n_clusters : int
        Number of clusters.
    centers_scaled : ndarray, optional
        Cluster centers in scaled space to project and overlay as markers.

    Returns
    -------
    X_pca : ndarray, shape (n_samples, 2)
        PCA-transformed observations.
    pca : PCA
        Fitted PCA object.
    centers_pca : ndarray or None
        PCA-transformed centers (None if centers_scaled is None).
    """
    labels = np.asarray(labels)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centers_pca = pca.transform(centers_scaled) if centers_scaled is not None else None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                               cmap='tab10', alpha=0.4, s=15)
    if centers_pca is not None:
        axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', marker='X',
                        s=200, edgecolors='white', linewidths=2, zorder=5, label='Centroids')
    axes[0].set_title('PCA Projection — Colored by Regime', fontweight='bold')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    axes[0].legend(loc='best')
    plt.colorbar(scatter, ax=axes[0], label='Regime')

    # PCA loadings
    loadings = pd.DataFrame(pca.components_.T, index=feature_names, columns=['PC1', 'PC2'])
    y_pos = np.arange(len(feature_names))
    axes[1].barh(y_pos + 0.2, loadings['PC1'], height=0.35, label='PC1', alpha=0.8)
    axes[1].barh(y_pos - 0.2, loadings['PC2'], height=0.35, label='PC2', alpha=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(feature_names, fontsize=9)
    axes[1].set_title('PCA Loadings', fontweight='bold')
    axes[1].set_xlabel('Loading')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].axvline(x=0, color='black', linewidth=0.5)

    plt.suptitle('Principal Component Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%, "
          f"Total={sum(pca.explained_variance_ratio_[:2])*100:.1f}%")

    return X_pca, pca, centers_pca


def plot_tsne(X_scaled, labels, n_clusters, *, perplexity=30, random_state=42):
    """Plot t-SNE scatter colored by regime.

    Parameters
    ----------
    X_scaled : ndarray
        Scaled feature matrix.
    labels : array-like
        Cluster/regime labels.
    n_clusters : int
        Number of clusters.
    perplexity : int
        t-SNE perplexity parameter (default 30).
    random_state : int
        Random seed (default 42).

    Returns
    -------
    X_tsne : ndarray, shape (n_samples, 2)
        t-SNE-transformed observations.
    """
    labels = np.asarray(labels)
    norm = Normalize(vmin=0, vmax=n_clusters - 1)
    cmap = plt.cm.tab10

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels,
               cmap=cmap, norm=norm, alpha=0.4, s=15)
    ax.set_title('t-SNE Projection — Colored by Regime', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    legend_handles = [Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=cmap(norm(i)), markersize=8,
                             label=f'Regime {i}') for i in range(n_clusters)]
    ax.legend(handles=legend_handles, loc='best', fontsize=10, frameon=True)
    plt.tight_layout()
    plt.show()

    return X_tsne


# ── Post-hoc Analysis ────────────────────────────────────────────────────────

def plot_recession_overlay(df_cluster, labels, n_clusters, X_pca, pca,
                           *, centers_pca=None):
    """Plot recession rate per regime and PCA scatter colored by recession flag.

    Parameters
    ----------
    df_cluster : DataFrame
        Data with 'oecd_rec' column and regime labels.
    labels : array-like
        Cluster/regime labels.
    n_clusters : int
        Number of clusters.
    X_pca : ndarray, shape (n_samples, 2)
        PCA-projected observations.
    pca : PCA
        Fitted PCA object (for axis labels).
    centers_pca : ndarray, optional
        PCA-projected cluster centers.
    """
    if 'oecd_rec' not in df_cluster.columns:
        print("oecd_rec column not found — skipping recession overlay.")
        return

    labels = np.asarray(labels)
    recession_flag = (df_cluster['oecd_rec'] == 1)
    overall_rec_rate = recession_flag.mean() * 100

    print("Post-hoc Recession Overlap Analysis")
    print("=" * 60)
    print("(Fraction of each regime's observations that fall during known OECD recessions)\n")

    rec_rates = []
    for regime in range(n_clusters):
        mask = labels == regime
        total = mask.sum()
        in_rec = (mask & recession_flag.values).sum()
        pct = in_rec / total * 100 if total > 0 else 0
        rec_rates.append(pct)
        print(f"  Regime {regime}: {in_rec:>4d}/{total} during recession ({pct:.1f}%)")

    print(f"\n  Overall recession rate in dataset: {overall_rec_rate:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: recession rate per regime
    colors = ['#e74c3c' if r > overall_rec_rate else '#2ecc71' for r in rec_rates]
    axes[0].bar(range(n_clusters), rec_rates, color=colors, edgecolor='black', alpha=0.8)
    axes[0].axhline(y=overall_rec_rate, color='black', linestyle='--', alpha=0.7,
                     label=f'Dataset average: {overall_rec_rate:.1f}%')
    axes[0].set_xticks(range(n_clusters))
    axes[0].set_xticklabels([f'Regime {i}' for i in range(n_clusters)])
    axes[0].set_ylabel('% Observations During OECD Recession')
    axes[0].set_title('Recession Rate by Regime', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # PCA scatter colored by recession flag
    scatter_rec = axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                                  c=recession_flag.values.astype(int),
                                  cmap='RdYlGn_r', alpha=0.4, s=15)
    if centers_pca is not None:
        axes[1].scatter(centers_pca[:, 0], centers_pca[:, 1], c='blue', marker='X',
                        s=200, edgecolors='white', linewidths=2, zorder=5,
                        label='Cluster centroids')
    axes[1].set_title('PCA — Colored by OECD Recession Flag', fontweight='bold')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].legend(loc='best')
    plt.colorbar(scatter_rec, ax=axes[1], label='In Recession')

    plt.tight_layout()
    plt.show()


# ── Stability ────────────────────────────────────────────────────────────────

def plot_stability_ari(ari_scores, model_name, n_seeds):
    """Plot ARI histogram and print summary statistics.

    Parameters
    ----------
    ari_scores : list[float]
        Adjusted Rand Index scores from repeated fits.
    model_name : str
        Display name (e.g. 'K-Means', 'HMM').
    n_seeds : int
        Number of random seeds used.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(ari_scores, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=np.mean(ari_scores), color='red', linestyle='--', linewidth=2,
               label=f'Mean ARI: {np.mean(ari_scores):.3f}')
    ax.set_title(f'Cluster Stability (Adjusted Rand Index Across {n_seeds} Random Seeds)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel(f'Adjusted Rand Index vs. Reference (seed=42)')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"Stability Summary:")
    print(f"  Mean ARI: {np.mean(ari_scores):.4f}")
    print(f"  Std ARI:  {np.std(ari_scores):.4f}")
    print(f"  Min ARI:  {np.min(ari_scores):.4f}")
    print(f"  Max ARI:  {np.max(ari_scores):.4f}")

    mean_ari = np.mean(ari_scores)
    if mean_ari > 0.9:
        print("  Clusters are highly stable across initializations.")
    elif mean_ari > 0.7:
        print("  Clusters are moderately stable across initializations.")
    else:
        print("  Clusters show instability — results are sensitive to initialization.")
