"""
=============================================================================
COMP2024 - Artificial Intelligence Methods
IDS Feature Selection & Hyperparameter Optimisation using Metaheuristics
=============================================================================
Module  : evaluation.py
Purpose : Shared evaluation utilities and baseline Random Forest model.
          All metaheuristic modules import fitness / metric helpers from here.
Authors : Group XXX
=============================================================================
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — fixes Windows CMD tkinter error
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)


# ---------------------------------------------------------------------------
# Default hyperparameter space shared by all metaheuristics
# ---------------------------------------------------------------------------

# Each entry: (min, max, is_integer)
HP_SPACE = {
    "n_estimators":  (10,  300, True),
    "max_depth":     (2,   30,  True),
    "min_samples_split": (2, 20, True),
    "min_samples_leaf":  (1, 10, True),
    "max_features":  (0.1, 1.0, False),   # fraction of features
}

HP_KEYS = list(HP_SPACE.keys())


def decode_hyperparams(hp_vector: np.ndarray) -> dict:
    """
    Decode a continuous hyperparameter vector (values in [0,1]) into the
    actual hyperparameter dictionary used by RandomForestClassifier.

    Parameters
    ----------
    hp_vector : array of shape (len(HP_KEYS),)
        Normalised values in [0, 1].

    Returns
    -------
    dict of hyperparameter name → value
    """
    params = {}
    for i, key in enumerate(HP_KEYS):
        lo, hi, is_int = HP_SPACE[key]
        val = lo + hp_vector[i] * (hi - lo)
        params[key] = int(round(val)) if is_int else float(val)
    return params


# ---------------------------------------------------------------------------
# Core fitness / evaluation function
# ---------------------------------------------------------------------------

def evaluate_solution(
    feature_mask: np.ndarray,
    hp_vector: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float = 0.9,
    beta: float = 0.1,
) -> float:
    """
    Fitness function shared by all metaheuristics.

    Fitness = alpha * F1_score  -  beta * (n_selected / n_total)

    A HIGHER fitness is BETTER.  The second term penalises selecting too
    many features, encouraging sparse solutions.

    Parameters
    ----------
    feature_mask : binary array of shape (n_features,)
    hp_vector    : continuous array of shape (len(HP_KEYS),), values in [0,1]
    X_train, y_train : training data
    X_val,   y_val   : validation / test data
    alpha : weight for F1 score (accuracy component)
    beta  : weight for feature penalty

    Returns
    -------
    fitness : float (higher = better)
    """
    # If no features selected, return worst possible score
    selected_idx = np.where(feature_mask > 0.5)[0]
    if len(selected_idx) == 0:
        return 0.0

    X_tr = X_train[:, selected_idx]
    X_vl = X_val[:, selected_idx]

    params = decode_hyperparams(hp_vector)

    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=min(params["max_features"], 1.0),
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_vl)

    f1  = f1_score(y_val, y_pred, zero_division=0)
    fpr_penalty = feature_mask.mean()           # fraction of features used

    fitness = alpha * f1 - beta * fpr_penalty
    return fitness


# ---------------------------------------------------------------------------
# Full metrics computation (used after optimisation is complete)
# ---------------------------------------------------------------------------

def compute_metrics(
    clf: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_mask: np.ndarray,
    method_name: str = "Model",
    n_total_features: int = None,
) -> dict:
    """
    Evaluate a trained classifier and return a metrics dictionary.
    """
    selected_idx = np.where(feature_mask > 0.5)[0]
    X_sel = X_test[:, selected_idx]

    y_pred = clf.predict(X_sel)
    y_prob = clf.predict_proba(X_sel)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # False Positive Rate
    tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # True Positive Rate / Detection Rate

    n_features_used  = int(feature_mask.sum())
    n_total = n_total_features or len(feature_mask)

    metrics = {
        "Method":           method_name,
        "Accuracy":         round(acc,  4),
        "Precision":        round(prec, 4),
        "Recall (TPR)":     round(rec,  4),
        "F1-Score":         round(f1,   4),
        "FPR":              round(fpr,  4),
        "N_Features":       n_features_used,
        "N_Total_Features": n_total,
        "Feature_Ratio":    round(n_features_used / n_total, 4),
    }
    return metrics


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_mask: np.ndarray,
    hp_vector: np.ndarray,
    method_name: str,
    n_total_features: int,
) -> tuple[dict, RandomForestClassifier, float]:
    """
    Final train → evaluate pass.  Returns (metrics_dict, fitted_clf, runtime).
    """
    params = decode_hyperparams(hp_vector)
    selected_idx = np.where(feature_mask > 0.5)[0]

    X_tr = X_train[:, selected_idx]
    X_te = X_test[:, selected_idx]

    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=min(params["max_features"], 1.0),
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_train)
    runtime = time.time() - t0

    metrics = compute_metrics(
        clf, X_test, y_test,
        feature_mask, method_name, n_total_features
    )
    metrics["Runtime_s"] = round(runtime, 2)
    return metrics, clf, runtime


# ---------------------------------------------------------------------------
# Baseline model (all features, default hyperparameters)
# ---------------------------------------------------------------------------

def run_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
) -> tuple[dict, RandomForestClassifier]:
    """
    Train a Random Forest with ALL features and DEFAULT sklearn hyperparameters.
    This is the benchmark against which all metaheuristic methods are compared.

    Default sklearn RF:
        n_estimators = 100
        max_depth    = None  (expand until all leaves are pure)
        max_features = 'sqrt'
        etc.

    Returns
    -------
    metrics : dict
    clf     : fitted RandomForestClassifier
    """
    print("\n" + "=" * 60)
    print("  BASELINE: Random Forest (all features, default HPs)")
    print("=" * 60)

    t0 = time.time()
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    runtime = time.time() - t0

    # All features selected → mask of ones
    full_mask = np.ones(X_train.shape[1])

    metrics = compute_metrics(
        clf, X_test, y_test,
        full_mask, "Baseline RF", X_train.shape[1]
    )
    metrics["Runtime_s"] = round(runtime, 2)

    _print_metrics(metrics)

    # Feature importance plot
    from data_preprocessing import feature_importance_plot
    feature_importance_plot(clf.feature_importances_, feature_names)

    return metrics, clf


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_metrics(m: dict) -> None:
    print(f"\n  Method      : {m['Method']}")
    print(f"  Accuracy    : {m['Accuracy']:.4f}")
    print(f"  Precision   : {m['Precision']:.4f}")
    print(f"  Recall(TPR) : {m['Recall (TPR)']:.4f}")
    print(f"  F1-Score    : {m['F1-Score']:.4f}")
    print(f"  FPR         : {m['FPR']:.4f}")
    print(f"  Features    : {m['N_Features']} / {m['N_Total_Features']}")
    print(f"  Runtime     : {m.get('Runtime_s', '?')} s")


# ---------------------------------------------------------------------------
# Visualisation – comparison charts
# ---------------------------------------------------------------------------

def plot_convergence_curves(
    curves: dict[str, list[float]],
    title: str = "Convergence Curves",
    save_path: str = "plots/convergence_curves.png",
) -> None:
    """
    Plot fitness vs iteration for multiple algorithms.

    Parameters
    ----------
    curves : {method_name: [fitness_per_iteration]}
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = ["steelblue", "tomato", "seagreen", "darkorange", "purple"]
    for i, (name, curve) in enumerate(curves.items()):
        ax.plot(curve, label=name, color=colours[i % len(colours)], linewidth=2)
    ax.set_xlabel("Iteration / Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot] Saved → {save_path}")


def plot_metrics_comparison(
    all_metrics: list[dict],
    save_path: str = "plots/metrics_comparison.png",
) -> None:
    """
    Grouped bar chart comparing Accuracy, F1, FPR, and Feature Ratio
    across all methods.
    """
    df = pd.DataFrame(all_metrics)
    metrics_to_plot = ["Accuracy", "Precision", "Recall (TPR)", "F1-Score", "FPR"]

    x = np.arange(len(df))
    width = 0.15
    fig, ax = plt.subplots(figsize=(13, 6))
    colours = ["steelblue", "tomato", "seagreen", "darkorange", "purple"]

    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + i * width, df[metric], width, label=metric,
               color=colours[i], edgecolor="black", alpha=0.85)

    ax.set_xticks(x + width * (len(metrics_to_plot) - 1) / 2)
    ax.set_xticklabels(df["Method"], rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Performance Comparison Across Methods")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot] Saved → {save_path}")


def plot_feature_reduction(
    all_metrics: list[dict],
    save_path: str = "plots/feature_reduction.png",
) -> None:
    """
    Bar chart of number of features selected vs total per method.
    """
    df = pd.DataFrame(all_metrics)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    total_bar = ax.bar(x, df["N_Total_Features"], label="Total Features",
                       color="lightsteelblue", edgecolor="black")
    sel_bar   = ax.bar(x, df["N_Features"], label="Selected Features",
                       color="steelblue", edgecolor="black")

    for i, row in df.iterrows():
        ax.text(i, row["N_Features"] + 0.5,
                f'{int(row["N_Features"])}', ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(df["Method"], rotation=15, ha="right")
    ax.set_ylabel("Number of Features")
    ax.set_title("Feature Selection: Selected vs Total")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot] Saved → {save_path}")


def plot_tradeoff_scatter(
    all_metrics: list[dict],
    save_path: str = "plots/tradeoff_scatter.png",
) -> None:
    """
    Scatter plot: F1-Score vs FPR, bubble size proportional to feature ratio.
    Visualises the trade-off between detection quality, false positives, and
    feature count.
    """
    df = pd.DataFrame(all_metrics)
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = ["steelblue", "tomato", "seagreen", "darkorange", "purple"]

    for i, row in df.iterrows():
        size = 300 + 1000 * row["Feature_Ratio"]   # bigger → more features
        ax.scatter(row["FPR"], row["F1-Score"],
                   s=size, color=colours[i % len(colours)],
                   alpha=0.7, edgecolors="black", linewidth=1.2)
        ax.annotate(row["Method"],
                    xy=(row["FPR"], row["F1-Score"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)

    ax.set_xlabel("False Positive Rate (FPR)  ← lower is better")
    ax.set_ylabel("F1-Score  ← higher is better")
    ax.set_title("Trade-Off: F1-Score vs FPR\n(bubble size ∝ feature ratio)")
    ax.grid(True, linestyle="--", alpha=0.4)
    # Ideal corner annotation
    ax.annotate("Ideal", xy=(0, 1), xytext=(0.02, 0.92),
                fontsize=9, color="green",
                arrowprops=dict(arrowstyle="->", color="green"))
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot] Saved → {save_path}")


def save_results_csv(
    all_metrics: list[dict],
    save_path: str = "results/all_metrics.csv",
) -> None:
    """Save all metrics to CSV for easy inclusion in the paper."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(all_metrics)
    df.to_csv(save_path, index=False)
    print(f"\n  [Results] Saved → {save_path}")
    print(df.to_string(index=False))
