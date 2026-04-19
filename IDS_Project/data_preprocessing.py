"""
=============================================================================
COMP2024 - Artificial Intelligence Methods
IDS Feature Selection & Hyperparameter Optimisation using Metaheuristics
=============================================================================
Module  : data_preprocessing.py
Purpose : Load and preprocess the CICIDS2017 dataset for binary
          classification (Normal vs Attack).
Dataset : CICIDS2017
          https://www.unb.ca/cic/datasets/ids-2017.html
Authors : Group XXX
=============================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- fixes Windows CMD tkinter error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "MachineLearningCSV", "MachineLearningCVE")


# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------

def load_cicids2017(data_dir: str = DATA_DIR, sample_frac: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    """
    Load CICIDS2017 CSV files from the given directory, clean col names, 
    handle Inf/NaN, and return a concatenated, sampled DataFrame.
    """
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. Please ensure the CICIDS2017 dataset is extracted there.")

    print(f"  [Data] Found {len(all_files)} CSV files. Loading and concatenating...")
    
    df_list = []
    total_rows = 0
    for file in all_files:
        df = pd.read_csv(file)
        total_rows += len(df)
        df_list.append(df)
        
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"  [Data] Concatenated shape : {full_df.shape} (Total rows: {total_rows})")

    # Clean column names (strip leading/trailing whitespaces)
    full_df.columns = full_df.columns.str.strip()

    # Replace Infinity with NaN and drop rows with NaN
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    initial_len = len(full_df)
    full_df.dropna(inplace=True)
    dropped = initial_len - len(full_df)
    if dropped > 0:
        print(f"  [Data] Dropped {dropped} rows containing Infinity or NaN.")

    # Stratified downsampling
    if sample_frac < 1.0:
        print(f"  [Data] Downsampling dataset by fraction {sample_frac} (stratified by Label)...")
        # Ensure we do a stratified sample
        _, sampled_df = train_test_split(
            full_df, 
            test_size=sample_frac, 
            stratify=full_df["Label"], 
            random_state=random_state
        )
        print(f"  [Data] Downsampled shape : {sampled_df.shape}")
        return sampled_df
    
    return full_df


def binarise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert multi-class attack labels to binary:
        'BENIGN' -> 0
        anything else -> 1  (attack)
    """
    df = df.copy()
    # Unique values like 'BENIGN', 'DDoS', 'PortScan', etc.
    df["Label"] = df["Label"].apply(lambda x: 0 if str(x).strip() == "BENIGN" else 1)
    return df


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Z-score normalise features.  Fitted on training data only to avoid
    data leakage.
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler


# ---------------------------------------------------------------------------
# Exploratory analysis helpers
# ---------------------------------------------------------------------------

def class_distribution(df: pd.DataFrame, label_col: str = "Label") -> None:
    """Print and plot class distribution."""
    counts = df[label_col].value_counts()
    labels_map = {0: "Normal (BENIGN)", 1: "Attack"}
    print("\n[EDA] Class distribution:")
    for k, v in counts.items():
        name = labels_map.get(k, str(k))
        print(f"  {name:15s}: {v:6d}  ({100*v/len(df):.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([labels_map.get(k, str(k)) for k in counts.index],
           counts.values, color=["steelblue", "tomato"], edgecolor="black")
    ax.set_title("Class Distribution (CICIDS2017, Binary)")
    ax.set_ylabel("Sample Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.01*len(df), str(v), ha="center", fontsize=10)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/class_distribution.png", dpi=150)
    plt.close()
    print("  [Plot] Saved -> plots/class_distribution.png")


def feature_correlation_heatmap(
    X: pd.DataFrame,
    top_n: int = 20,
    label_col: str = "Label",
) -> None:
    """
    Plot Pearson correlation of the top-N most label-correlated features.
    """
    # Compute absolute correlation with label, pick top-N features
    corr_with_label = X.corr()[label_col].abs().sort_values(ascending=False)
    top_features = corr_with_label.index[1 : top_n + 1].tolist()

    sub = X[top_features + [label_col]]
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        sub.corr(),
        annot=False,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
        linewidths=0.3,
    )
    ax.set_title(f"Correlation Heatmap – Top {top_n} Features")
    plt.tight_layout()
    plt.savefig("plots/feature_correlation_heatmap.png", dpi=150)
    plt.close()
    print("  [Plot] Saved -> plots/feature_correlation_heatmap.png")


def feature_importance_plot(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
) -> None:
    """Bar chart of top-N feature importances from Random Forest."""
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(top_n), importances[idx], color="steelblue", edgecolor="black")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(
        [feature_names[i] for i in idx], rotation=45, ha="right", fontsize=8
    )
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest Baseline)")
    ax.set_ylabel("Mean Decrease in Impurity")
    plt.tight_layout()
    plt.savefig("plots/feature_importances.png", dpi=150)
    plt.close()
    print("  [Plot] Saved -> plots/feature_importances.png")


# ---------------------------------------------------------------------------
# Master preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    data_dir: str = DATA_DIR,
    sample_frac: float = 0.05,
    test_size: float = 0.2,
    random_state: int = 42,
    eda: bool = True,
) -> dict:
    """
    Full preprocessing pipeline:
        1. Load & clean CICIDS2017 dataset
        2. Binarise labels
        3. Train / test split
        4. Scale features

    Returns
    -------
    dict with keys:
        X_train, X_test      : np.ndarray  (scaled)
        y_train, y_test      : np.ndarray
        feature_names        : list[str]
        scaler               : StandardScaler
        n_features           : int
    """
    print("=" * 60)
    print("  CICIDS2017 Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load, concatenate, clean inf/nans, and downsample
    df = load_cicids2017(data_dir, sample_frac=sample_frac, random_state=random_state)

    # 2. Binary labels
    df = binarise_labels(df)

    if eda:
        class_distribution(df, label_col="Label")

    # 3. Train / Test Split
    feature_names = [c for c in df.columns if c != "Label"]
    X_raw = df[feature_names].values.astype(np.float64)
    y     = df["Label"].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )

    if eda:
        feature_correlation_heatmap(df, top_n=20, label_col="Label")

    # 4. Scale
    X_train, X_test, scaler = scale_features(X_train_raw, X_test_raw)

    n_features = X_train.shape[1]
    print(f"\n[Preprocessing] Complete.")
    print(f"  Features     : {n_features}")
    print(f"  Train samples: {X_train.shape[0]}")
    print(f"  Test  samples: {X_test.shape[0]}")
    print("=" * 60)

    return {
        "X_train":      X_train,
        "X_test":       X_test,
        "y_train":      y_train,
        "y_test":       y_test,
        "feature_names": feature_names,
        "scaler":       scaler,
        "n_features":   n_features,
    }


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = preprocess(eda=True)
    print("\nSample feature names:", result["feature_names"][:5])
