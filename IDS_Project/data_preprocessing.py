"""
=============================================================================
COMP2024 - Artificial Intelligence Methods
IDS Feature Selection & Hyperparameter Optimisation using Metaheuristics
=============================================================================
Module  : data_preprocessing.py
Purpose : Download, load, and preprocess the NSL-KDD dataset for binary
          classification (Normal vs Attack).
Dataset : NSL-KDD  
          https://www.unb.ca/cic/datasets/nsl.html
Authors : Group XXX
=============================================================================
"""

import os
import io
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- fixes Windows CMD tkinter error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# NSL-KDD Column Definitions
# ---------------------------------------------------------------------------
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

# URLs for NSL-KDD dataset files (plain .txt, no login required)
TRAIN_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
    "KDDTrain%2B.txt"
)
TEST_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
    "KDDTest%2B.txt"
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest_path: str) -> None:
    """Download a file from *url* to *dest_path* with a progress indicator."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  Downloading {os.path.basename(dest_path)} … ", end="", flush=True)
    urllib.request.urlretrieve(url, dest_path)
    size_kb = os.path.getsize(dest_path) / 1024
    print(f"done  ({size_kb:.0f} KB)")


def download_nsl_kdd(data_dir: str = DATA_DIR) -> tuple[str, str]:
    """
    Ensure the NSL-KDD train / test files exist locally.

    Returns
    -------
    train_path, test_path : str
    """
    train_path = os.path.join(data_dir, "KDDTrain+.txt")
    test_path  = os.path.join(data_dir, "KDDTest+.txt")

    if not os.path.exists(train_path):
        _download_file(TRAIN_URL, train_path)
    else:
        print(f"  Train file already exists: {train_path}")

    if not os.path.exists(test_path):
        _download_file(TEST_URL, test_path)
    else:
        print(f"  Test  file already exists: {test_path}")

    return train_path, test_path


# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------

def load_nsl_kdd(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load NSL-KDD CSV files into DataFrames.

    The raw files have no header; columns are assigned from NSL_KDD_COLUMNS.
    The 'difficulty' column (last) is dropped as it is not a feature.
    """
    train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    test_df  = pd.read_csv(test_path,  header=None, names=NSL_KDD_COLUMNS)

    # Drop difficulty score – not used in modelling
    train_df.drop(columns=["difficulty"], inplace=True)
    test_df.drop(columns=["difficulty"],  inplace=True)

    print(f"\n[Data] Train shape : {train_df.shape}")
    print(f"[Data] Test  shape : {test_df.shape}")
    return train_df, test_df


def binarise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert multi-class attack labels to binary:
        'normal' -> 0
        anything else -> 1  (attack)
    """
    df = df.copy()
    df["label"] = df["label"].apply(lambda x: 0 if x.strip() == "normal" else 1)
    return df


def encode_categoricals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: list[str] = ("protocol_type", "service", "flag"),
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    One-Hot-Encode categorical columns.

    The training-set category vocabulary is used as the reference.
    Any category present in the test set but absent from training is
    silently ignored (its OHE columns will all be 0).  This avoids
    data leakage and handles unseen categories safely.

    Returns
    -------
    train_enc, test_enc : DataFrames with OHE columns replacing originals
    ohe_columns         : list of newly created OHE column names
    """
    train_enc = train_df.copy()
    test_enc  = test_df.copy()

    # Get dummies on training set (creates the reference column set)
    train_enc = pd.get_dummies(train_enc, columns=list(cat_cols), dtype=float)

    # Get dummies on test set, then align columns to training set
    test_enc = pd.get_dummies(test_enc, columns=list(cat_cols), dtype=float)

    # Align: add missing columns as 0, drop extra columns
    missing_cols = set(train_enc.columns) - set(test_enc.columns)
    for c in missing_cols:
        test_enc[c] = 0.0
    test_enc = test_enc[train_enc.columns]   # enforce same column order

    ohe_columns = [c for c in train_enc.columns
                   if any(c.startswith(cat + "_") for cat in cat_cols)]

    print(f"  [OHE] Categorical columns expanded: {list(cat_cols)}")
    print(f"  [OHE] New feature count: {len(train_enc.columns) - 1}")
    print(f"  [OHE] One-hot columns added: {len(ohe_columns)}")

    return train_enc, test_enc, ohe_columns


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

def class_distribution(df: pd.DataFrame, label_col: str = "label") -> None:
    """Print and plot class distribution."""
    counts = df[label_col].value_counts()
    labels_map = {0: "Normal", 1: "Attack"}
    print("\n[EDA] Class distribution:")
    for k, v in counts.items():
        name = labels_map.get(k, str(k))
        print(f"  {name:8s}: {v:6d}  ({100*v/len(df):.1f}%)")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([labels_map.get(k, str(k)) for k in counts.index],
           counts.values, color=["steelblue", "tomato"], edgecolor="black")
    ax.set_title("Class Distribution (NSL-KDD, Binary)")
    ax.set_ylabel("Sample Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 100, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/class_distribution.png", dpi=150)
    plt.close()
    print("  [Plot] Saved -> plots/class_distribution.png")


def feature_correlation_heatmap(
    X: pd.DataFrame,
    top_n: int = 20,
    label_col: str = "label",
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
    test_size: float = 0.2,
    random_state: int = 42,
    eda: bool = True,
) -> dict:
    """
    Full preprocessing pipeline:
        1. Download (if needed) and load NSL-KDD
        2. Binarise labels
        3. Encode categoricals
        4. Train / test split (if combined) or use provided split
        5. Scale features

    Returns
    -------
    dict with keys:
        X_train, X_test      : np.ndarray  (scaled)
        y_train, y_test      : np.ndarray
        feature_names        : list[str]
        scaler               : StandardScaler
        encoders             : dict
        n_features           : int
    """
    print("=" * 60)
    print("  NSL-KDD Preprocessing Pipeline")
    print("=" * 60)

    # 1. Download / load
    train_path, test_path = download_nsl_kdd(data_dir)
    train_df, test_df = load_nsl_kdd(train_path, test_path)

    # 2. Binary labels
    train_df = binarise_labels(train_df)
    test_df  = binarise_labels(test_df)

    if eda:
        class_distribution(train_df)

    # 3. One-Hot Encode categoricals
    cat_cols = ["protocol_type", "service", "flag"]
    train_df, test_df, ohe_columns = encode_categoricals(train_df, test_df, cat_cols)

    # 4. Split features / labels
    feature_names = [c for c in train_df.columns if c != "label"]
    X_train_raw = train_df[feature_names].values.astype(np.float64)
    y_train     = train_df["label"].values
    X_test_raw  = test_df[feature_names].values.astype(np.float64)
    y_test      = test_df["label"].values

    if eda:
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        feature_correlation_heatmap(full_df, top_n=20)

    # 5. Scale
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
        "ohe_columns":  ohe_columns,
        "n_features":   n_features,
    }


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = preprocess(eda=True)
    print("\nSample feature names:", result["feature_names"][:5])
