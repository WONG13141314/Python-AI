"""
=============================================================================
COMP2024 - Artificial Intelligence Methods
IDS Feature Selection & Hyperparameter Optimisation using Metaheuristics
=============================================================================
Module  : main.py
Purpose : Entry point – runs the full experimental pipeline:
            1. Data preprocessing & EDA
            2. Baseline Random Forest (all features, default HPs)
            3. Metaheuristic 1: Genetic Algorithm (GA)
            4. Metaheuristic 2: Particle Swarm Optimisation (PSO)
            5. Metaheuristic 3: Simulated Annealing (SA)
            6. Comparative evaluation, plots, and results CSV

Dataset : NSL-KDD (downloaded automatically on first run)
Model   : Random Forest (fixed base classifier)

Usage
-----
    python main.py [--quick]

    --quick  : reduce population/iterations for a fast test run (<5 min)
               omit this flag for the full experiment (~30-60 min)

Authors : Group XXX
=============================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Local module imports
# ---------------------------------------------------------------------------
from data_preprocessing import preprocess
from evaluation import (
    run_baseline,
    plot_convergence_curves,
    plot_metrics_comparison,
    plot_feature_reduction,
    plot_tradeoff_scatter,
    save_results_csv,
    _print_metrics,
)
from metaheuristic_ga  import run_ga
from metaheuristic_pso import run_pso
from metaheuristic_sa  import run_sa


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="COMP2024 IDS Metaheuristics Experiment"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with reduced iterations for quick testing"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Global random seed"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # Hyper-config: full vs quick mode
    # -----------------------------------------------------------------------
    if args.quick:
        print("\n[MODE] Quick run – reduced iterations for fast testing.\n")
        GA_PARAMS  = dict(pop_size=15, n_generations=10)
        PSO_PARAMS = dict(n_particles=15, n_iterations=10)
        SA_PARAMS  = dict(max_iter=100, T0=1.0, cooling_rate=0.97)
    else:
        print("\n[MODE] Full experiment.\n")
        GA_PARAMS  = dict(pop_size=30, n_generations=40)
        PSO_PARAMS = dict(n_particles=30, n_iterations=40)
        SA_PARAMS  = dict(max_iter=500, T0=1.0, cooling_rate=0.97)

    # -----------------------------------------------------------------------
    # 1.  Data preprocessing
    # -----------------------------------------------------------------------
    data = preprocess(eda=True)
    X_train_full = data["X_train"]
    X_test       = data["X_test"]
    y_train_full = data["y_train"]
    y_test       = data["y_test"]
    feature_names = data["feature_names"]
    n_features    = data["n_features"]

    # Create a validation split from training data (used by metaheuristics)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=args.seed,
        stratify=y_train_full,
    )
    print(f"\n[Split] Train={X_train.shape[0]} | Val={X_val.shape[0]} | Test={X_test.shape[0]}")

    # -----------------------------------------------------------------------
    # 2.  Baseline model
    # -----------------------------------------------------------------------
    baseline_metrics, baseline_clf = run_baseline(
        X_train_full, y_train_full,
        X_test, y_test,
        feature_names,
    )

    all_metrics        = [baseline_metrics]
    convergence_curves = {}        # populated by each metaheuristic

    # -----------------------------------------------------------------------
    # 3.  Genetic Algorithm
    # -----------------------------------------------------------------------
    ga_result = run_ga(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        n_features=n_features,
        feature_names=feature_names,
        seed=args.seed,
        **GA_PARAMS,
    )
    all_metrics.append(ga_result["metrics"])
    convergence_curves["GA"] = ga_result["convergence_curve"]

    # -----------------------------------------------------------------------
    # 4.  Particle Swarm Optimisation
    # -----------------------------------------------------------------------
    pso_result = run_pso(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        n_features=n_features,
        feature_names=feature_names,
        seed=args.seed,
        **PSO_PARAMS,
    )
    all_metrics.append(pso_result["metrics"])
    convergence_curves["PSO"] = pso_result["convergence_curve"]

    # -----------------------------------------------------------------------
    # 5.  Simulated Annealing
    # -----------------------------------------------------------------------
    sa_result = run_sa(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        n_features=n_features,
        feature_names=feature_names,
        seed=args.seed,
        **SA_PARAMS,
    )
    all_metrics.append(sa_result["metrics"])
    convergence_curves["SA"] = sa_result["convergence_curve"]

    # -----------------------------------------------------------------------
    # 6.  Comparative visualisation & results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for m in all_metrics:
        _print_metrics(m)
        print()

    # Plots
    os.makedirs("plots",   exist_ok=True)
    os.makedirs("results", exist_ok=True)

    plot_convergence_curves(convergence_curves)
    plot_metrics_comparison(all_metrics)
    plot_feature_reduction(all_metrics)
    plot_tradeoff_scatter(all_metrics)

    # Save CSV
    save_results_csv(all_metrics)

    # Save feature masks as JSON (useful for reproducibility)
    feature_masks = {
        "GA":  ga_result["best_feature_mask"].tolist(),
        "PSO": pso_result["best_feature_mask"].tolist(),
        "SA":  sa_result["best_feature_mask"].tolist(),
    }
    with open("results/best_feature_masks.json", "w") as f:
        json.dump(feature_masks, f, indent=2)
    print("  [Results] Saved → results/best_feature_masks.json")

    # Save selected feature names per method
    for method, mask in feature_masks.items():
        selected = [feature_names[i] for i, v in enumerate(mask) if v > 0.5]
        print(f"\n  {method} selected features ({len(selected)}): {selected}")

    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPLETE")
    print("  Plots  → ./plots/")
    print("  Results→ ./results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
