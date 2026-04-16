"""
=============================================================================
COMP2024 - Artificial Intelligence Methods
IDS Feature Selection & Hyperparameter Optimisation using Metaheuristics
=============================================================================
Module  : metaheuristic_nsga2.py
Purpose : Non-dominated Sorting Genetic Algorithm II (NSGA-II) for
          multi-objective feature selection and hyperparameter optimisation
          of a Random Forest IDS model.

Algorithm overview
------------------
NSGA-II is a multi-objective evolutionary algorithm that maintains a
population of solutions and evolves them toward the Pareto-optimal front.

Unlike single-objective methods (GA, PSO, SA), NSGA-II does NOT combine
objectives into a single fitness scalar.  Instead, it simultaneously
optimises two conflicting objectives:

    Objective 1 (minimise): -F1_score       (maximise detection quality)
    Objective 2 (minimise):  Feature_ratio  (minimise model complexity)

With an optional penalty on FPR embedded in Objective 1 to encourage
low false-alarm solutions.

Key mechanisms:
    - Non-dominated sorting: rank solutions into Pareto fronts (F1, F2, ...)
    - Crowding distance: within a front, prefer diverse solutions
    - Tournament selection: compare by (1) front rank, (2) crowding distance
    - Standard GA operators: crossover, mutation (same as metaheuristic_ga.py)

The result is a set of Pareto-optimal solutions spanning the trade-off
between detection quality and feature count.

Constraint handling
-------------------
    Hard constraint: at least one feature must be selected.
    Enforced after mutation: if all bits are 0, one random bit is set to 1.

References
----------
Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T. (2002). A fast and
    elitist multiobjective genetic algorithm: NSGA-II. IEEE TEC, 6(2).
Authors : Group XXX
=============================================================================
"""

import numpy as np
import time
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from evaluation import (
    decode_hyperparams, HP_KEYS, compute_metrics,
    train_and_evaluate, _print_metrics,
)


# ---------------------------------------------------------------------------
# Multi-objective fitness evaluation
# ---------------------------------------------------------------------------

def _evaluate_objectives(
    feature_mask: np.ndarray,
    hp_vector: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    rf_n_jobs: int = -1,
) -> tuple[float, float]:
    """
    Evaluate a solution on TWO objectives (both to be MINIMISED):

        obj1 = -F1_score + 0.1 * FPR   (lower -> better detection with fewer false alarms)
        obj2 =  feature_ratio           (lower -> fewer features selected)

    Returns
    -------
    (obj1, obj2) : tuple of floats – both are to be MINIMISED.
    """
    selected_idx = np.where(feature_mask > 0.5)[0]
    if len(selected_idx) == 0:
        return (1.0, 1.0)   # worst possible on both objectives

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
        n_jobs=rf_n_jobs,
    )
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_vl)

    f1  = f1_score(y_val, y_pred, zero_division=0)

    # FPR from confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    obj1 = -f1 + 0.1 * fpr    # minimise (~= maximise F1, minimise FPR)
    obj2 = feature_mask.mean() # minimise (fewer features)

    return (obj1, obj2)


# ---------------------------------------------------------------------------
# NSGA-II core mechanisms
# ---------------------------------------------------------------------------

def _non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """
    Fast non-dominated sorting (Deb et al. 2002).

    Parameters
    ----------
    objectives : (N, M) array where N = population size, M = number of objectives.
                 All objectives are to be MINIMISED.

    Returns
    -------
    fronts : list of lists; fronts[0] = Pareto-optimal indices, etc.
    """
    n = len(objectives)
    domination_count = np.zeros(n, dtype=int)      # how many solutions dominate i
    dominated_set    = [[] for _ in range(n)]       # which solutions does i dominate
    fronts           = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            # Check if i dominates j or vice versa
            diff = objectives[i] - objectives[j]
            if np.all(diff <= 0) and np.any(diff < 0):
                # i dominates j
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif np.all(diff >= 0) and np.any(diff > 0):
                # j dominates i
                dominated_set[j].append(i)
                domination_count[i] += 1

    # First front: non-dominated solutions
    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)

    # Build subsequent fronts
    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    # Remove trailing empty front
    if not fronts[-1]:
        fronts.pop()

    return fronts


def _crowding_distance(objectives: np.ndarray, front_indices: list[int]) -> np.ndarray:
    """
    Compute crowding distance for solutions in a single front.

    Solutions at the boundary of the objective space receive infinite distance.
    """
    n = len(front_indices)
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    obj_subset = objectives[front_indices]

    n_obj = obj_subset.shape[1]
    for m in range(n_obj):
        sorted_idx = np.argsort(obj_subset[:, m])
        distances[sorted_idx[0]]  = np.inf
        distances[sorted_idx[-1]] = np.inf

        obj_range = obj_subset[sorted_idx[-1], m] - obj_subset[sorted_idx[0], m]
        if obj_range == 0:
            continue

        for i in range(1, n - 1):
            distances[sorted_idx[i]] += (
                (obj_subset[sorted_idx[i + 1], m] - obj_subset[sorted_idx[i - 1], m])
                / obj_range
            )

    return distances


# ---------------------------------------------------------------------------
# Chromosome helpers (same structure as GA)
# ---------------------------------------------------------------------------

def _random_chromosome(n_features: int, rng: np.random.Generator) -> np.ndarray:
    """Random chromosome: binary feature mask + continuous HP vector."""
    n_hp = len(HP_KEYS)
    feature = rng.integers(0, 2, size=n_features).astype(float)
    hp      = rng.uniform(0, 1, size=n_hp)
    return np.concatenate([feature, hp])


def _split_chromosome(chrom: np.ndarray, n_features: int):
    """Split chromosome into (feature_mask, hp_vector)."""
    return chrom[:n_features], chrom[n_features:]


def _crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    n_features: int,
    crossover_rate: float,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover (features) + BLX-alpha (HPs)."""
    if rng.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    mask = rng.integers(0, 2, size=n_features)
    f1   = np.where(mask == 0, parent1[:n_features], parent2[:n_features])
    f2   = np.where(mask == 0, parent2[:n_features], parent1[:n_features])

    p_hp1 = parent1[n_features:]
    p_hp2 = parent2[n_features:]
    lo = np.minimum(p_hp1, p_hp2) - alpha * np.abs(p_hp1 - p_hp2)
    hi = np.maximum(p_hp1, p_hp2) + alpha * np.abs(p_hp1 - p_hp2)
    hp1 = np.clip(rng.uniform(lo, hi), 0, 1)
    hp2 = np.clip(rng.uniform(lo, hi), 0, 1)

    child1 = np.concatenate([f1, hp1])
    child2 = np.concatenate([f2, hp2])
    return child1, child2


def _mutate(
    chrom: np.ndarray,
    n_features: int,
    mutation_rate_feature: float,
    mutation_rate_hp: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bit-flip (features) + Gaussian perturbation (HPs)."""
    chrom = chrom.copy()

    for i in range(n_features):
        if rng.random() < mutation_rate_feature:
            chrom[i] = 1.0 - chrom[i]

    # Constraint: at least one feature
    if chrom[:n_features].sum() == 0:
        chrom[rng.integers(n_features)] = 1.0

    n_hp = len(HP_KEYS)
    for j in range(n_features, n_features + n_hp):
        if rng.random() < mutation_rate_hp:
            chrom[j] = np.clip(chrom[j] + rng.normal(0, 0.1), 0, 1)

    return chrom


def _tournament_select_nsga2(
    front_ranks: np.ndarray,
    crowding: np.ndarray,
    pop_size: int,
    rng: np.random.Generator,
    k: int = 3,
) -> int:
    """
    Binary tournament based on (1) front rank (lower better),
    then (2) crowding distance (higher better) as tiebreaker.
    """
    contestants = rng.choice(pop_size, size=k, replace=False)
    best = contestants[0]
    for c in contestants[1:]:
        if front_ranks[c] < front_ranks[best]:
            best = c
        elif front_ranks[c] == front_ranks[best] and crowding[c] > crowding[best]:
            best = c
    return best


# ---------------------------------------------------------------------------
# Pareto front visualisation
# ---------------------------------------------------------------------------

def plot_pareto_front(
    pareto_objectives: np.ndarray,
    all_objectives: np.ndarray = None,
    save_path: str = "plots/pareto_front.png",
) -> None:
    """
    Scatter plot of the Pareto front (F1-Score vs Feature Ratio).

    Parameters
    ----------
    pareto_objectives : (K, 2) array of Pareto-optimal objectives (minimised form).
    all_objectives    : (N, 2) array of all final population objectives (optional).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))

    # Convert from minimisation form back to intuitive axes
    # obj1 = -F1 + 0.1*FPR  -> approximate F1 ~= -obj1  (ignore FPR for display)
    # obj2 = feature_ratio

    if all_objectives is not None:
        ax.scatter(-all_objectives[:, 0], all_objectives[:, 1],
                   color="lightgray", alpha=0.5, s=30, label="Dominated solutions")

    ax.scatter(-pareto_objectives[:, 0], pareto_objectives[:, 1],
               color="tomato", s=80, edgecolors="black", linewidth=1.2,
               zorder=5, label="Pareto front")

    # Connect Pareto front points with a line (sorted by obj1)
    sorted_idx = np.argsort(-pareto_objectives[:, 0])
    ax.plot(-pareto_objectives[sorted_idx, 0],
            pareto_objectives[sorted_idx, 1],
            color="tomato", linewidth=1.5, alpha=0.6, linestyle="--")

    ax.set_xlabel("F1-Score (approx.)  <- higher is better")
    ax.set_ylabel("Feature Ratio  <- lower is better")
    ax.set_title("NSGA-II Pareto Front: Detection Quality vs Feature Count")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Annotate ideal corner
    ax.annotate("Ideal", xy=(1.0, 0.0), xytext=(0.92, 0.08),
                fontsize=9, color="green",
                arrowprops=dict(arrowstyle="->", color="green"))

    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot] Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main NSGA-II routine
# ---------------------------------------------------------------------------

def run_nsga2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_features: int,
    feature_names: list,
    X_train_full: np.ndarray = None,
    y_train_full: np.ndarray = None,
    pop_size: int             = 30,
    n_generations: int        = 40,
    crossover_rate: float     = 0.85,
    mutation_rate_feature: float = 0.02,
    mutation_rate_hp: float   = 0.15,
    tournament_k: int         = 3,
    blx_alpha: float          = 0.3,
    seed: int                 = 42,
) -> dict:
    """
    Run NSGA-II for multi-objective feature selection + HP tuning.

    Two objectives (both minimised):
        1. -F1 + 0.1*FPR  (maximise detection, minimise false positives)
        2.  Feature_ratio  (minimise feature count)

    Returns the "knee-point" solution from the final Pareto front as the
    recommended solution, plus the full Pareto front for visualisation.

    Parameters
    ----------
    pop_size, n_generations, crossover_rate, mutation_rate_feature,
    mutation_rate_hp, tournament_k, blx_alpha : same as GA.

    Returns
    -------
    dict with keys: metrics, convergence_curve, best_feature_mask,
                    best_hp_vector, runtime_s, pareto_front, pareto_objectives
    """
    print("\n" + "=" * 60)
    print("  METAHEURISTIC 4: NSGA-II (Multi-Objective)")
    print(f"  Population={pop_size}, Generations={n_generations}")
    print(f"  Objectives: (1) maximise F1 - 0.1 x FPR, (2) minimise feature ratio")
    print("=" * 60)

    X_fit = X_train_full if X_train_full is not None else X_train
    y_fit = y_train_full if y_train_full is not None else y_train

    rng     = np.random.default_rng(seed)
    t_start = time.time()
    n_obj   = 2

    # ---- Initialise population ----
    population = np.array([
        _random_chromosome(n_features, rng) for _ in range(pop_size)
    ])

    convergence_curve = []   # track best F1 per generation

    # ---- Evolution loop ----
    for gen in range(n_generations):
        # Evaluate objectives for current population in parallel
        objectives = np.array(
            Parallel(n_jobs=-1)(
                delayed(_evaluate_objectives)(
                    *_split_chromosome(chrom, n_features),
                    X_train, y_train, X_val, y_val, rf_n_jobs=1
                )
                for chrom in population
            )
        )

        # Non-dominated sorting
        fronts = _non_dominated_sort(objectives)

        # Assign front ranks and crowding distances
        front_ranks = np.zeros(pop_size, dtype=int)
        crowding    = np.zeros(pop_size)
        for rank, front in enumerate(fronts):
            for idx in front:
                front_ranks[idx] = rank
            cd = _crowding_distance(objectives, front)
            for i, idx in enumerate(front):
                crowding[idx] = cd[i]

        # Track best F1 (from front 0, pick solution with lowest obj1)
        best_obj1_idx = fronts[0][np.argmin(objectives[fronts[0], 0])]
        best_f1_approx = -objectives[best_obj1_idx, 0]  # approximate F1
        convergence_curve.append(float(best_f1_approx))

        n_pareto = len(fronts[0])
        print(f"  Gen {gen+1:3d}/{n_generations} | Pareto front size={n_pareto}"
              f" | Best F1~={best_f1_approx:.4f}")

        # ---- Generate offspring ----
        offspring = []
        while len(offspring) < pop_size:
            i1 = _tournament_select_nsga2(front_ranks, crowding, pop_size, rng, tournament_k)
            i2 = _tournament_select_nsga2(front_ranks, crowding, pop_size, rng, tournament_k)
            c1, c2 = _crossover(population[i1], population[i2],
                                n_features, crossover_rate, blx_alpha, rng)
            c1 = _mutate(c1, n_features, mutation_rate_feature, mutation_rate_hp, rng)
            c2 = _mutate(c2, n_features, mutation_rate_feature, mutation_rate_hp, rng)
            offspring.extend([c1, c2])
        offspring = np.array(offspring[:pop_size])

        # ---- Evaluate offspring objectives in parallel ----
        offspring_obj = np.array(
            Parallel(n_jobs=-1)(
                delayed(_evaluate_objectives)(
                    *_split_chromosome(chrom, n_features),
                    X_train, y_train, X_val, y_val, rf_n_jobs=1
                )
                for chrom in offspring
            )
        )

        # ---- Combine parent + offspring (environmental selection) ----
        combined_pop = np.vstack([population, offspring])
        combined_obj = np.vstack([objectives, offspring_obj])

        # Non-dominated sort on combined
        combined_fronts = _non_dominated_sort(combined_obj)

        # Select next generation: fill pop_size slots by front rank
        next_pop = []
        next_idx_list = []
        for front in combined_fronts:
            if len(next_pop) + len(front) <= pop_size:
                next_pop.extend(front)
                next_idx_list.extend(front)
            else:
                # Partial fill: select by crowding distance
                remaining = pop_size - len(next_pop)
                cd = _crowding_distance(combined_obj, front)
                sorted_by_cd = np.argsort(-cd)   # descending crowding distance
                for idx in sorted_by_cd[:remaining]:
                    next_pop.append(front[idx])
                    next_idx_list.append(front[idx])
                break

        population = combined_pop[next_idx_list]

    runtime = time.time() - t_start

    # ---- Final evaluation ----
    # Re-evaluate final population in parallel
    final_objectives = np.array(
        Parallel(n_jobs=-1)(
            delayed(_evaluate_objectives)(
                *_split_chromosome(chrom, n_features),
                X_train, y_train, X_val, y_val, rf_n_jobs=1
            )
            for chrom in population
        )
    )

    final_fronts = _non_dominated_sort(final_objectives)
    pareto_indices = final_fronts[0]
    pareto_objectives = final_objectives[pareto_indices]

    # ---- Select "knee-point" solution ----
    # The knee point balances both objectives: use the minimum distance
    # to the ideal point (min obj1, min obj2) after normalisation.
    ideal = np.min(pareto_objectives, axis=0)
    nadir = np.max(pareto_objectives, axis=0)
    ranges = nadir - ideal
    ranges[ranges == 0] = 1.0   # avoid division by zero

    norm_obj = (pareto_objectives - ideal) / ranges
    distances_to_ideal = np.sqrt(np.sum(norm_obj ** 2, axis=1))
    knee_idx_in_pareto = np.argmin(distances_to_ideal)
    knee_idx = pareto_indices[knee_idx_in_pareto]

    best_chrom = population[knee_idx]
    best_feat_mask, best_hp_vec = _split_chromosome(best_chrom, n_features)

    # Constraint guard
    if best_feat_mask.sum() == 0:
        best_feat_mask[rng.integers(n_features)] = 1.0

    print(f"\n  NSGA-II Pareto front size: {len(pareto_indices)}")
    print(f"  Knee-point solution: F1~={-final_objectives[knee_idx, 0]:.4f}, "
          f"Feature ratio={final_objectives[knee_idx, 1]:.4f}")

    # Plot Pareto front
    plot_pareto_front(pareto_objectives, final_objectives)

    # ---- Final refit on full training set ----
    metrics, clf, _ = train_and_evaluate(
        X_fit, y_fit, X_test, y_test,
        best_feat_mask, best_hp_vec,
        method_name="NSGA-II",
        n_total_features=n_features,
    )
    metrics["Runtime_s"] = round(runtime, 2)
    _print_metrics(metrics)

    selected_names = [feature_names[i] for i in range(n_features)
                      if best_feat_mask[i] > 0.5]
    print(f"\n  NSGA-II selected features ({len(selected_names)}):")
    print(f"  {selected_names}")

    return {
        "metrics":            metrics,
        "convergence_curve":  convergence_curve,
        "best_feature_mask":  best_feat_mask,
        "best_hp_vector":     best_hp_vec,
        "runtime_s":          runtime,
        "clf":                clf,
        "pareto_front":       population[pareto_indices],
        "pareto_objectives":  pareto_objectives,
    }
