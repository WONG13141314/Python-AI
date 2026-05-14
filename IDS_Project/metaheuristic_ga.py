"""
=============================================================================
COMP2024 - Artificial Intelligence Methods
IDS Feature Selection & Hyperparameter Optimisation using Metaheuristics
=============================================================================
Module  : metaheuristic_ga.py
Purpose : Genetic Algorithm (GA) for joint feature selection and
          hyperparameter optimisation of a Random Forest IDS model.

Algorithm overview
------------------
A GA evolves a population of candidate solutions (chromosomes).
Each chromosome encodes:
    - a binary feature mask  (length = n_features)
    - a continuous HP vector (length = len(HP_KEYS))  values in [0,1]

Operators:
    Selection   : Tournament selection (k=3)
    Crossover   : Uniform crossover for feature bits;
                  arithmetic (BLX-alpha) crossover for HP values
    Mutation    : Bit-flip for features; Gaussian perturbation for HPs
    Elitism     : Best solution carried over each generation

Constraint handling
-------------------
    Hard constraint: at least one feature must be selected.
    Enforced at two points:
        (1) After mutation  - if all bits are 0, one random bit is flipped to 1.
        (2) After evolution - fallback guard on the final best chromosome.

References
----------
Holland, J.H. (1975). Adaptation in Natural and Artificial Systems.
Whitley, D. (1994). A genetic algorithm tutorial. Statistics and Computing.
Authors : Group XXX
=============================================================================
"""

import numpy as np
import time
from joblib import Parallel, delayed
from evaluation import evaluate_solution, train_and_evaluate, HP_KEYS, _print_metrics


# ---------------------------------------------------------------------------
# Chromosome helpers
# ---------------------------------------------------------------------------

def _random_chromosome(n_features: int, rng: np.random.Generator) -> np.ndarray:
    """
    Create a random chromosome of length (n_features + len(HP_KEYS)).

    Layout:
        [0 : n_features]               -> binary feature mask (0 or 1)
        [n_features : n_features+n_hp] -> HP values in [0, 1]
    """
    n_hp    = len(HP_KEYS)
    feature = rng.integers(0, 2, size=n_features).astype(float)   # {0, 1}
    hp      = rng.uniform(0, 1, size=n_hp)
    return np.concatenate([feature, hp])


def _split_chromosome(chrom: np.ndarray, n_features: int):
    """Split chromosome into (feature_mask, hp_vector)."""
    return chrom[:n_features], chrom[n_features:]


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def _tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    k: int = 3,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Select one individual via k-way tournament."""
    contestants = rng.choice(len(population), size=k, replace=False)
    winner      = contestants[np.argmax(fitness[contestants])]
    return population[winner].copy()


def _crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    n_features: int,
    crossover_rate: float,
    alpha: float,           # BLX-alpha parameter
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce two offspring from two parents.

    Feature bits : uniform crossover  (each bit independently swapped)
    HP values    : BLX-alpha arithmetic crossover
    """
    if rng.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    # ---- Feature mask: uniform crossover ----
    mask  = rng.integers(0, 2, size=n_features)   # which parent provides bit
    f1    = np.where(mask == 0, parent1[:n_features], parent2[:n_features])
    f2    = np.where(mask == 0, parent2[:n_features], parent1[:n_features])

    # ---- HP values: BLX-alpha crossover ----
    p_hp1 = parent1[n_features:]
    p_hp2 = parent2[n_features:]
    lo    = np.minimum(p_hp1, p_hp2) - alpha * np.abs(p_hp1 - p_hp2)
    hi    = np.maximum(p_hp1, p_hp2) + alpha * np.abs(p_hp1 - p_hp2)
    hp1   = np.clip(rng.uniform(lo, hi), 0, 1)
    hp2   = np.clip(rng.uniform(lo, hi), 0, 1)

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
    """
    Apply mutation in-place:
        Feature bits : flip with probability mutation_rate_feature
        HP values    : Gaussian perturbation with probability mutation_rate_hp

    Constraint: if all feature bits are 0 after mutation, one random bit
    is set to 1 (hard constraint - at least one feature must be selected).
    Uses the seeded rng generator for full reproducibility.
    """
    chrom = chrom.copy()

    # Feature bit-flip mutation
    for i in range(n_features):
        if rng.random() < mutation_rate_feature:
            chrom[i] = 1.0 - chrom[i]          # flip 0 <-> 1

    # --- FIX: use seeded rng.integers instead of np.random.randint ---
    # Constraint: ensure at least one feature is selected
    if chrom[:n_features].sum() == 0:
        chrom[rng.integers(n_features)] = 1.0

    # HP Gaussian mutation
    n_hp = len(HP_KEYS)
    for j in range(n_features, n_features + n_hp):
        if rng.random() < mutation_rate_hp:
            chrom[j] = np.clip(chrom[j] + rng.normal(0, 0.1), 0, 1)

    return chrom


# ---------------------------------------------------------------------------
# Main GA routine
# ---------------------------------------------------------------------------

def run_ga(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_features: int,
    feature_names: list,
    # --- FIX: accept X_train_full / y_train_full for the final refit ---
    # These are the FULL training set (before validation split).
    # The validation split (X_train, y_train) is only used during
    # fitness evaluation inside the evolution loop.
    X_train_full: np.ndarray = None,
    y_train_full: np.ndarray = None,
    pop_size: int       = 30,
    n_generations: int  = 40,
    crossover_rate: float = 0.85,
    mutation_rate_feature: float = 0.02,   # per-bit flip probability
    mutation_rate_hp: float = 0.15,         # per-HP mutation probability
    tournament_k: int   = 3,
    blx_alpha: float    = 0.3,
    seed: int           = 42,
) -> dict:
    """
    Run Genetic Algorithm for feature selection + HP tuning.

    Parameters
    ----------
    pop_size       : number of individuals in the population
    n_generations  : number of generations to evolve
    crossover_rate : probability of performing crossover
    mutation_rate_feature : per-bit flip probability for feature mask
    mutation_rate_hp      : per-gene Gaussian mutation probability for HPs
    tournament_k   : tournament size for selection
    blx_alpha      : BLX-alpha parameter for HP crossover
    X_train_full   : full training set used ONLY for the final model refit
                     (if None, falls back to X_train -- for backward compat)
    y_train_full   : labels for X_train_full

    Returns
    -------
    dict with keys: metrics, convergence_curve, best_feature_mask,
                    best_hp_vector, runtime_s
    """
    print("\n" + "=" * 60)
    print("  METAHEURISTIC 1: Genetic Algorithm (GA)")
    print(f"  Population={pop_size}, Generations={n_generations}")
    print("=" * 60)

    # Fall back to training split if full set not provided
    X_fit = X_train_full if X_train_full is not None else X_train
    y_fit = y_train_full if y_train_full is not None else y_train

    rng = np.random.default_rng(seed)
    t_start = time.time()

    # ---- Initialise population ----
    population = np.array([
        _random_chromosome(n_features, rng) for _ in range(pop_size)
    ])

    best_fitness   = -np.inf
    best_chrom     = population[0].copy()
    convergence_curve = []

    # ---- Evolution loop ----
    for gen in range(n_generations):
        # Evaluate all individuals in parallel (uses validation split for fitness)
        fitness = np.array(
            Parallel(n_jobs=-1)(
                delayed(evaluate_solution)(
                    *_split_chromosome(chrom, n_features),
                    X_train, y_train, X_val, y_val, rf_n_jobs=1
                )
                for chrom in population
            )
        )

        # Track best
        gen_best_idx = np.argmax(fitness)
        if fitness[gen_best_idx] > best_fitness:
            best_fitness = fitness[gen_best_idx]
            best_chrom   = population[gen_best_idx].copy()

        convergence_curve.append(best_fitness)
        n_sel = int(best_chrom[:n_features].sum())
        print(f"  Gen {gen+1:3d}/{n_generations} | Best fitness={best_fitness:.4f}"
              f"  | Features selected={n_sel}")

        # ---- Elitism: carry best individual ----
        new_population = [best_chrom.copy()]

        # ---- Generate offspring ----
        while len(new_population) < pop_size:
            p1 = _tournament_selection(population, fitness, tournament_k, rng)
            p2 = _tournament_selection(population, fitness, tournament_k, rng)
            c1, c2 = _crossover(p1, p2, n_features, crossover_rate, blx_alpha, rng)
            c1 = _mutate(c1, n_features, mutation_rate_feature, mutation_rate_hp, rng)
            c2 = _mutate(c2, n_features, mutation_rate_feature, mutation_rate_hp, rng)
            new_population.extend([c1, c2])

        population = np.array(new_population[:pop_size])

    runtime = time.time() - t_start

    # ---- Final evaluation on held-out test set ----
    best_feat_mask, best_hp_vec = _split_chromosome(best_chrom, n_features)

    # Constraint guard: ensure at least one feature selected
    if best_feat_mask.sum() == 0:
        best_feat_mask[rng.integers(n_features)] = 1.0

    # --- FIX: final refit uses the FULL training set (X_fit / y_fit),
    # not just the 80% validation-excluded split. This ensures a fair
    # comparison with the baseline RF which also trains on all data. ---
    metrics, clf, _ = train_and_evaluate(
        X_fit, y_fit, X_test, y_test,
        best_feat_mask, best_hp_vec,
        method_name="GA",
        n_total_features=n_features,
    )
    metrics["Runtime_s"] = round(runtime, 2)

    _print_metrics(metrics)

    selected_names = [feature_names[i] for i in range(n_features)
                      if best_feat_mask[i] > 0.5]
    print(f"\n  GA selected features ({len(selected_names)}):")
    print(f"  {selected_names}")

    return {
        "metrics":           metrics,
        "convergence_curve": convergence_curve,
        "best_feature_mask": best_feat_mask,
        "best_hp_vector":    best_hp_vec,
        "runtime_s":         runtime,
        "clf":               clf,
    }
