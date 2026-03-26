"""
=============================================================================
COMP2024 - Artificial Intelligence Methods
IDS Feature Selection & Hyperparameter Optimisation using Metaheuristics
=============================================================================
Module  : metaheuristic_sa.py
Purpose : Simulated Annealing (SA) for joint feature selection and
          hyperparameter optimisation of a Random Forest IDS model.

Algorithm overview
------------------
SA is a probabilistic local search algorithm inspired by the annealing
process in metallurgy.  Starting from a random solution, it explores the
neighbourhood by generating a perturbed candidate solution.

The acceptance criterion (Metropolis criterion) allows uphill moves
with probability exp(−ΔE / T), where T is the current "temperature".
This prevents premature convergence to local optima.

Temperature schedule : geometric cooling  T(k) = T0 * r^k
    T0 = initial temperature
    r  = cooling rate (< 1)
Stopping: when T < T_min or max_iter reached.

Neighbourhood operators:
    Feature mask : randomly flip a subset of bits
    HP values    : Gaussian perturbation clamped to [0, 1]

References
----------
Kirkpatrick, S., Gelatt, C.D., & Vecchi, M.P. (1983). Optimization by
    Simulated Annealing. Science, 220(4598), 671-680.
Cerny, V. (1985). Thermodynamical approach to the travelling salesman
    problem. JOTA, 45, 41-51.
Authors : Group XXX
=============================================================================
"""

import numpy as np
import time
from evaluation import evaluate_solution, train_and_evaluate, HP_KEYS, _print_metrics


# ---------------------------------------------------------------------------
# Solution representation helpers
# ---------------------------------------------------------------------------

def _random_solution(n_features: int, rng: np.random.Generator) -> tuple:
    """
    Generate a random initial solution.

    Returns
    -------
    feature_mask : binary ndarray of shape (n_features,)
    hp_vector    : continuous ndarray of shape (len(HP_KEYS),), values in [0,1]
    """
    # Start with roughly half features selected
    feature_mask = (rng.uniform(size=n_features) > 0.5).astype(float)
    hp_vector    = rng.uniform(0, 1, size=len(HP_KEYS))
    return feature_mask, hp_vector


def _perturb(
    feature_mask: np.ndarray,
    hp_vector: np.ndarray,
    n_flip: int,
    hp_sigma: float,
    rng: np.random.Generator,
) -> tuple:
    """
    Generate a neighbour solution by small random perturbation.

    Parameters
    ----------
    n_flip   : number of feature bits to flip
    hp_sigma : std of Gaussian perturbation applied to each HP
    """
    new_feat = feature_mask.copy()
    new_hp   = hp_vector.copy()

    # Flip n_flip random feature bits
    flip_idx = rng.choice(len(feature_mask), size=n_flip, replace=False)
    for idx in flip_idx:
        new_feat[idx] = 1.0 - new_feat[idx]

    # Gaussian perturbation for each HP
    noise     = rng.normal(0, hp_sigma, size=len(hp_vector))
    new_hp    = np.clip(new_hp + noise, 0, 1)

    return new_feat, new_hp


# ---------------------------------------------------------------------------
# Main SA routine
# ---------------------------------------------------------------------------

def run_sa(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_features: int,
    feature_names: list,
    max_iter: int       = 500,
    T0: float           = 1.0,    # initial temperature
    T_min: float        = 1e-4,   # stopping temperature
    cooling_rate: float = 0.97,   # geometric cooling factor r
    n_flip: int         = 3,      # bits to flip per perturbation step
    hp_sigma: float     = 0.1,    # std for HP Gaussian perturbation
    seed: int           = 42,
) -> dict:
    """
    Run Simulated Annealing (SA) for feature selection + HP tuning.

    Parameters
    ----------
    max_iter     : maximum number of iterations
    T0           : initial temperature
    T_min        : algorithm stops when T drops below this value
    cooling_rate : geometric cooling rate (T ← T * cooling_rate each step)
    n_flip       : number of feature bits flipped to create neighbour
    hp_sigma     : Gaussian noise std applied to HP values each step

    Returns
    -------
    dict with keys: metrics, convergence_curve, best_feature_mask,
                    best_hp_vector, runtime_s
    """
    print("\n" + "=" * 60)
    print("  METAHEURISTIC 3: Simulated Annealing (SA)")
    print(f"  Max iter={max_iter}, T0={T0}, cooling={cooling_rate}")
    print("=" * 60)

    rng     = np.random.default_rng(seed)
    t_start = time.time()

    # ---- Initialise ----
    current_feat, current_hp = _random_solution(n_features, rng)
    current_fitness = evaluate_solution(
        current_feat, current_hp,
        X_train, y_train, X_val, y_val,
    )

    best_feat    = current_feat.copy()
    best_hp      = current_hp.copy()
    best_fitness = current_fitness

    T = T0
    convergence_curve = []
    accepted_count = 0

    # ---- Main loop ----
    for it in range(max_iter):
        if T < T_min:
            print(f"  [SA] Temperature below T_min ({T_min}). Stopping early at iter {it}.")
            break

        # Generate neighbour
        n_flip_cur = max(1, int(n_flip * (T / T0)))   # flip more bits when hot
        new_feat, new_hp = _perturb(current_feat, current_hp,
                                    n_flip_cur, hp_sigma, rng)

        # Ensure at least one feature selected
        if new_feat.sum() == 0:
            new_feat[rng.integers(n_features)] = 1.0

        new_fitness = evaluate_solution(
            new_feat, new_hp,
            X_train, y_train, X_val, y_val,
        )

        # Metropolis acceptance criterion
        delta = new_fitness - current_fitness
        if delta > 0:
            # Better solution – always accept
            current_feat    = new_feat
            current_hp      = new_hp
            current_fitness = new_fitness
            accepted_count += 1
        else:
            # Worse solution – accept with probability exp(delta / T)
            prob = np.exp(delta / T)
            if rng.random() < prob:
                current_feat    = new_feat
                current_hp      = new_hp
                current_fitness = new_fitness
                accepted_count += 1

        # Update global best
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_feat    = current_feat.copy()
            best_hp      = current_hp.copy()

        convergence_curve.append(best_fitness)

        # Cool down
        T *= cooling_rate

        # Progress report every 50 iterations
        if (it + 1) % 50 == 0 or it == 0:
            n_sel = int(best_feat.sum())
            print(f"  Iter {it+1:4d}/{max_iter} | T={T:.6f} | "
                  f"Best fitness={best_fitness:.4f} | Features={n_sel} | "
                  f"Accepted={accepted_count}")

    runtime = time.time() - t_start

    if best_feat.sum() == 0:
        best_feat[0] = 1.0

    # ---- Final evaluation on held-out test set ----
    metrics, clf, _ = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        best_feat, best_hp,
        method_name="SA",
        n_total_features=n_features,
    )
    metrics["Runtime_s"] = round(runtime, 2)

    _print_metrics(metrics)

    selected_names = [feature_names[i] for i in range(n_features)
                      if best_feat[i] > 0.5]
    print(f"\n  SA selected features ({len(selected_names)}):")
    print(f"  {selected_names}")

    return {
        "metrics":           metrics,
        "convergence_curve": convergence_curve,
        "best_feature_mask": best_feat,
        "best_hp_vector":    best_hp,
        "runtime_s":         runtime,
        "clf":               clf,
    }
