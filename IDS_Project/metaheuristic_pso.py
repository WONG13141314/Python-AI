"""
=============================================================================
COMP2024 - Artificial Intelligence Methods
IDS Feature Selection & Hyperparameter Optimisation using Metaheuristics
=============================================================================
Module  : metaheuristic_pso.py
Purpose : Particle Swarm Optimisation (PSO) for joint feature selection and
          hyperparameter optimisation of a Random Forest IDS model.

Algorithm overview
------------------
PSO simulates a swarm of particles (candidate solutions) flying through the
search space.  Each particle has a position (x) and velocity (v).

Position encoding (same as GA chromosome):
    [0 : n_features]               -> feature mask (sigmoid -> binarised)
    [n_features : n_features+n_hp] -> HP values clamped to [0, 1]

Velocity update:
    v(t+1) = w * v(t)
           + c1 * r1 * (pbest - x(t))     <- cognitive component
           + c2 * r2 * (gbest - x(t))     <- social component

Position update:
    x(t+1) = x(t) + v(t+1)

For feature bits, the S-shaped transfer function maps position to a
probability of the bit being 1 (binary PSO / BPSO approach by Kennedy &
Eberhart 1997).

Constraint handling
-------------------
    Hard constraint: at least one feature must be selected.
    Enforced after binarisation: if all bits are 0, the feature dimension
    with the highest sigmoid score is forced to 1. The final deterministic
    mask (threshold 0.5) applies the same fallback using argmax.

References
----------
Kennedy, J. & Eberhart, R. (1995). Particle swarm optimization. ICNN.
Kennedy, J. & Eberhart, R. (1997). A discrete binary version of the PSO.
Clerc, M. & Kennedy, J. (2002). The particle swarm - explosion, stability.
Authors : Group XXX
=============================================================================
"""

import numpy as np
import time
from joblib import Parallel, delayed
from evaluation import evaluate_solution, train_and_evaluate, HP_KEYS, _print_metrics


# ---------------------------------------------------------------------------
# Sigmoid transfer function (used for binary feature decisions)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ---------------------------------------------------------------------------
# Position / velocity helpers
# ---------------------------------------------------------------------------

def _init_particle(n_features: int, rng: np.random.Generator) -> tuple:
    """
    Initialise a particle's position and velocity.

    Position layout (same as GA chromosome):
        [0:n_features]         -> continuous; binarised via sigmoid during eval
        [n_features:]          -> HP values in [0, 1]

    Velocity: small random values centred on zero.
    """
    n_hp = len(HP_KEYS)
    pos = np.concatenate([
        rng.uniform(-4, 4,  size=n_features),   # pre-sigmoid feature scores
        rng.uniform(0,  1,  size=n_hp),          # HP values
    ])
    vel = rng.uniform(-0.5, 0.5, size=n_features + n_hp)
    return pos, vel


def _binarise_features(pos: np.ndarray, n_features: int, rng: np.random.Generator) -> np.ndarray:
    """
    Convert continuous feature positions to binary mask using sigmoid.
    bit_i = 1  if  random() < sigmoid(pos_i)

    Constraint: if all bits are 0 (degenerate case), force the dimension
    with the highest sigmoid score to 1.
    """
    probs = _sigmoid(pos[:n_features])
    bits  = (rng.uniform(size=n_features) < probs).astype(float)
    # Hard constraint: at least one feature must be selected
    if bits.sum() == 0:
        bits[np.argmax(probs)] = 1.0
    return bits


def _clip_hp(pos: np.ndarray, n_features: int) -> np.ndarray:
    """Clip HP portion of position to [0, 1]."""
    pos = pos.copy()
    pos[n_features:] = np.clip(pos[n_features:], 0, 1)
    return pos


# ---------------------------------------------------------------------------
# Main PSO routine
# ---------------------------------------------------------------------------

def run_pso(
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
    # fitness evaluation inside the swarm loop.
    X_train_full: np.ndarray = None,
    y_train_full: np.ndarray = None,
    n_particles: int   = 30,
    n_iterations: int  = 40,
    w: float           = 0.7,    # inertia weight (kept for compat, overridden by decay)
    w_min: float       = 0.4,    # minimum inertia (linear decay)
    w_max: float       = 0.9,    # maximum inertia
    c1: float          = 2.0,    # cognitive coefficient
    c2: float          = 2.0,    # social coefficient
    v_max_feat: float  = 4.0,    # velocity clamp for feature dimensions
    v_max_hp: float    = 0.2,    # velocity clamp for HP dimensions
    seed: int          = 42,
) -> dict:
    """
    Run Particle Swarm Optimisation (PSO) for feature selection + HP tuning.

    Parameters
    ----------
    n_particles  : swarm size
    n_iterations : number of iteration steps
    w            : inertia weight (linearly decayed w_max->w_min each iter)
    c1           : cognitive learning factor (personal best attraction)
    c2           : social learning factor (global best attraction)
    v_max_feat   : velocity clamp for feature-dimension velocities
    v_max_hp     : velocity clamp for HP-dimension velocities
    X_train_full : full training set used ONLY for the final model refit
                   (if None, falls back to X_train for backward compat)
    y_train_full : labels for X_train_full

    Returns
    -------
    dict with keys: metrics, convergence_curve, best_feature_mask,
                    best_hp_vector, runtime_s
    """
    print("\n" + "=" * 60)
    print("  METAHEURISTIC 2: Particle Swarm Optimisation (PSO)")
    print(f"  Swarm size={n_particles}, Iterations={n_iterations}")
    print("=" * 60)

    # Fall back to training split if full set not provided
    X_fit = X_train_full if X_train_full is not None else X_train
    y_fit = y_train_full if y_train_full is not None else y_train

    rng     = np.random.default_rng(seed)
    n_hp    = len(HP_KEYS)
    n_dim   = n_features + n_hp
    t_start = time.time()

    # ---- Initialise swarm ----
    positions  = []
    velocities = []
    for _ in range(n_particles):
        pos, vel = _init_particle(n_features, rng)
        positions.append(pos)
        velocities.append(vel)

    positions  = np.array(positions)
    velocities = np.array(velocities)

    # Personal bests
    pbest_pos     = positions.copy()
    pbest_fitness = np.full(n_particles, -np.inf)

    # Global best
    gbest_pos     = positions[0].copy()
    gbest_fitness = -np.inf

    convergence_curve = []

    # ---- Iteration loop ----
    for iteration in range(n_iterations):

        # Linear inertia decay: improves exploration early, exploitation late
        w_cur = w_max - (w_max - w_min) * (iteration / n_iterations)

        # Prepare arguments for parallel evaluation
        feat_masks = [_binarise_features(positions[i], n_features, rng) for i in range(n_particles)]
        hp_vecs    = [np.clip(positions[i][n_features:], 0, 1) for i in range(n_particles)]

        # Evaluate all particles in parallel
        fitnesses = Parallel(n_jobs=-1)(
            delayed(evaluate_solution)(
                feat_masks[i], hp_vecs[i],
                X_train, y_train, X_val, y_val, rf_n_jobs=1
            )
            for i in range(n_particles)
        )

        for i in range(n_particles):
            fit = fitnesses[i]

            # Update personal best
            if fit > pbest_fitness[i]:
                pbest_fitness[i] = fit
                pbest_pos[i]     = positions[i].copy()

            # Update global best
            if fit > gbest_fitness:
                gbest_fitness = fit
                gbest_pos     = positions[i].copy()

        convergence_curve.append(gbest_fitness)
        n_sel = int(_binarise_features(gbest_pos, n_features, rng).sum())
        print(f"  Iter {iteration+1:3d}/{n_iterations} | Best fitness={gbest_fitness:.4f}"
              f"  | Features~={n_sel}")

        # ---- Update velocities and positions ----
        r1 = rng.uniform(0, 1, size=(n_particles, n_dim))
        r2 = rng.uniform(0, 1, size=(n_particles, n_dim))

        velocities = (
            w_cur * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )

        # Clamp velocities: feature dims and HP dims separately
        velocities[:, :n_features] = np.clip(
            velocities[:, :n_features], -v_max_feat, v_max_feat)
        velocities[:, n_features:] = np.clip(
            velocities[:, n_features:], -v_max_hp, v_max_hp)

        positions += velocities
        positions  = _clip_hp(positions.T, n_features).T   # clip HP dims only

    runtime = time.time() - t_start

    # ---- Decode best solution ----
    # Use deterministic threshold 0.5 on sigmoid for final mask
    best_feat_probs = _sigmoid(gbest_pos[:n_features])
    best_feat_mask  = (best_feat_probs >= 0.5).astype(float)
    best_hp_vec     = np.clip(gbest_pos[n_features:], 0, 1)

    # Constraint guard: ensure at least one feature selected
    if best_feat_mask.sum() == 0:
        best_feat_mask[np.argmax(best_feat_probs)] = 1.0

    print(f"\n  PSO final feature count (deterministic threshold 0.5): "
          f"{int(best_feat_mask.sum())} / {n_features}")

    # --- FIX: final refit uses the FULL training set (X_fit / y_fit),
    # not just the 80% validation-excluded split. This ensures a fair
    # comparison with the baseline RF which also trains on all data. ---
    metrics, clf, _ = train_and_evaluate(
        X_fit, y_fit, X_test, y_test,
        best_feat_mask, best_hp_vec,
        method_name="PSO",
        n_total_features=n_features,
    )
    metrics["Runtime_s"] = round(runtime, 2)

    _print_metrics(metrics)

    selected_names = [feature_names[i] for i in range(n_features)
                      if best_feat_mask[i] > 0.5]
    print(f"\n  PSO selected features ({len(selected_names)}):")
    print(f"  {selected_names}")

    return {
        "metrics":           metrics,
        "convergence_curve": convergence_curve,
        "best_feature_mask": best_feat_mask,
        "best_hp_vector":    best_hp_vec,
        "runtime_s":         runtime,
        "clf":               clf,
    }
