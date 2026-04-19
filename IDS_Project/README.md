# COMP2024 – IDS Feature Selection & Hyperparameter Optimisation
## Group XXX | Spring 2026

---

## Project Overview

This project implements **three metaheuristic algorithms** for joint
feature selection and hyperparameter tuning of a **Random Forest** Intrusion
Detection System (IDS) classifier trained on the **CICIDS2017** dataset.

| Algorithm | Type | Search Strategy |
|-----------|------|-----------------|
| Genetic Algorithm (GA) | Evolutionary | Population-based, crossover + mutation |
| Particle Swarm Optimisation (PSO) | Swarm intelligence | Velocity-guided particle movement |
| Simulated Annealing (SA) | Trajectory-based | Probabilistic local search |
| NSGA-II | Multi-objective evolutionary | Non-dominated sorting + crowding distance |

All three are benchmarked against a **Baseline Random Forest** (all features,
default sklearn hyperparameters).

---

## File Structure

```
IDS_Project/
├── main.py                   ← Entry point (run this)
├── data_preprocessing.py     ← Download, clean, OHE, scale CICIDS2017
├── evaluation.py             ← Fitness function, metrics, plots
├── metaheuristic_ga.py       ← Genetic Algorithm
├── metaheuristic_pso.py      ← Particle Swarm Optimisation
├── metaheuristic_sa.py       ← Simulated Annealing
├── metaheuristic_nsga2.py    ← NSGA-II (Multi-Objective)
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
├── data/                     ← CICIDS2017 files (auto-downloaded)
├── plots/                    ← All generated figures
└── results/                  ← CSV metrics + feature masks JSON
```

---

## Setup Instructions

### 1. Prerequisites
- Python **3.10+** (tested with 3.11)
- pip

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 3. Dataset

The **CICIDS2017** dataset is **downloaded automatically** on the first run
from the public GitHub mirror:

> https://github.com/defcom17/NSL_KDD

Files are saved to `data/KDDTrain+.txt` and `data/KDDTest+.txt`.

If you already have the files, place them in the `data/` folder with these
exact names and the downloader will skip them.

### 4. Run the Experiment

**Full experiment** (~30–60 min depending on hardware):

```bash
python main.py
```

**Quick test run** (< 5 minutes, reduced iterations):

```bash
python main.py --quick
```

**Custom seed** (for reproducibility):

```bash
python main.py --seed 123
```

---

## Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Metrics CSV | `results/all_metrics.csv` | Accuracy, F1, FPR, AUC etc. for all methods |
| Feature masks | `results/best_feature_masks.json` | Binary feature masks per method |
| Class distribution | `plots/class_distribution.png` | Attack vs Normal counts |
| Feature importance | `plots/feature_importances.png` | RF baseline feature ranking |
| Correlation heatmap | `plots/feature_correlation_heatmap.png` | Feature–label correlations |
| Convergence curves | `plots/convergence_curves.png` | Fitness over iterations |
| Metrics comparison | `plots/metrics_comparison.png` | Grouped bar chart |
| Feature reduction | `plots/feature_reduction.png` | Features selected vs total |
| Trade-off scatter | `plots/tradeoff_scatter.png` | F1 vs FPR bubble chart |
| ROC curves | `plots/roc_curves.png` | Overlaid ROC curves with AUC |
| Confusion matrices | `plots/confusion_matrices.png` | Side-by-side heatmaps |
| Runtime comparison | `plots/runtime_comparison.png` | Computational cost bar chart |
| Pareto front | `plots/pareto_front.png` | NSGA-II multi-objective trade-off |

---

## Methodology

### Base Model
- **Random Forest Classifier** (sklearn `RandomForestClassifier`)
- Fixed as the underlying IDS model for all experiments

### Optimisation Variables

Each metaheuristic searches over:

| Variable | Type | Range |
|----------|------|-------|
| Feature mask | Binary | {0, 1}^N (N ≈ 122 after OHE) |
| `n_estimators` | Integer | [10, 300] |
| `max_depth` | Integer | [2, 30] |
| `min_samples_split` | Integer | [2, 20] |
| `min_samples_leaf` | Integer | [1, 10] |
| `max_features` | Float | [0.1, 1.0] |

### Fitness Function (Single-Objective: GA, PSO, SA)

```
Fitness = 0.85 × F1_score  −  0.05 × feature_ratio  −  0.10 × FPR
```

A higher fitness is better. The penalty term encourages sparse feature subsets.

The FPR penalty directly encourages the optimiser to reduce false positives,
which is critical in IDS security deployments.

### Multi-Objective Fitness (NSGA-II)

NSGA-II does NOT combine objectives into a scalar. Instead it simultaneously
optimises two conflicting objectives:

```
Objective 1 (min): -F1 + 0.1×FPR   (maximise detection, minimise false alarms)
Objective 2 (min):  feature_ratio   (minimise complexity)
```

The result is a Pareto front; the "knee-point" solution is selected as the
recommended output.

### Evaluation Metrics

- Accuracy, Precision, Recall (TPR), F1-Score
- False Positive Rate (FPR)
- AUC (Area Under ROC Curve)
- Number of Features Selected
- Runtime (seconds)
- Pareto front (NSGA-II)

---

## Algorithm Summary

### Genetic Algorithm (GA)
- **Chromosome**: binary feature mask + continuous HP vector
- **Selection**: Tournament selection (k=3)
- **Crossover**: Uniform crossover (features) + BLX-α (hyperparameters)
- **Mutation**: Bit-flip (features) + Gaussian perturbation (HPs)
- **Elitism**: Best individual preserved each generation

### Particle Swarm Optimisation (PSO)
- **Position**: continuous pre-sigmoid scores (features) + HP values
- **Binary decode**: sigmoid transfer function → stochastic binarisation
- **Velocity update**: standard PSO with linear inertia decay (w: 0.9→0.4)
- **Coefficients**: c1=c2=2.0 (balanced cognitive/social)

### Simulated Annealing (SA)
- **Neighbourhood**: bit-flip (features) + Gaussian noise (HPs)
- **Acceptance**: Metropolis criterion  exp(ΔE / T)
- **Cooling**: geometric schedule  T ← T × 0.97
- **Adaptive**: number of bits flipped scales with temperature

### NSGA-II (Multi-Objective)
- **Encoding**: same chromosome as GA (binary features + continuous HPs)
- **Ranking**: fast non-dominated sorting into Pareto fronts
- **Diversity**: crowding distance to maintain spread on the front
- **Selection**: tournament based on (1) front rank, (2) crowding distance
- **Output**: full Pareto front + knee-point recommended solution

---

## References

1. Holland, J.H. (1975). *Adaptation in Natural and Artificial Systems*. MIT Press.
2. Kennedy, J. & Eberhart, R. (1995). Particle swarm optimization. *ICNN*.
3. Kirkpatrick, S., Gelatt, C.D. & Vecchi, M.P. (1983). Optimization by Simulated Annealing. *Science*, 220(4598).
4. Tavallaee, M. et al. (2009). A Detailed Analysis of the KDD Cup 99 Data Set. *IEEE CISDA*.
5. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
