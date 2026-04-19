# IDS Feature Selection & Hyperparameter Optimisation using Metaheuristics

**Module:** COMP2024 - Artificial Intelligence Methods  
**Group:** Group Ambatucode
**Authors:** Yew Xin Nie, Wong Yung Tung, Teoh Zhuo Qi  & Tai Sze-Song

---

## Prerequisites

- **Python 3.10+**
- **requirements.txt** – All dependencies are listed in `requirements.txt`

---

## Setup Instructions

### Windows (Command Prompt / PowerShell / Terminal)

1. **Navigate to the project directory:**
   ```bash
   cd IDS_Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### macOS (Terminal)

1. **Navigate to the project directory:**
   ```bash
   cd IDS_Project
   ```

2. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

> **Note:** On macOS, use `python3` instead of `python` for running the script.

---

## Dataset Instructions

The **NSL-KDD dataset** is automatically downloaded on the first run.

- The dataset files are stored in the `data/` folder:
  - `data/KDDTrain+.txt` – Training data
  - `data/KDDTest+.txt` – Test data

- If the files do not exist locally, the script will automatically download them from the official NSL-KDD GitHub repository.
  - https://github.com/Jehuty4949/NSL_KDD

---

## How to Run the Code

### Windows

**Full Experiment (~60-300 minutes)**
```bash
python main.py
```

**Quick Test Run (~5 minutes)**
```bash
python main.py --quick
```

**Custom Random Seed (optional)**
```bash
python main.py --seed 42
```

### macOS

**Full Experiment (~60-300 minutes)**
```bash
python3 main.py
```

**Quick Test Run (~5 minutes)**
```bash
python3 main.py --quick
```

**Custom Random Seed (optional)**
```bash
python3 main.py --seed 42
```

---

## Project Structure

```
IDS_Project/
├── main.py                     # Entry point - runs the full experimental pipeline
├── data_preprocessing.py       # Data loading, preprocessing, and EDA
├── evaluation.py               # Model evaluation and plotting functions
├── metaheuristic_ga.py         # Genetic Algorithm implementation
├── metaheuristic_pso.py        # Particle Swarm Optimisation implementation
├── metaheuristic_sa.py        # Simulated Annealing implementation
├── metaheuristic_nsga2.py      # NSGA-II (Multi-Objective) implementation
├── requirements.txt           # Python dependencies
├── data/                     # Dataset folder
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── plots/                     # Output plots
│   ├── convergence_curves.png
│   ├── metrics_comparison.png
│   ├── feature_reduction.png
│   ├── tradeoff_scatter.png
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── runtime_comparison.png
│   ├── pareto_front.png
│   ├── feature_importances.png
│   ├── feature_correlation_heatmap.png
│   └── class_distribution.png
└── results/                  # Output results
    ├── all_metrics.csv
    └── best_feature_masks.json
```

---

## Output

After running the experiment:

- **Plots:** All visualizations are saved in `./plots/`
- **Results:** Metrics CSV and feature masks JSON are saved in `./results/`

---

## Notes

- The experiment performs binary classification (Normal vs Attack) using Random Forest as the base classifier.
- Four metaheuristics are compared: GA, PSO, SA, and NSGA-II.
- The `--quick` flag reduces population size and iterations for fast testing.