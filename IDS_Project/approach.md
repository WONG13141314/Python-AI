# Project Approach & Implementation Details

This document outlines the technical methodology used in the updated IDS Metaheuristic project. Use this as a reference guide for writing your final report and explaining your design choices.

---

## 1. Data Preprocessing: Beyond Ordinality
### Methodology: One-Hot Encoding (OHE)
*   **What it does:** Converts categorical features (`protocol_type`, `service`, `flag`) into multiple binary columns (e.g., `protocol_type_tcp`, `protocol_type_udp`).
*   **How it works:** Instead of mapping "TCP" to 1 and "UDP" to 2 (which suggests UDP is "greater than" TCP), OHE creates orthogonal vectors.
*   **Why mention it in the report:** Standard Label Encoding introduces a "false ordinal relationship." Using OHE shows you understand that Random Forests are sensitive to these numerical relationships. It expands your feature space from **41 to 122**, making the metaheuristic search task more challenging and impressive.

---

## 2. The Multi-Objective Fitness Function
### Methodology: Weighted Penalty Function
*   **Function:** `Fitness = 0.85 * F1 - 0.05 * Feature_Ratio - 0.10 * FPR`
*   **Functionality:** 
    *   **F1-Score:** Primary detection quality.
    *   **Feature Ratio:** Penalises complex models (Occam's Razor).
    *   **FPR (False Positive Rate):** Explicitly penalises models that cause "alert fatigue."
*   **How it works:** Most students only optimize for accuracy. By adding a **security-specific penalty (FPR)**, you align the AI's behavior with real-world SOC (Security Operations Center) needs.

---

## 3. Metaheuristic Algorithms
### Genetic Algorithm (GA) & PSO & SA
*   **Joint Optimization:** These algorithms don't just pick features; they simultaneously tune the Random Forest hyperparameters (`max_depth`, `n_estimators`, etc.).
*   **Details for report:** Mention the **Chromosome/Particle Encoding**. You use a hybrid vector:
    1.  **Binary Head:** Positions 0–121 control feature selection.
    2.  **Continuous Tail:** Positions 122+ control model parameters.

### NSGA-II (The "Distinction" Feature)
*   **What it is:** A Multi-Objective Evolutionary Algorithm (MOEA).
*   **How it works:** Instead of turning objectives into one number (fitness), it keeps them separate. It uses **Non-Dominated Sorting** to find a "Pareto Front"—a set of solutions where you can't improve detection without increasing complexity.
*   **Detail for report:** Discuss the **Crowding Distance** mechanism. It ensures your search doesn't get stuck in one spot but instead finds a wide variety of "trade-off" options (e.g., a "lightweight" 10-feature model vs. a "heavyweight" 90-feature model).

---

## 4. Computational Efficiency: Parallelism
### Methodology: Joblib CPU Parallelisation
*   **Implementation:** `Parallel(n_jobs=-1)` wraps the population evaluation.
*   **How it works:** Metaheuristic "fitness checks" are **embarrassingly parallel**. We distribute the 30+ individuals across all your CPU cores simultaneously.
*   **Why mention it:** Shows you can optimize not just for accuracy, but for **training latency**—a critical factor in deploying AI for high-speed network traffic.

---

## 5. Visual Evidence & Validation
*   **ROC Curves & AUC:** Demonstrates the classifier's trade-off between sensitivity and specificity better than a single accuracy number.
*   **Confusion Matrix Heatmaps:** Provides transparent insight into "Normal vs Attack" misclassifications.
*   **Pareto Front Plot:** Visual evidence of the multi-objective search success.

---

# Why this Scores "Distinction"

To score a Distinction (>70%), you must demonstrate **critical evaluation** and **technical sophistication**. This implementation gives you both:

1.  **Exceeding Requirements:** The rubric asks for "metaheuristics." Most will do one. You did **four**, including NSGA-II, which is a graduate-level optimization technique.
2.  **Security-Aware Design:** By explicitly targeting **False Positive Rates (FPR)** in your fitness function, you prove you aren't just doing "generic AI," but are solving an **IDS-specific problem**. This satisfies the higher-level "Security Trade-offs" requirement.
3.  **Experimental Rigour:** Your use of **One-Hot Encoding** and **Parallelism** shows you follow industry-standard ML best practices. 
4.  **Rich Discussion Material:** Because you have a **Pareto Front**, your report doesn't just say "GA is better than PSO." It can say *"NSGA-II allows us to choose between a high-speed firewall model (10 features) or a high-precision forensic model (80 features) based on network load."*

---
**Reference for User:** This document is placed directly in your project folder as `approach.md` for easy access.
