# Ampire
Ampire: Distributed Kalman Approximate Message Passing Algorithm

A Python package implementing distributed Kalman-based Approximate Message Passing (DKAMP) for sparse signal recovery over directed acyclic graphs (DAGs), enabling efficient decentralized estimation with shared unknown vectors across nodes.

## Installation

Install from PyPI:
```bash
pip install ampire
```

Or install from source:
```bash
git clone https://github.com/YousfLHC/ampire.git
cd ampire
pip install -e .
```

## Usage

```python
from ampire import KAMP, DistributedKAMP

# Example usage
kamp = KAMP(alpha=0.5, tau=0.1, max_iter=100)
# ... fit and solve
```





# Distributed Kalman Approximate Message Passing Algorithm
This repository contains research code for solving sparse reconstruction and anomaly detection problems using Approximate Message Passing (AMP) and Kalman‑filtered AMP (KAMP) algorithms. In addition to the core solvers, a collection of utility functions live in src/ampire/utils/metrics.py to help evaluate models on classification, anomaly detection and compressive sensing tasks. This section documents those metrics and explains when to use them.
## Metrics utilities
The `metrics.py` module provides helper functions for computing standard performance metrics across three domains:
•	Anomaly detection – binary classification of rare events or outliers.
•	Compressive sensing and reconstruction – quantifying how well a reconstructed signal matches the ground truth.
•	Algorithm diagnostics – tracking convergence and phase transitions of iterative solvers.
## AUC‑based metrics
Function: `calculate_metrics(y_true, y_score)`
This helper computes two threshold‑independent scores from a list of true binary labels and continuous anomaly scores:
•	ROC AUC (`auc_roc`) – area under the Receiver Operating Characteristic curve, which plots the true positive rate against the false positive rate. A score of 1.0 indicates perfect discrimination; 0.5 corresponds to random guessing.
•	Precision–Recall AUC (`auc_pr`) – area under the precision–recall curve. Precision is the fraction of flagged samples that are actually anomalies[1], while recall (also called sensitivity) is the fraction of true anomalies detected[2]. The PR‑AUC is particularly useful when anomalies are extremely rare.
## Discrete anomaly‑detection metrics
Function: `calculate_anomaly_detection_metrics(y_true, y_pred)`
Given ground‑truth labels and binary predictions, this function returns a dictionary of metrics that summarise different aspects of detector performance:
Metric	Definition	When to use / interpretation
Precision	True positives divided by all positives (TP/(TP+FP))[1].
High precision means few false alarms.
Recall (sensitivity)	True positives divided by actual positives (TP/(TP+FN))[2].
High recall means the detector rarely misses anomalies.
Specificity	True negatives divided by normal samples (TN/(TN+FP)).	Measures how well normal data are correctly labelled; complements recall.
F1 score	Harmonic mean of precision and recall[3].
Balances detection and false‑alarm rates, robust to class imbalance[4].

Cohen’s kappa	(Observed accuracy − Expected accuracy) divided by (1 − Expected accuracy)[5].
Measures agreement beyond chance; values near 1 imply almost perfect agreement[6].

Matthews correlation coefficient (MCC)	Correlation coefficient between predicted and true labels.	Takes all four confusion matrix entries into account; suitable for imbalanced data.
Balanced accuracy	Average of recall and specificity[7].
Weights positive and negative classes equally, avoiding bias towards the majority class[8].

F2 score	Weighted harmonic mean favouring recall; computed as 5×precision×recall divided by (4×precision + recall).	Emphasises detecting anomalies at the expense of false alarms (recall weight 2).
G‑mean	Square root of recall multiplied by specificity[9].
Represents balanced performance across both classes.
False positive rate (FPR)	FP/(FP+TN).	Probability of incorrectly flagging normal samples.
False negative rate (FNR)	FN/(FN+TP).	Probability of missing an actual anomaly.
False discovery rate (FDR)	FP/(FP+TP).	Proportion of predicted anomalies that are false alarms.
Negative predictive value (NPV)	TN/(TN+FN).	Probability that a sample predicted as normal is truly normal.
False omission rate (FOR)	FN/(FN+TN).	Proportion of missed anomalies among normal predictions.
Youden’s J (Bookmaker informedness)	Recall + Specificity − 1.	Combines sensitivity and specificity; 0 indicates random performance.
Markedness	Precision + NPV − 1.	Indicates reliability of positive and negative predictions together.


These metrics allow a nuanced evaluation of anomaly detectors. Depending on the application, one may prioritise high recall (safety‑critical systems) or high precision (limited analyst time) or seek a balance.
## Compressive‑sensing metrics
Function: `calculate_compressive_sensing_metrics(y_true, y_pred)`
Computes several error measures when comparing a reconstructed signal with the ground truth:
•	Mean squared error (`MSE`) – average squared difference between true and reconstructed signals. Lower values indicate more accurate reconstructions.
•	Root mean squared error (`RMSE`) – square root of the MSE. Expresses error in the same units as the signal for easier interpretation.
•	Normalised mean squared error (`NMSE`) – MSE divided by the variance of the true signal. Commonly used in AMP literature to compare performance across signals of different scales.
•	Signal‑to‑noise ratio (`SNR`) – in decibels, 10·log10(var(y_true)/MSE). Higher SNR indicates a stronger signal relative to reconstruction error.
•	Peak signal‑to‑noise ratio (`Peak SNR`) – 20·log10(max(abs(y_true))/√MSE). Often used for images and audio where the peak amplitude is known. High values correspond to better reconstruction quality.
##  Image‑quality metrics
The following full‑reference measures compare predicted and true images:
•	Gradient magnitude similarity deviation (GMSD) (`calculate_gmsd`) – computes gradients of both images, forms a similarity map and returns its standard deviation. Lower GMSD values imply better preservation of edge structures.
•	Feature similarity index (FSIM) (`calculate_fsim`) – evaluates phase congruency and gradient magnitude similarity. Values close to 1 indicate high similarity of perceptually significant features.
•	Visual information fidelity (VIF) (`calculate_vif`) – approximates the ratio of mutual information between the predicted and true images. A higher VIF means the reconstruction preserves more of the original visual information.
•	Structural similarity index measure (SSIM) (`calculate_ssim`) – compares two images based on luminance, contrast and structural terms. SSIM values range from 0 (no similarity) to 1 (identical images)[10] and often align more closely with human perception than MSE or PSNR.
## Phase transition and convergence tools
•	`calculate_phase_transition(successes, sparsities, measurement_rates)` – aggregates binary success outcomes for combinations of sparsity and measurement rate and returns the probability of success for each pair. Useful for plotting phase diagrams that identify the boundary between successful and unsuccessful recovery regimes in sparse reconstruction[11].
•	`track_convergence(iterations, runtime, tolerance=None, error_history=None)` – records diagnostics of iterative solvers. Besides the iteration count and runtime, optional fields include the tolerance used, the final error, the full error history and the average change between consecutive errors. These summaries help compare the efficiency and stability of AMP, KAMP or related algorithms.
## Logging
•	`log_results(results, filepath)` – appends a timestamped summary of experiment results (e.g., AUC‑ROC and runtime) to a log file. The default path is results/logs/experiment_log.txt. This function is intended for convenient experiment bookkeeping.


________________________________________
[1] [3] [4] Classification: Accuracy, recall, precision, and related metrics  |  Machine Learning  |  Google for Developers
https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall
[2] Sensitivity and specificity - Wikipedia
https://en.wikipedia.org/wiki/Sensitivity_and_specificity
[5] [6] classification - Cohen's kappa in plain English - Cross Validated
https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english
[7] [8] Balanced Accuracy: When Should You Use It?
https://neptune.ai/blog/balanced-accuracy
[9] Classification performance metrics and indices • metrica
https://adriancorrendo.github.io/metrica/articles/available_metrics_classification.html
[10] Structural similarity index measure - Wikipedia
https://en.wikipedia.org/wiki/Structural_similarity_index_measure
[11] 1610.03082
https://arxiv.org/pdf/1610.03082















## File Structure
ampire/
├── src/
│   └── ampire/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── kamp.py              # Kalman-based AMP implementation
│       │   ├── enet_convex_hull.py  # Elastic Net-based convex hull classifier
│       │   └── threshold_finder.py  # Threshold finding for anomaly detection
│       ├── network/
│       │   ├── __init__.py
│       │   ├── graph.py             # Graph utilities (MyGraph, Node classes)
│       │   └── random_digraphs.py   # Random DAG generation
│       ├── distributed/
│       │   ├── __init__.py
│       │   └── distributed_kamp.py  # Distributed KAMP implementation
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── metrics.py           # Evaluation metrics (e.g., AUC-ROC, PR)
│       │   └── visualization.py     # Plotting and visualization utilities
│       ├── examples/
│       │   ├── __init__.py
│       │   └── example_*.py         # Example scripts
│       ├── tests/
│       │   ├── __init__.py
│       │   └── test_*.py            # Test files
│       └── docs/
│           ├── api.md               # API documentation
│           └── tutorial.md          # Usage tutorial
├── setup.py                    # Package installation script
├── MANIFEST.in                 # Files to include in distribution
├── README.md                   # Repository description
├── LICENSE                     # License (e.g., MIT)
└── requirements.txt            # Dependencies
