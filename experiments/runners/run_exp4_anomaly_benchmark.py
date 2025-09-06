import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_recall_curve
import sys
import os
from tqdm import tqdm
import networkx as nx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ampire.core.kamp import KAMP
from ampire.core.amp import AMP
from ampire.distributed.distributed_kamp import DistributedKAMP
from ampire.utils.metrics import calculate_anomaly_detection_metrics
from experiments.datasets.benchmark_loaders import load_odds_dataset  # Assuming this exists

def run_solver(method, X_train, X_test):
    n_features = X_train.shape[1]
    scores = []
    for x in X_test:
        if method == 'KAMP':
            kamp = KAMP(alpha=0.5, tau=0.1, max_iter=100)
            kamp.fit(np.eye(n_features), x.reshape(-1, 1))
            x_recon = kamp.solve().flatten()
        elif method == 'AMP':
            amp = AMP(alpha=0.5, tau=0.1, max_iter=100)
            amp.fit(np.eye(n_features), x.reshape(-1, 1))
            x_recon = amp.solve().flatten()
        elif method == 'DistributedKAMP':
            # For simplicity, use single node for reconstruction
            kamp = KAMP(alpha=0.5, tau=0.1, max_iter=100)
            kamp.fit(np.eye(n_features), x.reshape(-1, 1))
            x_recon = kamp.solve().flatten()
        else:
            raise ValueError(f"Unknown method {method}")
        score = np.sum((x - x_recon) ** 2)
        scores.append(score)
    return np.array(scores)

def run_experiment():
    results = []

    # Load a benchmark dataset, e.g., wine
    X, y = load_odds_dataset('wine')
    normal_idx = y == 0
    X_train = X[normal_idx]
    X_test = X
    y_test = y

    methods = ['AMP', 'KAMP', 'DistributedKAMP']

    for method in methods:
        scores = run_solver(method, X_train, X_test)

        # Compute metrics
        auc_roc = roc_auc_score(y_test, scores)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        auc_pr = np.trapezoid(precision, recall)

        # Binary predictions
        threshold = np.percentile(scores, 95)
        y_pred = (scores > threshold).astype(int)

        metrics = calculate_anomaly_detection_metrics(y_test, y_pred)
        metrics['auc_roc'] = auc_roc
        metrics['auc_pr'] = auc_pr

        results.append({
            'method': method,
            **metrics
        })

    return results

def main():
    results = run_experiment()

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = 'experiments/results/exp4_anomaly_benchmark/exp4_results.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    # Save to Excel
    excel_path = 'experiments/results/exp4_anomaly_benchmark/exp4_results.xlsx'
    df.to_excel(excel_path, index=False)

    print(f"Results saved to {csv_path} and {excel_path}")

if __name__ == '__main__':
    main()