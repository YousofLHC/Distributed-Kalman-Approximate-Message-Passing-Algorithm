import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_recall_curve
import sys
import os
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ampire.core.kamp import KAMP
from ampire.utils.metrics import calculate_anomaly_detection_metrics
from ..datasets.benchmark_loaders import load_odds_dataset  # Assuming this exists

def run_experiment():
    results = []

    # Load a benchmark dataset, e.g., Wine
    X, y = load_odds_dataset('Wine')
    normal_idx = y == 0
    X_train = X[normal_idx]
    X_test = X
    y_test = y

    # Methods
    methods = ['KAMP']  # Prioritize KAMP
    # Other methods (deactivated for now):
    # methods = ['KAMP', 'OCSVM']

    for method in methods:
        if method == 'KAMP':
            # Use KAMP for anomaly detection (simplified: reconstruction error)
            kamp = KAMP(alpha=0.5, tau=0.1, max_iter=100)
            kamp.fit(np.eye(X_train.shape[1]), X_train.T)  # Identity for simplicity
            X_recon = kamp.solve().T
            scores = np.sum((X_test - X_recon) ** 2, axis=1)  # Reconstruction error
        elif method == 'OCSVM':
            ocsvm = OneClassSVM()
            ocsvm.fit(X_train)
            scores = ocsvm.decision_function(X_test)

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