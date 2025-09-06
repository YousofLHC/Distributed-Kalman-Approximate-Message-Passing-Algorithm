import yaml
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_auc_score, precision_recall_curve
import sys
import os
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ampire.utils.metrics import calculate_anomaly_detection_metrics
from ampire.distributed.distributed_kamp import DKFAMP  # Assuming DKF-AMP class
from ..datasets.benchmark_loaders import load_odds_dataset, load_keel_dataset  # Assuming these functions exist

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_experiment(config):
    results = []

    # Load datasets
    datasets = []
    for ds in config['datasets']['odds']:
        datasets.append(('ODDS', ds, load_odds_dataset(ds)))
    for ds in config['datasets']['keel']:
        datasets.append(('KEEL', ds, load_keel_dataset(ds)))

    for repo, name, (X, y) in tqdm(datasets, desc="Datasets"):
        print(f"Processing {repo} - {name}")

        # Split into train/test (assuming y has labels, use normal for train)
        normal_idx = y == 0  # Assuming 0 is normal
        X_train = X[normal_idx]
        X_test = X
        y_test = y

        for method_config in [config['methods']['proposed']]:  # Prioritize DKF-AMP
            # Other baselines (deactivated for now):
            # for method_config in [config['methods']['proposed']] + config['methods']['baselines']:
            method_name = method_config['name']
            print(f"  Running {method_name}")

            # Initialize model
            if method_name == 'DKF-AMP':
                model = DKFAMP(**{k: v[0] if isinstance(v, list) else v for k, v in method_config['hyperparameters'].items() if k != 'kernels' and k != 'fold'})
                # Fit on normal data
                model.fit(X_train)
                scores = model.decision_function(X_test)
            elif method_name == 'OCSVM':
                model = OneClassSVM(**{k: v[0] if isinstance(v, list) else v for k, v in method_config.items() if k != 'name'})
                model.fit(X_train)
                scores = model.decision_function(X_test)
            elif method_name == 'LOF':
                model = LocalOutlierFactor(**{k: v[0] if isinstance(v, list) else v for k, v in method_config.items() if k != 'name'})
                scores = model.fit_predict(X_test)  # LOF gives labels, convert to scores
                scores = -scores  # Negative for anomaly scores
            elif method_name == 'IF':
                model = IsolationForest(**{k: v[0] if isinstance(v, list) else v for k, v in method_config.items() if k != 'name'})
                model.fit(X_train)
                scores = -model.decision_function(X_test)  # Negative for anomaly scores
            elif method_name == 'EE':
                model = EllipticEnvelope(**{k: v[0] if isinstance(v, list) else v for k, v in method_config.items() if k != 'name'})
                model.fit(X_train)
                scores = -model.decision_function(X_test)
            elif method_name == 'SVDD':
                # Assuming SVDD is implemented similarly to OCSVM
                model = OneClassSVM(**{k: v[0] if isinstance(v, list) else v for k, v in method_config.items() if k != 'name'})
                model.fit(X_train)
                scores = model.decision_function(X_test)

            # Compute metrics
            auc_roc = roc_auc_score(y_test, scores)
            precision, recall, _ = precision_recall_curve(y_test, scores)
            auc_pr = np.trapezoid(precision, recall)

            # Binary predictions for other metrics
            threshold = np.percentile(scores, 95)  # Example threshold
            y_pred = (scores > threshold).astype(int)

            metrics = calculate_anomaly_detection_metrics(y_test, y_pred)
            metrics['auc_roc'] = auc_roc
            metrics['auc_pr'] = auc_pr

            results.append({
                'repo': repo,
                'dataset': name,
                'method': method_name,
                **metrics
            })

    return results

def main():
    config_path = 'experiments/configs/exp6_config.yml'
    config = load_config(config_path)

    results = run_experiment(config)

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(config['output']['results_dir'], 'exp6_results.csv')
    df.to_csv(csv_path, index=False)

    # Save to Excel
    excel_path = os.path.join(config['output']['results_dir'], 'exp6_results.xlsx')
    df.to_excel(excel_path, index=False)

    print(f"Results saved to {csv_path} and {excel_path}")

if __name__ == '__main__':
    main()