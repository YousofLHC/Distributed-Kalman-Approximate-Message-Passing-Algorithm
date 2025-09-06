import yaml
import pandas as pd
import numpy as np
import networkx as nx
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ampire.utils.metrics import calculate_anomaly_detection_metrics
from ampire.core.kamp import KAMP
from ampire.core.amp import AMP
from ampire.distributed.distributed_kamp import DistributedKAMP  # Assuming DKF-AMP class
from experiments.datasets.benchmark_loaders import load_odds_dataset, load_keel_dataset  # Assuming these functions exist

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

        methods_to_run = ['AMP', 'KAMP', 'DKF-AMP']  # Add AMP and KAMP
        for method_name in methods_to_run:
            print(f"  Running {method_name}")

            # Initialize model
            if method_name == 'DKF-AMP':
                # Create single-node graph
                G = nx.DiGraph()
                G.add_node(0)
                model = DistributedKAMP(alpha=0.5, tau=0.1, node_max_iter=100, num_triggers=10, graph=G, A_list=[np.eye(X_train.shape[1])], y_list=[X_train.T])
                model.fit()

                # Plot and save topology
                out_dir = 'experiments/results/exp6_anomaly_dkf_amp'
                os.makedirs(out_dir, exist_ok=True)

                # Save adjacency matrix
                adj_matrix = nx.to_numpy_array(G)
                adj_path = os.path.join(out_dir, 'topology_single_node_adjacency.npy')
                np.save(adj_path, adj_matrix)

                # Plot topology
                plt.figure(figsize=(6, 6))
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, with_labels=True, node_color='lightblue',
                       node_size=500, font_size=16, font_weight='bold')
                plt.title('Distributed KAMP Topology: Single Node')
                plot_path = os.path.join(out_dir, 'topology_single_node_graph.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()

                X_recon = model.solve().T
                scores = np.sum((X_test - X_recon) ** 2, axis=1)
            elif method_name == 'KAMP':
                kamp = KAMP(alpha=0.5, tau=0.1, max_iter=100)
                kamp.fit(np.eye(X_train.shape[1]), X_train.T)
                X_recon = kamp.solve().T
                scores = np.sum((X_test - X_recon) ** 2, axis=1)
            elif method_name == 'AMP':
                amp = AMP(alpha=0.5, tau=0.1, max_iter=100)
                amp.fit(np.eye(X_train.shape[1]), X_train.T)
                X_recon = amp.solve().T
                scores = np.sum((X_test - X_recon) ** 2, axis=1)
            elif method_name == 'OCSVM':
                model = OneClassSVM()
                model.fit(X_train)
                scores = model.decision_function(X_test)
            elif method_name == 'LOF':
                model = LocalOutlierFactor()
                scores = model.fit_predict(X_test)
                scores = -scores
            elif method_name == 'IF':
                model = IsolationForest()
                model.fit(X_train)
                scores = -model.decision_function(X_test)
            elif method_name == 'EE':
                model = EllipticEnvelope()
                model.fit(X_train)
                scores = -model.decision_function(X_test)
            elif method_name == 'SVDD':
                model = OneClassSVM()
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