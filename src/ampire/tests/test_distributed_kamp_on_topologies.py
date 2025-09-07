# src/ampire/tests/test_distributed_kamp_on_topologies.py
import numpy as np
import os
import sys
import glob
import json
import time
import pandas as pd
import networkx as nx
from tqdm import tqdm
sys.path.insert(0, 'src')
from ampire.distributed.distributed_kamp import DistributedKAMP
from ampire.network.graph import MyGraph

def generate_synthetic_data(num_samples=900, num_features=70, sparsity=0.1, noise_std=0.1, random_state=42):
    """
    Generate synthetic data for sparse signal recovery.
    """
    rng = np.random.RandomState(random_state)
    # Sparse signal
    x_true = rng.randn(num_features)
    mask = rng.rand(num_features) < sparsity
    x_true[~mask] = 0
    # Measurement matrix
    A = rng.randn(num_samples, num_features)
    # Noise
    noise = rng.normal(0, noise_std, num_samples)
    y = A @ x_true + noise
    return A, y, x_true

def main():
    # Generate synthetic data
    A, y, x_true = generate_synthetic_data()
    num_samples, num_features = A.shape
    print(f"Generated synthetic data: {num_samples} samples, {num_features} features")
    print(f"True signal sparsity: {np.sum(x_true != 0)} non-zero elements")

    # Directory with topology plots
    topology_dir = 'topology_plots'
    adj_files = glob.glob(os.path.join(topology_dir, 'graph_*_adj.txt'))
    print(f"Found {len(adj_files)} adjacency files")

    results = []

    pbar = tqdm(adj_files, desc="Processing graphs")
    for i, adj_file in enumerate(pbar):
        pbar.set_description(f"Processing {os.path.basename(adj_file)}")
        # Load adjacency matrix
        adj = np.loadtxt(adj_file)
        num_nodes = adj.shape[0]

        # Create MyGraph
        G = MyGraph()
        G.add_nodes_from(range(num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj[i, j] == 1:
                    G.add_edge(i, j)

        # Partition data across nodes
        samples_per_node = num_samples // num_nodes
        A_list = []
        y_list = []
        start = 0
        for i in range(num_nodes):
            end = start + samples_per_node if i < num_nodes - 1 else num_samples
            A_list.append(A[start:end])
            y_list.append(y[start:end])
            start = end

        # Graph properties
        num_edges = G.number_of_edges()
        density = nx.density(G)
        is_dag = nx.is_directed_acyclic_graph(G)

        # Run DistributedKAMP
        start_time = time.time()
        dk = DistributedKAMP(
            alpha=0.5,
            tau=0.1,
            node_max_iter=50,
            num_triggers=100,
            graph=G,
            A_list=A_list,
            y_list=y_list,
            random_state=42,
            just_dag=False
        )
        dk.fit()
        fit_time = time.time() - start_time

        x_est = dk.solve()
        node_estimates = dk.get_node_estimates()

        # Compute metrics
        nmse_global = np.linalg.norm(x_est - x_true)**2 / np.linalg.norm(x_true)**2
        nmse_per_node = [np.linalg.norm(x - x_true)**2 / np.linalg.norm(x_true)**2 for x in node_estimates]
        mean_nmse_per_node = np.mean(nmse_per_node)
        std_nmse_per_node = np.std(nmse_per_node)

        # Consensus error (variance of norms)
        consensus_error = np.var([np.linalg.norm(x - x_est) for x in node_estimates])

        # Get report
        report = dk.report()

        result = {
            'graph_file': os.path.basename(adj_file),
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'is_dag': is_dag,
            'nmse_global': nmse_global,
            'mean_nmse_per_node': mean_nmse_per_node,
            'std_nmse_per_node': std_nmse_per_node,
            'consensus_error': consensus_error,
            'bytes': report['bytes'],
            'iters': report['iters'],
            'fit_time': fit_time
        }
        results.append(result)
        pbar.set_postfix({
            'NMSE': f"{nmse_global:.2f}",
            'Consensus': f"{consensus_error:.2f}",
            'Time': f"{fit_time:.2f}s"
        })

        # Save intermediate results every 10 graphs
        if (i + 1) % 10 == 0:
            df_temp = pd.DataFrame(results)
            temp_file = f'distributed_kamp_results_intermediate_{i+1}.xlsx'
            df_temp.to_excel(temp_file, index=False)
            print(f"Intermediate results saved to {temp_file}")

    pbar.close()

    # Save results to Excel
    df = pd.DataFrame(results)
    results_file = 'distributed_kamp_results.xlsx'
    df.to_excel(results_file, index=False)
    print(f"Results saved to {results_file}")

    # Also save to JSON for backup
    json_file = 'distributed_kamp_results.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"JSON backup saved to {json_file}")

if __name__ == "__main__":
    main()