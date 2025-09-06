import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ampire.core.kamp import KAMP
from ampire.core.amp import AMP
from ampire.distributed.distributed_kamp import DistributedKAMP
from ampire.network.random_digraphs import create_strongly_connected_graph
import os
from ampire.utils.metrics import calculate_compressive_sensing_metrics

def run_solver(method, measurements, graph, x):
    if method == 'DistributedKAMP':
        A_list = [A for A, y in measurements]
        y_list = [y for A, y in measurements]
        dkamp = DistributedKAMP(alpha=0.5, tau=0.1, node_max_iter=50, num_triggers=100, graph=graph, A_list=A_list, y_list=y_list)
        dkamp.fit()
        x_hat = dkamp.solve()
    elif method in ['AMP', 'KAMP']:
        # Centralized: concatenate measurements
        A_full = np.vstack([A for A, y in measurements])
        y_full = np.concatenate([y for A, y in measurements])
        if method == 'AMP':
            solver = AMP(alpha=0.5, tau=0.1, max_iter=100)
        else:
            solver = KAMP(alpha=0.5, tau=0.1, max_iter=100)
        solver.fit(A_full, y_full.reshape(-1, 1))
        x_hat = solver.solve()
    else:
        raise ValueError(f"Unknown method {method}")
    return x_hat

def run_experiment():
    results = []

    # Parameters
    n = 500  # Signal dimension
    m = 250  # Measurements per node
    num_nodes = 10
    topologies = ['dag', 'ring', 'self-loop']
    message_sizes = [10, 50, 100]
    consensus_errors = [0.01, 0.05, 0.1]
    methods = ['DistributedKAMP', 'AMP', 'KAMP']

    # Generate sparse signal
    k = 50
    x = np.zeros(n)
    support = np.random.choice(n, k, replace=False)
    x[support] = np.random.randn(k)

    for topology in tqdm(topologies, desc="Topology"):
        for msg_size in tqdm(message_sizes, desc="Message Size", leave=False):
            for cons_error in tqdm(consensus_errors, desc="Consensus Error", leave=False):
                # Generate sophisticated network topology
                if topology == 'dag':
                    graph = DistributedKAMP.create_dag(num_nodes, edge_prob=0.4, random_state=42)
                elif topology == 'ring':
                    # Sophisticated ring with additional connections
                    graph = nx.DiGraph()
                    graph.add_nodes_from(range(num_nodes))
                    for i in range(num_nodes):
                        graph.add_edge(i, (i + 1) % num_nodes)
                        # Add skip connections for sophistication
                        if num_nodes > 4:
                            graph.add_edge(i, (i + 2) % num_nodes)
                elif topology == 'self-loop':
                    graph = nx.DiGraph()
                    graph.add_nodes_from(range(num_nodes))
                    for i in range(num_nodes):
                        graph.add_edge(i, i)
                        # Add some cross connections
                        if i < num_nodes - 1:
                            graph.add_edge(i, i + 1)

                # Plot and save topology
                out_dir = 'experiments/results/exp5_distributed_sim'
                os.makedirs(out_dir, exist_ok=True)

                # Save adjacency matrix
                adj_matrix = nx.to_numpy_array(graph)
                adj_path = os.path.join(out_dir, f'topology_{topology}_adjacency.npy')
                np.save(adj_path, adj_matrix)

                # Plot topology (save to file)
                plt.figure(figsize=(10, 8))
                pos = nx.spring_layout(graph, seed=42)
                nx.draw(graph, pos, with_labels=True, node_color='lightblue',
                       node_size=500, font_size=16, font_weight='bold',
                       arrows=True, arrowstyle='->', arrowsize=20)
                plt.title(f'Distributed KAMP Topology: {topology.upper()}')
                plot_path = os.path.join(out_dir, f'topology_{topology}_graph.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()

                # Simulate distributed measurements
                measurements = []
                for i in range(num_nodes):
                    A_i = np.random.randn(m, n) / np.sqrt(m)
                    y_i = A_i @ x + 0.01 * np.random.randn(m)
                    measurements.append((A_i, y_i))

                for method in methods:
                    x_hat = run_solver(method, measurements, graph, x)

                    # Compute NMSE
                    metrics = calculate_compressive_sensing_metrics(x, x_hat)
                    nmse = metrics['nmse']

                    results.append({
                        'topology': topology,
                        'message_size': msg_size,
                        'consensus_error': cons_error,
                        'method': method,
                        'nmse': nmse
                    })

    return results

def main():
    results = run_experiment()

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = 'experiments/results/exp5_distributed_sim/exp5_results.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    # Save to Excel
    excel_path = 'experiments/results/exp5_distributed_sim/exp5_results.xlsx'
    df.to_excel(excel_path, index=False)

    print(f"Results saved to {csv_path} and {excel_path}")

if __name__ == '__main__':
    main()