import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ampire.distributed.distributed_kamp import DKFAMP
from ampire.network.random_digraphs import generate_dag  # Assuming this exists
from ampire.utils.metrics import calculate_compressive_sensing_metrics

def run_experiment():
    results = []

    # Parameters
    n = 500  # Signal dimension
    m = 250  # Measurements per node
    num_nodes = 10
    topologies = ['dag', 'ring', 'self-loop']
    message_sizes = [10, 50, 100]
    consensus_errors = [0.01, 0.05, 0.1]

    # Generate sparse signal
    k = 50
    x = np.zeros(n)
    support = np.random.choice(n, k, replace=False)
    x[support] = np.random.randn(k)

    for topology in tqdm(topologies, desc="Topology"):
        for msg_size in tqdm(message_sizes, desc="Message Size", leave=False):
            for cons_error in tqdm(consensus_errors, desc="Consensus Error", leave=False):
                # Generate network
                if topology == 'dag':
                    graph = generate_dag(num_nodes)
                elif topology == 'ring':
                    # Simple ring
                    graph = {i: [(i-1)%num_nodes, (i+1)%num_nodes] for i in range(num_nodes)}
                elif topology == 'self-loop':
                    graph = {i: [i] for i in range(num_nodes)}

                # Simulate distributed measurements
                measurements = []
                for i in range(num_nodes):
                    A_i = np.random.randn(m, n) / np.sqrt(m)
                    y_i = A_i @ x + 0.01 * np.random.randn(m)
                    measurements.append((A_i, y_i))

                # Run DistributedKAMP
                A_list = [A for A, y in measurements]
                y_list = [y for A, y in measurements]
                dkfamp = DistributedKAMP(alpha=0.5, tau=0.1, node_max_iter=50, num_triggers=100, graph=graph, A_list=A_list, y_list=y_list)
                dkfamp.fit()
                x_hat = dkfamp.solve()

                # Compute NMSE
                metrics = calculate_compressive_sensing_metrics(x, x_hat)
                nmse = metrics['nmse']

                results.append({
                    'topology': topology,
                    'message_size': msg_size,
                    'consensus_error': cons_error,
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