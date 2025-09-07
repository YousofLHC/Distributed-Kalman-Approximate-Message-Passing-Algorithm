# src/ampire/tests/test_distributed_kamp_on_topologies.py
import numpy as np
import os
import sys
import glob
import json
import time
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.insert(0, 'src')
from ampire.distributed.distributed_kamp import DistributedKAMP
from ampire.network.graph import MyGraph

def generate_synthetic_data(num_samples=900, num_features=70, sparsity=0.1, noise_std=0.1, random_state=42, is_2d=False, shape_2d=(10, 7)):
    """
    Generate synthetic data for sparse signal recovery.
    If is_2d=True, generate 2D sparse data.
    """
    rng = np.random.RandomState(random_state)
    if is_2d:
        # 2D sparse signal
        x_true = rng.randn(*shape_2d)
        mask = rng.rand(*shape_2d) < sparsity
        x_true[~mask] = 0
        x_true_flat = x_true.flatten()
        num_features = x_true_flat.shape[0]
    else:
        # Complex sparse signal
        x_true = rng.randn(num_features)
        mask = rng.rand(num_features) < sparsity
        x_true[~mask] = 0
        x_true_flat = x_true
        x_true = None

    # Measurement matrix
    A = rng.randn(num_samples, num_features)
    # Noise
    noise = rng.normal(0, noise_std, num_samples)
    y = A @ x_true_flat + noise
    return A, y, x_true_flat, x_true

def main():
    # Check if topology_plots exists and has files
    topology_dir = 'topology_plots'
    if not os.path.exists(topology_dir) or not glob.glob(os.path.join(topology_dir, 'graph_*_adj.txt')):
        print("Topology plots not found. Generating topology graphs first...")
        # Run test_topology_plots.py
        os.system('python src/ampire/tests/test_topology_plots.py')
        print("Topology graphs generated.")

    # Directory with topology plots
    adj_files = glob.glob(os.path.join(topology_dir, 'graph_*_adj.txt'))
    if not adj_files:
        print("No adjacency files found. Exiting.")
        return

    print(f"Found {len(adj_files)} adjacency files")

    # Loop over Complex and 2D data
    for is_2d in [False, True]:
        print(f"\n{'='*50}")
        print(f"Testing with {'2D' if is_2d else 'Complex'} synthetic data")
        print(f"{'='*50}")

        num_samples = 900
        if is_2d:
            num_features = 2
            shape_2d = (1, num_features)
            A, y, x_true_flat, x_true_2d = generate_synthetic_data(num_samples=num_samples, num_features=num_features, is_2d=is_2d, shape_2d=shape_2d)
            print(f"Generated 2D synthetic data: {A.shape[0]} samples, {A.shape[1]} features ({shape_2d[0]}x{shape_2d[1]})")
        else:
            num_features = 70
            A, y, x_true_flat, x_true_2d = generate_synthetic_data(num_samples=num_samples, num_features=num_features, is_2d=is_2d)
            print(f"Generated Complex synthetic data: {A.shape[0]} samples, {A.shape[1]} features")

        num_samples, num_features = A.shape
        print(f"True signal sparsity: {np.sum(x_true_flat != 0)} non-zero elements")

        # Create directory for plots
        data_type = '2d' if is_2d else '1d'
        plots_dir = f'distributed_kamp_plots_{data_type}'
        os.makedirs(plots_dir, exist_ok=True)

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

            # Plot original vs reconstructed
            if is_2d:
                # Reshape to 2D for plotting
                x_est_2d = x_est.reshape(shape_2d)
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(x_true_2d, cmap='viridis')
                axes[0].set_title('Original 2D Signal')
                axes[0].axis('off')
                axes[1].imshow(x_est_2d, cmap='viridis')
                axes[1].set_title('Reconstructed 2D Signal')
                axes[1].axis('off')
            else:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].plot(x_true_flat, label='Original')
                axes[0].set_title('Original Complex Signal')
                axes[1].plot(x_est, label='Reconstructed')
                axes[1].set_title('Reconstructed Complex Signal')

            nmse_placeholder = np.linalg.norm(x_est - x_true_flat)**2 / np.linalg.norm(x_true_flat)**2
            plt.suptitle(f"{os.path.basename(adj_file)} - NMSE: {nmse_placeholder:.4f}")
            plot_file = os.path.join(plots_dir, f"{os.path.basename(adj_file).replace('.txt', '')}_reconstruction.png")
            plt.savefig(plot_file)
            plt.close()

            # Compute metrics
            nmse_global = np.linalg.norm(x_est - x_true_flat)**2 / np.linalg.norm(x_true_flat)**2
            nmse_per_node = [np.linalg.norm(x - x_true_flat)**2 / np.linalg.norm(x_true_flat)**2 for x in node_estimates]
            mean_nmse_per_node = np.mean(nmse_per_node)
            std_nmse_per_node = np.std(nmse_per_node)

            # Consensus error (variance of norms)
            consensus_error = np.var([np.linalg.norm(x - x_est) for x in node_estimates])

            # Get report
            report = dk.report()

            result = {
                'data_type': data_type,
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
                temp_file = f'distributed_kamp_results_{data_type}_intermediate_{i+1}.xlsx'
                df_temp.to_excel(temp_file, index=False)
                print(f"Intermediate results saved to {temp_file}")

        pbar.close()

        # Save results to Excel
        df = pd.DataFrame(results)
        results_file = f'distributed_kamp_results_{data_type}.xlsx'
        df.to_excel(results_file, index=False)
        print(f"Results for {data_type} saved to {results_file}")

        # Also save to JSON for backup
        json_file = f'distributed_kamp_results_{data_type}.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"JSON backup for {data_type} saved to {json_file}")

if __name__ == "__main__":
    main()