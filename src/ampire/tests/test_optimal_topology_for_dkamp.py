import numpy as np
import os
import glob
import json
import time
import logging
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from sklearn.metrics import mean_squared_error
from ampire.core.amp import AMP
from ampire.core.kamp import KAMP
from ampire.distributed.distributed_kamp import DistributedKAMP
from ampire.network.graph import MyGraph
from ampire.utils.metrics import safe_divide, DEFAULT_EPS
from ampire.utils.helpers import generate_synthetic_data,load_graph, evaluate_model,save_plot
# -------------------- Setup logging ----------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'test_optimal_topology_for_dkamp.log'),
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)



def grid_search_hyperparameters(A, y, x_true_flat, G, param_grids, random_state=42):
    """Perform grid search for DistributedKAMP hyperparameters."""
    num_nodes = G.number_of_nodes()
    samples_per_node = len(y) // num_nodes
    A_list = [A[start:start + samples_per_node if i < num_nodes - 1 else len(y)]
              for i, start in enumerate(range(0, len(y), samples_per_node))]
    y_list = [y[start:start + samples_per_node if i < num_nodes - 1 else len(y)]
              for i, start in enumerate(range(0, len(y), samples_per_node))]

    param_combinations = list(product(*param_grids.values()))
    results = []

    print("Starting grid search...")
    for params in tqdm(param_combinations, desc="Grid search"):
        param_dict = dict(zip(param_grids.keys(), params))
        try:
            start_time = time.time()
            dk = DistributedKAMP(**param_dict, graph=G, A_list=A_list, y_list=y_list, random_state=random_state, just_dag=False)
            dk.fit()
            x_est = dk.solve()
            node_estimates = dk.get_node_estimates()

            mse_global = mean_squared_error(x_true_flat, x_est)
            mse_per_node = [mean_squared_error(x_true_flat, x) for x in node_estimates]
            result = {
                'params': param_dict,
                'mse_global': mse_global,
                'mean_mse_per_node': np.mean(mse_per_node),
                'std_mse_per_node': np.std(mse_per_node),
                'consensus_error': np.var([np.linalg.norm(x - x_est) for x in node_estimates]),
                'fit_time': time.time() - start_time
            }
            results.append(result)
        except Exception as e:
            logging.error(f"Error with params {param_dict}: {e}")
            print(f"Warning: Error with params {param_dict}: {e}")

    return min(results, key=lambda x: x.get('mse_global', float('inf')))['params'] if results else None



def main():
    # Common parameters
    config = {
        'num_samples': 50,
        'noise_std': 0.01,
        'random_state': 42,
        'sparsity_levels': [0.1, 0.35, 0.9],
        'param_grids': {
            'alpha': [0.1, 0.5, 0.9],
            'tau': [0.01, 0.1, 0.5],
            'node_max_iter': [10, 50],
            'num_triggers': [50, 200]
        },
        'data_types': [
            {'is_2d': False, 'name': 'complex_signal', 'num_features': 100, 'shape_2d': None},
            {'is_2d': True, 'name': '2d_signal', 'num_features': 2, 'shape_2d': (1, 2)}
        ]
    }

    # Setup directories
    root = 'ThesisExperiments/DKAMP'
    base_plot_dir = f'{root}/plots'
    os.makedirs(base_plot_dir, exist_ok=True)

    # Check topology files
    topology_dir = 'topology_plots'
    if not os.path.exists(topology_dir) or not glob.glob(os.path.join(topology_dir, 'graph_*_adj.txt')):
        print("Topology plots not found. Generating topology graphs...")
        try:
            os.system('python src/ampire/tests/test_topology_plots.py')
            print("Topology graphs generated.")
        except Exception as e:
            logging.error(f"Error generating topology plots: {e}")
            print(f"Warning: Error generating topology plots: {e}")
            return
    adj_files = glob.glob(os.path.join(topology_dir, 'graph_*_adj.txt'))
    if not adj_files:
        print("No adjacency files found. Exiting.")
        return
    print(f"Found {len(adj_files)} adjacency files")

    # Load first topology for grid search
    first_G, _ = load_graph(adj_files[0])
    if first_G is None:
        return

    all_results = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for data_type in config['data_types']:
        is_2d = data_type['is_2d']
        data_name = data_type['name']
        num_features = data_type['num_features']
        shape_2d = data_type['shape_2d']

        print(f"\n{'='*60}\nTesting with {data_name} synthetic data\n{'='*60}")

        for sparsity in config['sparsity_levels']:
            sparsity_percent = int(sparsity * 100)
            print(f"\nTesting with {sparsity_percent}% non-zero elements")

            # Generate data
            try:
                A, y, x_true_flat, x_true_2d = generate_synthetic_data(
                    num_samples=config['num_samples'],
                    num_features=num_features,
                    sparsity=sparsity,
                    noise_std=config['noise_std'],
                    random_state=config['random_state'],
                    is_2d=is_2d,
                    shape_2d=shape_2d
                )
                print(f"Generated {data_name} data: {A.shape[0]} samples, {A.shape[1]} features")
                non_zero_count = np.sum(x_true_flat != 0)
                print(f"True signal sparsity: {non_zero_count} non-zero elements ({non_zero_count / len(x_true_flat):.1%})")
            except Exception as e:
                logging.error(f"Error generating data for {data_name} with {sparsity_percent}% sparsity: {e}")
                print(f"Warning: Error generating data for {data_name} with {sparsity_percent}% sparsity: {e}")
                continue

            # Grid search for DistributedKAMP
            optimal_params = grid_search_hyperparameters(A, y, x_true_flat, first_G, config['param_grids'], config['random_state'])
            if not optimal_params:
                optimal_params = {'alpha': 0.5, 'tau': 0.1, 'node_max_iter': 50, 'num_triggers': 100}
                print(f"Grid search failed, using default parameters: {optimal_params}")

            # Setup results directory
            results_base_dir = f'{root}/results/distributed_kamp_modes_{data_name}_sparsity_{sparsity_percent}'
            plots_dir = base_plot_dir
            os.makedirs(plots_dir, exist_ok=True)

            # Process topologies
            for adj_file in tqdm(adj_files, desc="Processing graphs"):
                G, num_nodes = load_graph(adj_file)
                if G is None:
                    continue

                is_dag = nx.is_directed_acyclic_graph(G)
                samples_per_node = config['num_samples'] // num_nodes
                A_list = [A[start:start + samples_per_node if i < num_nodes - 1 else len(y)]
                          for i, start in enumerate(range(0, len(y), samples_per_node))]
                y_list = [y[start:start + samples_per_node if i < num_nodes - 1 else len(y)]
                          for i, start in enumerate(range(0, len(y), samples_per_node))]

                for just_dag, mode_name in [(False, 'general'), (True, 'dag')]:
                    if just_dag and not is_dag:
                        continue

                    try:
                        start_time = time.time()
                        dkamp = DistributedKAMP(
                            **optimal_params, graph=G, A_list=A_list, y_list=y_list,
                            random_state=config['random_state'], just_dag=just_dag, record_history=True, verbose=False
                        )
                        dkamp.fit()
                        fit_time = time.time() - start_time

                        x_est = dkamp.solve().flatten()
                        node_estimates = dkamp.get_node_estimates()
                        mse_global = mean_squared_error(x_true_flat, x_est)
                        mse_per_node = [mean_squared_error(x_true_flat, x.flatten()) for x in node_estimates]
                        consensus_error = np.var([np.linalg.norm(x.flatten() - x_est) for x in node_estimates])
                        nmse_global = safe_divide(mse_global, np.var(x_true_flat), mode="epsilon", epsilon=DEFAULT_EPS) if np.var(x_true_flat) != 0 else mse_global

                        print(f"\n=== DistributedKAMP - Topology: {os.path.basename(adj_file)}, Mode: {mode_name} ===")
                        print(f"MSE Global: {mse_global:.6f}, Mean MSE per Node: {np.mean(mse_per_node):.6f}")
                        print(f"Consensus Error: {consensus_error:.6f}, Fit Time: {fit_time:.2f}s")

                        all_results.append({
                            'data_type': data_name,
                            'sparsity_percent': sparsity_percent,
                            'topology': os.path.basename(adj_file),
                            'mode': mode_name,
                            'num_nodes': num_nodes,
                            'is_dag': is_dag,
                            'mse_global': mse_global,
                            'mean_mse_per_node': np.mean(mse_per_node),
                            'std_mse_per_node': np.std(mse_per_node),
                            'consensus_error': consensus_error,
                            'fit_time': fit_time
                        })

                        # Plotting
                        base_filename = f"{os.path.basename(adj_file).replace('.txt', '')}_{mode_name}_sparsity_{sparsity_percent}"

                        # Node-wise Recovery
                        fig = plt.figure(figsize=(12, 6))
                        plt.plot(x_true_flat, label='True Signal', color='black', linewidth=2)
                        for i, x_node in enumerate(node_estimates):
                            plt.plot(x_node.flatten(), label=f'Node {i}', linestyle='--')
                        save_plot(fig, os.path.join(plots_dir, f'nodewise_recovery_{base_filename}.png'),
                                  f'Node-wise Recovery - {base_filename}', 'Index', 'Signal Value')

                        # Consensus Error
                        if hasattr(dkamp, 'history') and dkamp.history:
                            fig = plt.figure(figsize=(10, 5))
                            plt.plot(dkamp.history, label='Consensus Error', color='blue')
                            save_plot(fig, os.path.join(plots_dir, f'consensus_error_{base_filename}.png'),
                                      f'Consensus Error - {base_filename}', 'Trigger Iteration', 'Consensus Error', log_scale=True)

                        # Graph Visualization
                        dkamp.plot_graph(show=False)
                        plt.savefig(os.path.join(plots_dir, f'graph_{base_filename}.png'), bbox_inches='tight')
                        plt.close()

                    except Exception as e:
                        logging.error(f"Error running DistributedKAMP for {adj_file} in mode {mode_name}: {e}")
                        print(f"Warning: Error running DistributedKAMP for {adj_file} in mode {mode_name}: {e}")

            # Save sparsity results
            df_sparsity = pd.DataFrame([r for r in all_results if r['sparsity_percent'] == sparsity_percent and r['data_type'] == data_name])
            results_file = f'{results_base_dir}/distributed_kamp_modes_{data_name}_sparsity_{sparsity_percent}_{timestamp}.xlsx'
            df_sparsity.to_excel(results_file, index=False)
            print(f"Results for {data_name} with {sparsity_percent}% sparsity saved to {results_file}")

    # Save all results
    df_all = pd.DataFrame(all_results)
    all_results_file = f'{root}/results/distributed_kamp_all_modes_{timestamp}.xlsx'
    df_all.to_excel(all_results_file, index=False)
    print(f"All results saved to {all_results_file}")
    
if __name__ == "__main__":
    main()