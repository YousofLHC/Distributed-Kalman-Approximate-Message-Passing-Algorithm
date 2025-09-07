# src/ampire/tests/test_distributed_kamp_on_topologies.py
import numpy as np
import os
import sys
import glob
import json
import time
import logging
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
sys.path.insert(0, 'src')
from ampire.distributed.distributed_kamp import DistributedKAMP
from ampire.network.graph import MyGraph
from ampire.utils.metrics import (
    log_results, save_detailed_report, save_experiment_summary,
    calculate_compressive_sensing_metrics, calculate_ssim, calculate_gmsd,
    calculate_fsim, calculate_vif
)

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'test_distributed_kamp_errors.log'),
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

def grid_search_hyperparameters(A, y, x_true_flat, G, param_grids, is_2d=False, x_true_2d=None, random_state=42):
    """
    Perform grid search over hyperparameters for DistributedKAMP.

    Args:
        A: Measurement matrix
        y: Observations
        x_true_flat: True signal
        G: Graph
        param_grids: Dict of parameter grids
        random_state: Random seed

    Returns:
        Dict of best parameters for each metric
    """
    # Partition data
    num_nodes = G.number_of_nodes()
    samples_per_node = len(y) // num_nodes
    A_list = []
    y_list = []
    start = 0
    for i in range(num_nodes):
        end = start + samples_per_node if i < num_nodes - 1 else len(y)
        A_list.append(A[start:end])
        y_list.append(y[start:end])
        start = end

    # Generate all parameter combinations
    keys = param_grids.keys()
    values = param_grids.values()
    param_combinations = list(product(*values))

    results = []

    print("Starting grid search...")
    for params in tqdm(param_combinations, desc="Grid search"):
        param_dict = dict(zip(keys, params))
        try:
            start_time = time.time()
            dk = DistributedKAMP(
                alpha=param_dict['alpha'],
                tau=param_dict['tau'],
                node_max_iter=param_dict['node_max_iter'],
                num_triggers=param_dict['num_triggers'],
                graph=G,
                A_list=A_list,
                y_list=y_list,
                random_state=random_state,
                just_dag=False
            )
            dk.fit()
            fit_time = time.time() - start_time

            x_est = dk.solve()
            node_estimates = dk.get_node_estimates()

            # Compute compressive sensing metrics
            try:
                cs_metrics = calculate_compressive_sensing_metrics(x_true_flat, x_est)
                nmse_global = cs_metrics['nmse']
                mse_global = cs_metrics['mse']
                rmse_global = cs_metrics['rmse']
                snr_global = cs_metrics['snr']
                peak_snr_global = cs_metrics['peak_snr']
            except Exception as e:
                logging.error(f"Error computing compressive sensing metrics for params {param_dict}: {e}")
                print(f"Warning: Error computing compressive sensing metrics for params {param_dict}: {e}")
                nmse_global = mse_global = rmse_global = snr_global = peak_snr_global = np.nan

            # Compute per-node metrics
            try:
                nmse_per_node = [calculate_compressive_sensing_metrics(x_true_flat, x)['nmse'] for x in node_estimates]
                mean_nmse_per_node = np.mean(nmse_per_node)
                std_nmse_per_node = np.std(nmse_per_node)
            except Exception as e:
                logging.error(f"Error computing per-node metrics for params {param_dict}: {e}")
                print(f"Warning: Error computing per-node metrics for params {param_dict}: {e}")
                mean_nmse_per_node = std_nmse_per_node = np.nan

            # Consensus error
            try:
                consensus_error = np.var([np.linalg.norm(x - x_est) for x in node_estimates])
            except Exception as e:
                logging.error(f"Error computing consensus error for params {param_dict}: {e}")
                print(f"Warning: Error computing consensus error for params {param_dict}: {e}")
                consensus_error = np.nan

            # Image quality metrics (for 2D case)
            image_metrics = {}
            if is_2d and x_true_2d is not None:
                try:
                    x_est_2d = x_est.reshape(x_true_2d.shape)
                    image_metrics['ssim'] = calculate_ssim(x_true_2d, x_est_2d)
                    image_metrics['gmsd'] = calculate_gmsd(x_true_2d, x_est_2d)
                    image_metrics['fsim'] = calculate_fsim(x_true_2d, x_est_2d)
                    image_metrics['vif'] = calculate_vif(x_true_2d, x_est_2d)
                except Exception as e:
                    logging.error(f"Error computing image metrics for params {param_dict}: {e}")
                    print(f"Warning: Error computing image metrics for params {param_dict}: {e}")
                    image_metrics = {'ssim': np.nan, 'gmsd': np.nan, 'fsim': np.nan, 'vif': np.nan}

            result = {
                'params': param_dict,
                'nmse_global': nmse_global,
                'mse_global': mse_global,
                'rmse_global': rmse_global,
                'snr_global': snr_global,
                'peak_snr_global': peak_snr_global,
                'mean_nmse_per_node': mean_nmse_per_node,
                'std_nmse_per_node': std_nmse_per_node,
                'consensus_error': consensus_error,
                'fit_time': fit_time
            }
            result.update(image_metrics)
            results.append(result)
        except Exception as e:
            logging.error(f"Error with params {param_dict}: {e}")
            print(f"Warning: Error with params {param_dict}: {e}")
            continue

    # Find best parameters for each metric
    best_params = {}

    # Metrics to minimize
    minimize_metrics = ['nmse_global', 'mse_global', 'rmse_global', 'consensus_error', 'fit_time']
    if is_2d:
        minimize_metrics.append('gmsd')

    # Metrics to maximize
    maximize_metrics = ['snr_global', 'peak_snr_global']
    if is_2d:
        maximize_metrics.extend(['ssim', 'fsim', 'vif'])

    for metric in minimize_metrics:
        if any(metric in r for r in results):
            best = min(results, key=lambda x: x.get(metric, float('inf')))
            best_params[metric] = best['params']
            print(f"Best params for {metric}: {best['params']} with value {best[metric]:.4f}")

    for metric in maximize_metrics:
        if any(metric in r for r in results):
            best = max(results, key=lambda x: x.get(metric, -float('inf')))
            best_params[metric] = best['params']
            print(f"Best params for {metric}: {best['params']} with value {best[metric]:.4f}")

    return best_params

def main():
    try:
        # Check if topology_plots exists and has files
        topology_dir = 'topology_plots'
        if not os.path.exists(topology_dir) or not glob.glob(os.path.join(topology_dir, 'graph_*_adj.txt')):
            print("Topology plots not found. Generating topology graphs first...")
            try:
                # Run test_topology_plots.py
                os.system('python src/ampire/tests/test_topology_plots.py')
                print("Topology graphs generated.")
            except Exception as e:
                logging.error(f"Error generating topology plots: {e}")
                print(f"Warning: Error generating topology plots: {e}")
                return

        # Directory with topology plots
        adj_files = glob.glob(os.path.join(topology_dir, 'graph_*_adj.txt'))
        if not adj_files:
            print("No adjacency files found. Exiting.")
            return
    except Exception as e:
        logging.error(f"Error in main initialization: {e}")
        print(f"Warning: Error in main initialization: {e}")
        return

    print(f"Found {len(adj_files)} adjacency files")

    # Define parameter grids for grid search
    param_grids = {
        'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
        'tau': [0.01, 0.05, 0.1, 0.2, 0.5],
        'node_max_iter': [10, 20, 50, 100],
        'num_triggers': [50, 100, 200, 500]
    }

    # Load first topology for grid search
    try:
        first_adj_file = adj_files[0]
        first_adj = np.loadtxt(first_adj_file)
        first_num_nodes = first_adj.shape[0]
        first_G = MyGraph()
        first_G.add_nodes_from(range(first_num_nodes))
        for i in range(first_num_nodes):
            for j in range(first_num_nodes):
                if first_adj[i, j] == 1:
                    first_G.add_edge(i, j)
    except Exception as e:
        logging.error(f"Error loading first topology for grid search: {e}")
        print(f"Warning: Error loading first topology for grid search: {e}")
        return

    # Loop over Complex and 2D data
    for is_2d in [True, False]:
        print(f"\n{'='*50}")
        print(f"Testing with {'2D' if is_2d else 'Complex'} synthetic data")
        print(f"{'='*50}")

        try:
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
        except Exception as e:
            logging.error(f"Error generating synthetic data for {data_type}: {e}")
            print(f"Warning: Error generating synthetic data for {data_type}: {e}")
            continue

        num_samples, num_features = A.shape
        print(f"True signal sparsity: {np.sum(x_true_flat != 0)} non-zero elements")

        # Perform grid search for hyperparameter optimization
        try:
            print(f"Performing grid search for {data_type} data...")
            best_params = grid_search_hyperparameters(A, y, x_true_flat, first_G, param_grids, is_2d=is_2d, x_true_2d=x_true_2d, random_state=42)
            optimal_params = best_params['nmse_global']  # Use best for NMSE as primary metric
            print(f"Optimal parameters for {data_type}: {optimal_params}")

            # Save best parameters
            best_params_file = f'best_hyperparams_{data_type}.json'
            with open(best_params_file, 'w') as f:
                json.dump(best_params, f, indent=4)
            print(f"Best hyperparameters saved to {best_params_file}")
        except Exception as e:
            logging.error(f"Error in grid search for {data_type}: {e}")
            print(f"Warning: Error in grid search for {data_type}: {e}")
            # Use default parameters if grid search fails
            optimal_params = {'alpha': 0.5, 'tau': 0.1, 'node_max_iter': 50, 'num_triggers': 100}
            print(f"Using default parameters: {optimal_params}")

        # Create directory for plots
        data_type = '2d' if is_2d else 'Complex'
        plots_dir = f'distributed_kamp_plots_{data_type}'
        os.makedirs(plots_dir, exist_ok=True)

        results = []

        pbar = tqdm(adj_files, desc="Processing graphs")
        for i, adj_file in enumerate(pbar):
            try:
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
            except Exception as e:
                logging.error(f"Error loading topology {adj_file}: {e}")
                print(f"Warning: Error loading topology {adj_file}: {e}")
                continue

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

            # Run DistributedKAMP with optimal parameters
            try:
                start_time = time.time()
                dk = DistributedKAMP(
                    alpha=optimal_params['alpha'],
                    tau=optimal_params['tau'],
                    node_max_iter=optimal_params['node_max_iter'],
                    num_triggers=optimal_params['num_triggers'],
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
            except Exception as e:
                logging.error(f"Error running DistributedKAMP for {adj_file}: {e}")
                print(f"Warning: Error running DistributedKAMP for {adj_file}: {e}")
                continue

            # Plot original vs reconstructed
            try:
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
            except Exception as e:
                logging.error(f"Error creating/saving plot for {adj_file}: {e}")
                print(f"Warning: Error creating/saving plot for {adj_file}: {e}")

            # Compute metrics
            try:
                nmse_global = np.linalg.norm(x_est - x_true_flat)**2 / np.linalg.norm(x_true_flat)**2
                nmse_per_node = [np.linalg.norm(x - x_true_flat)**2 / np.linalg.norm(x_true_flat)**2 for x in node_estimates]
                mean_nmse_per_node = np.mean(nmse_per_node)
                std_nmse_per_node = np.std(nmse_per_node)

                # Consensus error (variance of norms)
                consensus_error = np.var([np.linalg.norm(x - x_est) for x in node_estimates])

                # Get report
                report = dk.report()
            except Exception as e:
                logging.error(f"Error computing metrics for {adj_file}: {e}")
                print(f"Warning: Error computing metrics for {adj_file}: {e}")
                nmse_global = mean_nmse_per_node = std_nmse_per_node = consensus_error = np.nan
                report = {'bytes': 0, 'iters': 0}

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
            try:
                if (i + 1) % 10 == 0:
                    df_temp = pd.DataFrame(results)
                    temp_file = f'distributed_kamp_results_{data_type}_intermediate_{i+1}.xlsx'
                    df_temp.to_excel(temp_file, index=False)
                    print(f"Intermediate results saved to {temp_file}")
            except Exception as e:
                logging.error(f"Error saving intermediate results at iteration {i+1}: {e}")
                print(f"Warning: Error saving intermediate results at iteration {i+1}: {e}")

        pbar.close()

        # Save results to Excel
        try:
            df = pd.DataFrame(results)
            results_file = f'distributed_kamp_results_{data_type}.xlsx'
            df.to_excel(results_file, index=False)
            print(f"Results for {data_type} saved to {results_file}")
        except Exception as e:
            logging.error(f"Error saving Excel results for {data_type}: {e}")
            print(f"Warning: Error saving Excel results for {data_type}: {e}")

        # Also save to JSON for backup
        try:
            json_file = f'distributed_kamp_results_{data_type}.json'
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"JSON backup for {data_type} saved to {json_file}")
        except Exception as e:
            logging.error(f"Error saving JSON results for {data_type}: {e}")
            print(f"Warning: Error saving JSON results for {data_type}: {e}")

        # Save report using metrics.py log_results function
        try:
            log_file = f'distributed_kamp_results_{data_type}.log'
            # Format results for log_results function (using NMSE instead of AUC-ROC for compressive sensing)
            formatted_results = {}
            for i, result in enumerate(results):
                method_name = f"{result['graph_file']}_nodes{result['num_nodes']}"
                formatted_results[method_name] = {
                    'auc_roc': result.get('nmse_global', 0),  # Using NMSE as primary metric
                    'time': result.get('fit_time', 0)
                }
            log_results(formatted_results, filepath=log_file)
            print(f"Log report for {data_type} saved to {log_file}")
        except Exception as e:
            logging.error(f"Error saving log report for {data_type}: {e}")
            print(f"Warning: Error saving log report for {data_type}: {e}")

        # Save detailed comprehensive report
        try:
            detailed_report_file = f'results/detailed_report_{data_type}.txt'
            save_detailed_report(results, filepath=detailed_report_file, format_type='txt')
            print(f"Detailed text report for {data_type} saved to {detailed_report_file}")
        except Exception as e:
            logging.error(f"Error saving detailed text report for {data_type}: {e}")
            print(f"Warning: Error saving detailed text report for {data_type}: {e}")

        # Save detailed report in CSV format
        try:
            csv_report_file = f'results/detailed_report_{data_type}.csv'
            save_detailed_report(results, filepath=csv_report_file, format_type='csv')
            print(f"Detailed CSV report for {data_type} saved to {csv_report_file}")
        except Exception as e:
            logging.error(f"Error saving detailed CSV report for {data_type}: {e}")
            print(f"Warning: Error saving detailed CSV report for {data_type}: {e}")

        # Save detailed report in JSON format
        try:
            json_report_file = f'results/detailed_report_{data_type}.json'
            save_detailed_report(results, filepath=json_report_file, format_type='json')
            print(f"Detailed JSON report for {data_type} saved to {json_report_file}")
        except Exception as e:
            logging.error(f"Error saving detailed JSON report for {data_type}: {e}")
            print(f"Warning: Error saving detailed JSON report for {data_type}: {e}")

        # Save experiment summary
        try:
            summary_file = f'results/experiment_summary_{data_type}.txt'
            save_experiment_summary(results, filepath=summary_file)
            print(f"Experiment summary for {data_type} saved to {summary_file}")
        except Exception as e:
            logging.error(f"Error saving experiment summary for {data_type}: {e}")
            print(f"Warning: Error saving experiment summary for {data_type}: {e}")

if __name__ == "__main__":
    main()