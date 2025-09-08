import numpy as np
import os
import glob
import sys
import time
import logging
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from sklearn.metrics import mean_squared_error
sys.path.insert(0, 'src')
from ampire.distributed.distributed_kamp import DistributedKAMP
from ampire.network.graph import MyGraph
from ampire.utils.metrics import safe_divide, DEFAULT_EPS, calculate_compressive_sensing_metrics, calculate_ssim, calculate_gmsd, calculate_fsim, calculate_vif
from ampire.utils.helpers import generate_synthetic_data, load_graph, save_plot

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'test_optimal_topology_for_dkamp.log'),
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def grid_search_hyperparameters(A, y, x_true_flat, x_true_2d, G, param_grids, is_2d=False, random_state=42):
    """Perform grid search for DistributedKAMP hyperparameters, evaluating multiple metrics."""
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
            fit_time = time.time() - start_time

            # Compressive sensing metrics
            cs_metrics = calculate_compressive_sensing_metrics(x_true_flat, x_est)
            nmse_global = cs_metrics.get('nmse', np.nan)
            mse_global = cs_metrics.get('mse', np.nan)
            rmse_global = cs_metrics.get('rmse', np.nan)
            snr_global = cs_metrics.get('snr', np.nan)
            peak_snr_global = cs_metrics.get('peak_snr', np.nan)

            # Per-node NMSE
            nmse_per_node = [calculate_compressive_sensing_metrics(x_true_flat, x).get('nmse', np.nan) for x in node_estimates]
            mean_nmse_per_node = np.mean(nmse_per_node)
            std_nmse_per_node = np.std(nmse_per_node)

            # Consensus error
            consensus_error = np.var([np.linalg.norm(x - x_est) for x in node_estimates])

            # Image quality metrics for 2D data
            image_metrics = {}
            if is_2d and x_true_2d is not None:
                try:
                    x_est_2d = x_est.reshape(x_true_2d.shape)
                    image_metrics = {
                        'ssim': calculate_ssim(x_true_2d, x_est_2d),
                        'gmsd': calculate_gmsd(x_true_2d, x_est_2d),
                        'fsim': calculate_fsim(x_true_2d, x_est_2d),
                        'vif': calculate_vif(x_true_2d, x_est_2d)
                    }
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

    # Find best parameters for each metric
    best_params = {}
    minimize_metrics = ['nmse_global', 'mse_global', 'rmse_global', 'consensus_error', 'fit_time']
    maximize_metrics = ['snr_global', 'peak_snr_global']
    if is_2d:
        minimize_metrics.append('gmsd')
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

    return best_params.get('nmse_global', None) if results else None

def main():
    # Configuration
    config = {
        'num_samples': 900,
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
            {'is_2d': False, 'name': 'complex_signal', 'num_features': 700, 'shape_2d': None},
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
            optimal_params = grid_search_hyperparameters(A, y, x_true_flat, x_true_2d, first_G, config['param_grids'], is_2d, config['random_state'])
            if not optimal_params:
                optimal_params = {'alpha': 0.5, 'tau': 0.1, 'node_max_iter': 50, 'num_triggers': 100}
                print(f"Grid search failed, using default parameters: {optimal_params}")

            # Setup results directory
            results_base_dir = f'{root}/results/distributed_kamp_modes_{data_name}_sparsity_{sparsity_percent}'
            plots_dir = base_plot_dir
            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(results_base_dir, exist_ok=True)
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

                        # Compressive sensing metrics
                        cs_metrics = calculate_compressive_sensing_metrics(x_true_flat, x_est)
                        nmse_global = cs_metrics.get('nmse', np.nan)
                        mse_global = cs_metrics.get('mse', np.nan)
                        rmse_global = cs_metrics.get('rmse', np.nan)
                        snr_global = cs_metrics.get('snr', np.nan)
                        peak_snr_global = cs_metrics.get('peak_snr', np.nan)

                        # Per-node NMSE
                        nmse_per_node = [calculate_compressive_sensing_metrics(x_true_flat, x).get('nmse', np.nan) for x in node_estimates]
                        mean_nmse_per_node = np.mean(nmse_per_node)
                        std_nmse_per_node = np.std(nmse_per_node)

                        # Consensus error
                        consensus_error = np.var([np.linalg.norm(x.flatten() - x_est) for x in node_estimates])

                        # Image quality metrics for 2D data
                        image_metrics = {}
                        if is_2d and x_true_2d is not None:
                            try:
                                x_est_2d = x_est.reshape(x_true_2d.shape)
                                image_metrics = {
                                    'ssim': calculate_ssim(x_true_2d, x_est_2d),
                                    'gmsd': calculate_gmsd(x_true_2d, x_est_2d),
                                    'fsim': calculate_fsim(x_true_2d, x_est_2d),
                                    'vif': calculate_vif(x_true_2d, x_est_2d)
                                }
                            except Exception as e:
                                logging.error(f"Error computing image metrics for {adj_file}, mode {mode_name}: {e}")
                                print(f"Warning: Error computing image metrics for {adj_file}, mode {mode_name}: {e}")
                                image_metrics = {'ssim': np.nan, 'gmsd': np.nan, 'fsim': np.nan, 'vif': np.nan}

                        print(f"\n=== DistributedKAMP - Topology: {os.path.basename(adj_file)}, Mode: {mode_name} ===")
                        print(f"NMSE Global: {nmse_global:.6f}, MSE Global: {mse_global:.6f}, RMSE Global: {rmse_global:.6f}")
                        print(f"SNR Global: {snr_global:.6f}, Peak SNR Global: {peak_snr_global:.6f}")
                        print(f"Mean NMSE per Node: {mean_nmse_per_node:.6f}, Consensus Error: {consensus_error:.6f}, Fit Time: {fit_time:.2f}s")
                        if is_2d:
                            print(f"Image Metrics - SSIM: {image_metrics['ssim']:.6f}, GMSD: {image_metrics['gmsd']:.6f}, FSIM: {image_metrics['fsim']:.6f}, VIF: {image_metrics['vif']:.6f}")

                        result = {
                            'data_type': data_name,
                            'sparsity_percent': sparsity_percent,
                            'topology': os.path.basename(adj_file),
                            'mode': mode_name,
                            'num_nodes': num_nodes,
                            'is_dag': is_dag,
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
                        all_results.append(result)

                        # Plotting
                        base_filename = f"{os.path.basename(adj_file).replace('.txt', '')}_{mode_name}_sparsity_{sparsity_percent}"

                        # Signal Comparison Plot
                        try:
                            if num_features >= 10 and x_true_2d is not None:
                                # For high-dimensional data: use imshow for comparison
                                x_est_2d = x_est.reshape(x_true_2d.shape)
                                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                                im1 = axes[0].imshow(x_true_2d, cmap='viridis', aspect='auto')
                                axes[0].set_title('Original Signal')
                                plt.colorbar(im1, ax=axes[0], label='Value')
                                im2 = axes[1].imshow(x_est_2d, cmap='viridis', aspect='auto')
                                axes[1].set_title('Reconstructed Signal')
                                plt.colorbar(im2, ax=axes[1], label='Value')
                            else:
                                # For low-dimensional data: create comprehensive visualization
                                fig = plt.figure(figsize=(15, 10))
                                # Subplot 1: Measurement matrix A
                                plt.subplot(2, 2, 1)
                                plt.imshow(A, aspect='auto', cmap='viridis')
                                plt.title(f'Measurement Matrix A ({A.shape[0]}×{A.shape[1]})')
                                plt.xlabel('Features')
                                plt.ylabel('Samples')
                                plt.colorbar(label='Value')
                                # Subplot 2: Observation vector y
                                plt.subplot(2, 2, 2)
                                plt.plot(y, 'g-', linewidth=1)
                                plt.title(f'Observation Vector y ({len(y)}×1)')
                                plt.xlabel('Sample Index')
                                plt.ylabel('Observation Value')
                                plt.grid(True, alpha=0.3)
                                # Subplot 3: True vs Reconstructed signal
                                plt.subplot(2, 2, 3)
                                plt.plot(x_true_flat, 'b-', label='True Signal', linewidth=2, alpha=0.8)
                                plt.plot(x_est, 'r--', label='Reconstructed', linewidth=2, alpha=0.8)
                                plt.title(f'Signal Comparison ({num_features} features)')
                                plt.xlabel('Feature Index')
                                plt.ylabel('Signal Value')
                                plt.legend()
                                plt.grid(True, alpha=0.3)
                                # Subplot 4: Reconstruction error
                                plt.subplot(2, 2, 4)
                                error = x_true_flat - x_est
                                plt.plot(error, 'm-', linewidth=1.5)
                                plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                                plt.title('Reconstruction Error')
                                plt.xlabel('Feature Index')
                                plt.ylabel('Error Value')
                                plt.grid(True, alpha=0.3)
                                plt.tight_layout()
                            plt.savefig(os.path.join(plots_dir, f'signal_comparison_{base_filename}.png'), bbox_inches='tight')
                            plt.close()
                        except Exception as e:
                            logging.error(f"Error plotting signal comparison for {adj_file}, mode {mode_name}: {e}")
                            print(f"Warning: Error plotting signal comparison for {adj_file}, mode {mode_name}: {e}")

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
    df_all.to_csv(all_results_file.replace('.xlsx', '.csv'), index=False)
    df_all.to_excel(all_results_file, index=False)
    print(f"All results saved to {all_results_file.replace('.xlsx', '.csv')} and {all_results_file}")

if __name__ == "__main__":
    main()