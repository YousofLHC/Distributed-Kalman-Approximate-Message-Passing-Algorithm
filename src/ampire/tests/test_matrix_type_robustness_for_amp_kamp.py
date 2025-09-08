import numpy as np
import os
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
import sys
sys.path.insert(0,'src')
from ampire.core.amp import AMP
from ampire.core.kamp import KAMP
from ampire.utils.metrics import calculate_compressive_sensing_metrics
from ampire.utils.helpers import generate_synthetic_data, create_measurement_matrix

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'test_matrix_type_robustness_for_amp_kamp.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def grid_search_hyperparameters(A, y, x_true_flat, method, param_grids):
    """Perform grid search for AMP or KAMP hyperparameters, evaluating multiple metrics."""
    param_combinations = list(product(*param_grids.values()))
    results = []

    print(f"Starting grid search for {method.__name__}...")
    logging.info(f"Starting grid search for {method.__name__} with {len(param_combinations)} parameter combinations")
    for params in tqdm(param_combinations, desc="Grid search"):
        param_dict = dict(zip(param_grids.keys(), params))
        try:
            start_time = time.time()
            model = method(**param_dict)  # Removed random_state
            model.fit(A, y)
            x_est = model.solve().flatten()
            fit_time = time.time() - start_time

            # Compressive sensing metrics
            cs_metrics = calculate_compressive_sensing_metrics(x_true_flat, x_est)
            nmse = cs_metrics.get('nmse', np.nan)
            mse = cs_metrics.get('mse', np.nan)
            rmse = cs_metrics.get('rmse', np.nan)
            snr = cs_metrics.get('snr', np.nan)
            peak_snr = cs_metrics.get('peak_snr', np.nan)

            result = {
                'params': param_dict,
                'nmse': nmse,
                'mse': mse,
                'rmse': rmse,
                'snr': snr,
                'peak_snr': peak_snr,
                'fit_time': fit_time
            }
            results.append(result)
        except Exception as e:
            logging.error(f"Error with {method.__name__} params {param_dict}: {e}")
            print(f"Warning: Error with {method.__name__} params {param_dict}: {e}")

    # Find best parameters for nmse
    if results:
        best = min(results, key=lambda x: x.get('nmse', float('inf')))
        print(f"Best params for {method.__name__} (nmse): {best['params']} with value {best['nmse']:.4f}")
        logging.info(f"Best params for {method.__name__} (nmse): {best['params']} with value {best['nmse']:.4f}")
        return best['params']
    logging.warning(f"No valid results for {method.__name__} grid search")
    print(f"Warning: No valid results for {method.__name__} grid search")
    return None

def modified_generate_synthetic_data(num_samples, num_features, sparsity, noise_std, random_state, matrix_type='gaussian'):
    """Modified generate_synthetic_data to use create_measurement_matrix."""
    logging.debug(f"Generating synthetic data: num_samples={num_samples}, num_features={num_features}, sparsity={sparsity}, matrix_type={matrix_type}, random_state={random_state}")
    rng = np.random.default_rng(random_state)
    
    # Generate measurement matrix
    try:
        A = create_measurement_matrix(num_samples, num_features, matrix_type, random_state)
        logging.debug(f"Generated measurement matrix A of shape {A.shape} for {matrix_type}")
    except Exception as e:
        logging.error(f"Error in create_measurement_matrix: {e}")
        raise
    
    # Generate sparse signal
    try:
        x_true_flat = np.zeros(num_features)
        non_zero_indices = rng.choice(num_features, size=int(sparsity * num_features), replace=False)
        x_true_flat[non_zero_indices] = rng.normal(0, 1, size=len(non_zero_indices))
        logging.debug(f"Generated sparse signal with {len(non_zero_indices)} non-zero elements")
    except Exception as e:
        logging.error(f"Error generating sparse signal: {e}")
        raise
    
    # Generate observations
    try:
        y = A @ x_true_flat + rng.normal(0, noise_std, num_samples)
        logging.debug(f"Generated observation vector y of shape {y.shape}")
    except Exception as e:
        logging.error(f"Error generating observations: {e}")
        raise
    
    return A, y, x_true_flat

def plot_metric_comparison(df, data_name, sparsity_percent, metric, plots_dir, timestamp):
    """Plot comparison of a metric across matrix types and methods."""
    if df.empty or 'matrix_type' not in df.columns or 'method' not in df.columns:
        logging.warning(f"Skipping metric comparison plot for {data_name}, sparsity {sparsity_percent}%, metric {metric}: DataFrame is empty or missing required columns")
        print(f"Warning: Skipping metric comparison plot for {data_name}, sparsity {sparsity_percent}%, metric {metric}: DataFrame is empty or missing required columns")
        return
    
    fig = plt.figure(figsize=(10, 6))
    matrix_types = df['matrix_type'].unique()
    methods = df['method'].unique()
    bar_width = 0.35
    x = np.arange(len(matrix_types))
    
    for i, method in enumerate(methods):
        metric_values = []
        for matrix_type in matrix_types:
            subset = df[(df['method'] == method) & (df['matrix_type'] == matrix_type)]
            metric_values.append(subset[metric].mean() if not subset.empty else np.nan)
        plt.bar(x + i * bar_width, metric_values, bar_width, label=method, alpha=0.8)
    
    plt.xlabel('Matrix Type')
    plt.ylabel(metric)
    plt.title(f'{metric.upper()} Comparison for {data_name}, Sparsity {sparsity_percent}%')
    plt.xticks(x + bar_width / 2, matrix_types)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, f'{metric}_comparison_{data_name}_sparsity_{sparsity_percent}_{timestamp}.png'), bbox_inches='tight')
    plt.close()
    logging.info(f"Saved {metric} comparison plot for {data_name}, sparsity {sparsity_percent}%")

def main():
    # Configuration
    config = {
        'noise_std': 0.01,
        'random_state': 42,
        'sparsity_levels': [0.1, 0.35, 0.9],
        'num_trials': 10,
        'matrix_types': ['gaussian', 'heavy', 'orthogonal'],
        'param_grids': {
            'AMP': {
                'alpha': [0.1, 0.5, 0.9],
                'tau': [0.01, 0.1, 0.5],
                'max_iter': [100, 500]
            },
            'KAMP': {
                'alpha': [0.1, 0.5, 0.9],
                'tau': [0.01, 0.1, 0.5],
                'max_iter': [100, 500]
            }
        },
        'data_types': [
            {'is_2d': False, 'name': 'complex_signal', 'num_features': 700, 'num_samples': 900, 'shape_2d': None}
        ]
    }

    # Setup directories
    root = 'ThesisExperiments/test_matrix_type_robustness'
    base_plot_dir = f'{root}/plots'
    os.makedirs(base_plot_dir, exist_ok=True)
    os.makedirs(f'{root}/results', exist_ok=True)

    all_results = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for data_type in config['data_types']:
        is_2d = data_type['is_2d']
        data_name = data_type['name']
        num_features = data_type['num_features']
        num_samples = data_type['num_samples']

        print(f"\n{'='*60}\nTesting with {data_name} synthetic data\n{'='*60}")
        logging.info(f"Testing with {data_name} synthetic data: num_samples={num_samples}, num_features={num_features}")

        for sparsity in config['sparsity_levels']:
            sparsity_percent = int(sparsity * 100)
            print(f"\nTesting with {sparsity_percent}% non-zero elements")
            logging.info(f"Testing with {sparsity_percent}% sparsity")

            for matrix_type in config['matrix_types']:
                print(f"\nTesting with {matrix_type} matrix")
                logging.info(f"Testing with {matrix_type} matrix")

                for trial in range(config['num_trials']):
                    trial_seed = config['random_state'] + trial
                    print(f"\nTrial {trial + 1}/{config['num_trials']} (seed={trial_seed})")
                    logging.info(f"Trial {trial + 1}/{config['num_trials']} (seed={trial_seed})")

                    # Generate data
                    try:
                        A, y, x_true_flat = modified_generate_synthetic_data(
                            num_samples=num_samples,
                            num_features=num_features,
                            sparsity=sparsity,
                            noise_std=config['noise_std'],
                            random_state=trial_seed,
                            matrix_type=matrix_type
                        )
                        print(f"Generated {data_name} data: {A.shape[0]} samples, {A.shape[1]} features")
                        non_zero_count = np.sum(x_true_flat != 0)
                        print(f"True signal sparsity: {non_zero_count} non-zero elements ({non_zero_count / len(x_true_flat):.1%})")
                        logging.info(f"Generated {data_name} data: {A.shape[0]} samples, {A.shape[1]} features, {non_zero_count} non-zero elements")
                    except Exception as e:
                        logging.error(f"Error generating data for {data_name}, {matrix_type}, sparsity {sparsity_percent}%, trial {trial + 1}: {e}")
                        print(f"Warning: Error generating data for {data_name}, {matrix_type}, sparsity {sparsity_percent}%, trial {trial + 1}: {e}")
                        continue

                    for method_name, method in [('AMP', AMP), ('KAMP', KAMP)]:
                        # Grid search for optimal parameters
                        logging.info(f"Running grid search for {method_name}")
                        optimal_params = grid_search_hyperparameters(
                            A, y, x_true_flat, method, config['param_grids'][method_name]
                        )
                        if not optimal_params:
                            optimal_params = {'alpha': 0.5, 'tau': 0.1, 'max_iter': 500}
                            print(f"Grid search failed for {method_name}, using default parameters: {optimal_params}")
                            logging.warning(f"Grid search failed for {method_name}, using default parameters: {optimal_params}")

                        try:
                            start_time = time.time()
                            model = method(**optimal_params)  # Removed random_state
                            model.fit(A, y)
                            x_est = model.solve().flatten()
                            fit_time = time.time() - start_time
                            logging.info(f"{method_name} completed in {fit_time:.2f}s")

                            # Compressive sensing metrics
                            cs_metrics = calculate_compressive_sensing_metrics(x_true_flat, x_est)
                            nmse = cs_metrics.get('nmse', np.nan)
                            mse = cs_metrics.get('mse', np.nan)
                            rmse = cs_metrics.get('rmse', np.nan)
                            snr = cs_metrics.get('snr', np.nan)
                            peak_snr = cs_metrics.get('peak_snr', np.nan)

                            print(f"\n=== {method_name} - Matrix: {matrix_type}, Trial: {trial + 1} ===")
                            print(f"NMSE: {nmse:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}")
                            print(f"SNR: {snr:.6f}, Peak SNR: {peak_snr:.6f}, Fit Time: {fit_time:.2f}s")
                            logging.info(f"{method_name} metrics: NMSE={nmse:.6f}, MSE={mse:.6f}, RMSE={rmse:.6f}, SNR={snr:.6f}, Peak SNR={peak_snr:.6f}, Fit Time={fit_time:.2f}s")

                            result = {
                                'data_type': data_name,
                                'sparsity_percent': sparsity_percent,
                                'matrix_type': matrix_type,
                                'trial': trial + 1,
                                'seed': trial_seed,
                                'method': method_name,
                                'nmse': nmse,
                                'mse': mse,
                                'rmse': rmse,
                                'snr': snr,
                                'peak_snr': peak_snr,
                                'fit_time': fit_time
                            }
                            all_results.append(result)

                            # Plotting
                            base_filename = f"{method_name}_{matrix_type}_sparsity_{sparsity_percent}_trial_{trial + 1}"
                            plots_dir = base_plot_dir
                            os.makedirs(plots_dir, exist_ok=True)

                            # Signal Comparison Plot
                            try:
                                fig = plt.figure(figsize=(15, 10))
                                plt.subplot(2, 2, 1)
                                plt.imshow(A, aspect='auto', cmap='viridis')
                                plt.title(f'Measurement Matrix A ({A.shape[0]}×{A.shape[1]})')
                                plt.xlabel('Features')
                                plt.ylabel('Samples')
                                plt.colorbar(label='Value')
                                plt.subplot(2, 2, 2)
                                plt.plot(y, 'g-', linewidth=1)
                                plt.title(f'Observation Vector y ({len(y)}×1)')
                                plt.xlabel('Sample Index')
                                plt.ylabel('Observation Value')
                                plt.grid(True, alpha=0.3)
                                plt.subplot(2, 2, 3)
                                plt.plot(x_true_flat, 'b-', label='True Signal', linewidth=2, alpha=0.8)
                                plt.plot(x_est, 'r--', label=f'Reconstructed ({method_name})', linewidth=2, alpha=0.8)
                                plt.title(f'Signal Comparison ({num_features} features)')
                                plt.xlabel('Feature Index')
                                plt.ylabel('Signal Value')
                                plt.legend()
                                plt.grid(True, alpha=0.3)
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
                                logging.info(f"Saved signal comparison plot: {base_filename}.png")
                            except Exception as e:
                                logging.error(f"Error plotting signal comparison for {method_name}, {matrix_type}, trial {trial + 1}: {e}")
                                print(f"Warning: Error plotting signal comparison for {method_name}, {matrix_type}, trial {trial + 1}: {e}")

                        except Exception as e:
                            logging.error(f"Error running {method_name} for {matrix_type}, trial {trial + 1}: {e}")
                            print(f"Warning: Error running {method_name} for {matrix_type}, trial {trial + 1}: {e}")

            # Save sparsity results
            df_sparsity = pd.DataFrame([r for r in all_results if r['sparsity_percent'] == sparsity_percent and r['data_type'] == data_name])
            results_file = f'{root}/results/amp_kamp_{data_name}_sparsity_{sparsity_percent}_{timestamp}.xlsx'
            df_sparsity.to_excel(results_file, index=False)
            print(f"Results for {data_name} with {sparsity_percent}% sparsity saved to {results_file}")
            logging.info(f"Results for {data_name} with {sparsity_percent}% sparsity saved to {results_file}")

            # Plot metric comparisons
            try:
                for metric in ['nmse', 'mse', 'rmse', 'snr', 'peak_snr', 'fit_time']:
                    plot_metric_comparison(df_sparsity, data_name, sparsity_percent, metric, base_plot_dir, timestamp)
            except Exception as e:
                logging.error(f"Error plotting metric comparisons for {data_name}, sparsity {sparsity_percent}%: {e}")
                print(f"Warning: Error plotting metric comparisons for {data_name}, sparsity {sparsity_percent}%: {e}")

    # Save all results
    df_all = pd.DataFrame(all_results)
    all_results_file = f'{root}/results/amp_kamp_all_modes_{timestamp}.xlsx'
    df_all.to_csv(all_results_file.replace('.xlsx', '.csv'), index=False)
    df_all.to_excel(all_results_file, index=False)
    print(f"All results saved to {all_results_file.replace('.xlsx', '.csv')} and {all_results_file}")
    logging.info(f"All results saved to {all_results_file.replace('.xlsx', '.csv')} and {all_results_file}")

if __name__ == "__main__":
    main()