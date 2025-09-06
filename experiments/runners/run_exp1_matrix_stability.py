import numpy as np
import pandas as pd
# from scipy.linalg import orth   # removed, replaced with qr-based construction
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ampire.core.kamp import KAMP  # Assuming KAMP class
from ampire.utils.metrics import calculate_compressive_sensing_metrics

def generate_measurement_matrix(type_, n, m):
    if type_ == 'gaussian':
        return np.random.randn(m, n) / np.sqrt(m)
    elif type_ == 'heavy-tailed':
        return np.random.exponential(1, (m, n)) / np.sqrt(m)
    elif type_ == 'orthogonal':
        # Build a stable orthonormal-row matrix for m <= n (common in CS experiments).
        # If m <= n: generate an n x n random matrix, QR-decompose and take first m rows.
        # If m > n: generate m x m random matrix, QR-decompose and take first n columns.
        if m <= n:
            Q, _ = np.linalg.qr(np.random.randn(n, n))
            return Q[:m, :n]
        else:
            Q, _ = np.linalg.qr(np.random.randn(m, m))
            return Q[:, :n]
    else:
        raise ValueError("Unknown matrix type")

def run_experiment():
    results = []

    # Parameters
    n = 1000  # Signal dimension
    sparsities = [0.1, 0.2, 0.3]  # Sparsity levels
    measurement_rates = [0.5, 0.7, 0.9]  # m/n
    matrix_types = ['orthogonal', 'gaussian', 'heavy-tailed']
    methods = ['KAMP']  # 'AMP', 'Distributed KAMP' - add when available
    num_trials = 10

    for sparsity in tqdm(sparsities, desc="Sparsity"):
        k = int(sparsity * n)
        for rate in tqdm(measurement_rates, desc="Measurement Rate", leave=False):
            m = int(rate * n)
            for matrix_type in tqdm(matrix_types, desc="Matrix Type", leave=False):
                for method in tqdm(methods, desc="Method", leave=False):
                    nmse_list = []
                    for trial in tqdm(range(num_trials), desc="Trials", leave=False):
                        # Generate sparse signal as column vector (n,1)
                        x = np.zeros((n, 1))
                        support = np.random.choice(n, k, replace=False)
                        x[support, 0] = np.random.randn(k)

                        # Generate measurement matrix (m x n)
                        A = generate_measurement_matrix(matrix_type, n, m)

                        # Generate measurements y as column vector (m,1)
                        noise = 0.01 * np.random.randn(m, 1)
                        y = A @ x + noise

                        # Run KAMP
                        kamp = KAMP(alpha=0.5, tau=0.1, max_iter=100)
                        kamp.fit(A, y)
                        x_hat = kamp.solve()

                        # Ensure flattened vectors for metric computation
                        metrics = calculate_compressive_sensing_metrics(x.flatten(), np.asarray(x_hat).flatten())
                        nmse_list.append(metrics['nmse'])

                    avg_nmse = np.mean(nmse_list)
                    results.append({
                        'sparsity': sparsity,
                        'measurement_rate': rate,
                        'matrix_type': matrix_type,
                        'method': method,
                        'avg_nmse': avg_nmse
                    })

                    # Save results incrementally
                    df = pd.DataFrame(results)
                    csv_path = 'experiments/results/exp1_matrix_stability/exp1_results.csv'
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    df.to_csv(csv_path, index=False)

    return results, sparsities, measurement_rates

def main():
    results, sparsities, measurement_rates = run_experiment()

    # Save to Excel (final save)
    df = pd.DataFrame(results)
    excel_path = 'experiments/results/exp1_matrix_stability/exp1_results.xlsx'
    df.to_excel(excel_path, index=False)

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    for sparsity in sparsities:
        for rate in measurement_rates:
            subset = df[(df['sparsity'] == sparsity) & (df['measurement_rate'] == rate)]
            ax.bar(subset['matrix_type'], subset['avg_nmse'], label=f'Sparsity {sparsity}, Rate {rate}')
    ax.set_xlabel('Matrix Type')
    ax.set_ylabel('Average NMSE')
    ax.set_title('Matrix Stability Test')
    ax.legend()
    plot_path = 'experiments/results/exp1_matrix_stability/exp1_plot.png'
    plt.savefig(plot_path)
    plt.close()

    csv_path = 'experiments/results/exp1_matrix_stability/exp1_results.csv'
    print(f"Results saved to {csv_path} and {excel_path}, plot to {plot_path}")

if __name__ == '__main__':
    main()