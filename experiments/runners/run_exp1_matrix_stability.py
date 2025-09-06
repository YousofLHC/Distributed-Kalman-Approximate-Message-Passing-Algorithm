import numpy as np
import pandas as pd
from scipy.linalg import orth
import sys
import os

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
        return orth(np.random.randn(m, n))[:m, :n]
    else:
        raise ValueError("Unknown matrix type")

def run_experiment():
    results = []

    # Parameters
    n = 1000  # Signal dimension
    sparsities = [0.1, 0.2, 0.3]  # Sparsity levels
    measurement_rates = [0.5, 0.7, 0.9]  # m/n
    matrix_types = ['gaussian', 'heavy-tailed', 'orthogonal']
    num_trials = 10

    for sparsity in sparsities:
        k = int(sparsity * n)
        for rate in measurement_rates:
            m = int(rate * n)
            for matrix_type in matrix_types:
                nmse_list = []
                for trial in range(num_trials):
                    # Generate sparse signal
                    x = np.zeros(n)
                    support = np.random.choice(n, k, replace=False)
                    x[support] = np.random.randn(k)

                    # Generate measurement matrix
                    A = generate_measurement_matrix(matrix_type, n, m)

                    # Generate measurements
                    y = A @ x + 0.01 * np.random.randn(m)

                    # Run KAMP
                    kamp = KAMP(alpha=0.5, tau=0.1, max_iter=100)
                    kamp.fit(A, y)
                    x_hat = kamp.solve()

                    # Compute NMSE
                    metrics = calculate_compressive_sensing_metrics(x, x_hat)
                    nmse_list.append(metrics['nmse'])

                avg_nmse = np.mean(nmse_list)
                results.append({
                    'sparsity': sparsity,
                    'measurement_rate': rate,
                    'matrix_type': matrix_type,
                    'avg_nmse': avg_nmse
                })

    return results

def main():
    results = run_experiment()

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = 'experiments/results/exp1_matrix_stability/exp1_results.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    # Save to Excel
    excel_path = 'experiments/results/exp1_matrix_stability/exp1_results.xlsx'
    df.to_excel(excel_path, index=False)

    print(f"Results saved to {csv_path} and {excel_path}")

if __name__ == '__main__':
    main()