import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ampire.core.kamp import KAMP
from ampire.utils.metrics import calculate_compressive_sensing_metrics

def run_experiment():
    results = []

    # Parameters
    n = 500  # Signal dimension
    deltas = np.linspace(0.1, 0.9, 9)  # Measurement rates δ = m/n
    rhos = np.linspace(0.05, 0.4, 8)  # Sparsity levels ρ = k/n
    num_trials = 20
    success_threshold = 0.01  # NMSE < 0.01 for success

    for delta in tqdm(deltas, desc="Delta"):
        m = int(delta * n)
        for rho in tqdm(rhos, desc="Rho", leave=False):
            k = int(rho * n)
            successes_kamp = 0
            successes_exact = 0  # Assuming exact solution, e.g., if k < m, can recover exactly

            for trial in tqdm(range(num_trials), desc="Trials", leave=False):
                # Generate sparse signal
                x = np.zeros(n)
                support = np.random.choice(n, k, replace=False)
                x[support] = np.random.randn(k)

                # Generate Gaussian measurement matrix
                A = np.random.randn(m, n) / np.sqrt(m)

                # Generate measurements
                y = A @ x + 0.01 * np.random.randn(m)
                y = y.reshape(-1, 1)  # Ensure 2D for KAMP

                # Run KAMP
                kamp = KAMP(alpha=0.5, tau=0.1, max_iter=100)
                kamp.fit(A, y)
                x_hat = kamp.solve()

                # Compute NMSE
                metrics = calculate_compressive_sensing_metrics(x, x_hat)
                nmse = metrics['nmse']

                if nmse < success_threshold:
                    successes_kamp += 1

                # Exact recovery check (simplified: if k <= m/2 for Gaussian)
                if k <= m / 2:
                    successes_exact += 1

            success_prob_kamp = successes_kamp / num_trials
            success_prob_exact = successes_exact / num_trials

            results.append({
                'delta': delta,
                'rho': rho,
                'success_prob_kamp': success_prob_kamp,
                'success_prob_exact': success_prob_exact
            })

    return results

def main():
    results = run_experiment()

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = 'experiments/results/exp2_phase_transition/exp2_results.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    # Save to Excel
    excel_path = 'experiments/results/exp2_phase_transition/exp2_results.xlsx'
    df.to_excel(excel_path, index=False)

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    for rho in rhos:
        subset = df[df['rho'] == rho]
        ax.plot(subset['delta'], subset['success_prob_kamp'], label=f'Rho {rho}')
    ax.set_xlabel('Delta')
    ax.set_ylabel('Success Probability KAMP')
    ax.set_title('Phase Transition')
    ax.legend()
    plot_path = 'experiments/results/exp2_phase_transition/exp2_plot.png'
    plt.savefig(plot_path)
    plt.close()

    print(f"Results saved to {csv_path} and {excel_path}, plot to {plot_path}")

if __name__ == '__main__':
    main()