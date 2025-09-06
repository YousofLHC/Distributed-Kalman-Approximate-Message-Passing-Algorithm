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
from ampire.utils.metrics import calculate_compressive_sensing_metrics

def run_solver(method, A, y):
    if method == 'KAMP':
        kamp = KAMP(alpha=0.5, tau=0.1, max_iter=100)
        kamp.fit(A, y)
        x_hat = kamp.solve()
    elif method == 'AMP':
        amp = AMP(alpha=0.5, tau=0.1, max_iter=100)
        amp.fit(A, y)
        x_hat = amp.solve()
    elif method == 'DistributedKAMP':
        # For comparison, use single-node graph
        G = nx.DiGraph()
        G.add_node(0)
        A_list = [A]
        y_list = [y]
        dkamp = DistributedKAMP(alpha=0.5, tau=0.1, node_max_iter=100, num_triggers=1, graph=G, A_list=A_list, y_list=y_list)
        dkamp.fit()
        x_hat = dkamp.solve()
    else:
        raise ValueError(f"Unknown method {method}")
    return x_hat

def run_experiment():
    results = []

    # Parameters
    n = 500  # Signal dimension
    deltas = np.linspace(0.1, 0.9, 9)  # Measurement rates δ = m/n
    rhos = np.linspace(0.05, 0.4, 8)  # Sparsity levels ρ = k/n
    num_trials = 20
    success_threshold = 0.01  # NMSE < 0.01 for success
    methods = ['AMP', 'KAMP', 'DistributedKAMP']

    for delta in tqdm(deltas, desc="Delta"):
        m = int(delta * n)
        for rho in tqdm(rhos, desc="Rho", leave=False):
            k = int(rho * n)
            successes = {method: 0 for method in methods}
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
                y = y.reshape(-1, 1)  # Ensure 2D

                for method in methods:
                    # Run solver
                    x_hat = run_solver(method, A, y)

                    # Compute NMSE
                    metrics = calculate_compressive_sensing_metrics(x, x_hat)
                    nmse = metrics['nmse']

                    if nmse < success_threshold:
                        successes[method] += 1

                # Exact recovery check (simplified: if k <= m/2 for Gaussian)
                if k <= m / 2:
                    successes_exact += 1

            success_prob_exact = successes_exact / num_trials

            for method in methods:
                results.append({
                    'delta': delta,
                    'rho': rho,
                    'method': method,
                    'success_prob': successes[method] / num_trials,
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

    # Plot results for each method
    methods = df['method'].unique()
    rhos = df['rho'].unique()
    fig, axes = plt.subplots(len(methods), 1, figsize=(10, 6 * len(methods)))
    if len(methods) == 1:
        axes = [axes]
    for i, method in enumerate(methods):
        ax = axes[i]
        for rho in rhos:
            subset = df[(df['method'] == method) & (df['rho'] == rho)]
            ax.plot(subset['delta'], subset['success_prob'], label=f'Rho {rho}')
        ax.set_xlabel('Delta')
        ax.set_ylabel('Success Probability')
        ax.set_title(f'Phase Transition - {method}')
        ax.legend()
    plot_path = 'experiments/results/exp2_phase_transition/exp2_plot.png'
    plt.savefig(plot_path)
    plt.close()

    print(f"Results saved to {csv_path} and {excel_path}, plot to {plot_path}")

if __name__ == '__main__':
    main()