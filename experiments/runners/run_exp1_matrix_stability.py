import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Try imports for available solvers
from importlib import import_module

# Import KAMP (required)
try:
    from ampire.core.kamp import KAMP
except Exception as e:
    KAMP = None

# Try AMP (may be defined in core.kamp or core.amp)
AMP = None
try:
    mod = import_module('ampire.core.kamp')
    AMP = getattr(mod, 'AMP', None)
except Exception:
    AMP = None
if AMP is None:
    try:
        from ampire.core.amp import AMP
    except Exception:
        AMP = None

# Try DistributedKAMP
try:
    from ampire.distributed.distributed_kamp import DistributedKAMP
except Exception:
    DistributedKAMP = None

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

def run_solver(name, A, y):
    """Run solver by name. Return x_est (1D numpy) or raise."""
    if name == 'KAMP':
        if KAMP is None:
            raise RuntimeError("KAMP class not available")
        kamp = KAMP(alpha=0.5, tau=0.1, max_iter=100)
        # Some implementations return self from fit, others set internal state.
        try:
            res = kamp.fit(A, y)
            # If fit returns an object with solve, use it; else use kamp.solve()
            if hasattr(res, 'solve'):
                x_est = res.solve()
            else:
                x_est = kamp.solve()
        except TypeError:
            # fallback: call fit without expecting return
            kamp.fit(A, y)
            x_est = kamp.solve()
        return np.asarray(x_est).ravel()

    if name == 'AMP':
        if AMP is None:
            raise RuntimeError("AMP class not available")
        amp = AMP(alpha=0.5, tau=0.1, max_iter=100)
        try:
            res = amp.fit(A, y)
            if hasattr(res, 'solve'):
                x_est = res.solve()
            else:
                x_est = amp.solve()
        except TypeError:
            amp.fit(A, y)
            x_est = amp.solve()
        return np.asarray(x_est).ravel()

    if name == 'DistributedKAMP':
        if DistributedKAMP is None:
            raise RuntimeError("DistributedKAMP not available")
        # Try simple single-node wrapper: create a graph with one node and pass A_list/y_list
        import networkx as nx
        G = nx.DiGraph()
        G.add_node(0)
        A_list = [A]
        y_list = [y]
        # Attempt to instantiate with common signature; fallback to simple call
        try:
            dkamp = DistributedKAMP(alpha=0.5, tau=0.1, node_max_iter=1,
                                   num_triggers=1, graph=G, A_list=A_list, y_list=y_list,
                                   random_state=0)
            # dkamp.fit may not return object
            try:
                res = dkamp.fit()
                if hasattr(res, 'solve'):
                    x_est = res.solve()
                else:
                    x_est = dkamp.solve()
            except TypeError:
                dkamp.fit()
                x_est = dkamp.solve()
            # dkamp.solve might return array or list; flatten
            return np.asarray(x_est).ravel()
        except Exception as e:
            # If constructor signature differs, raise to be caught by caller
            raise

    raise RuntimeError(f"Unknown solver {name}")

def run_experiment():
    results = []
    errors = {}

    # Parameters
    n = 1000  # Signal dimension
    sparsities = [0.1, 0.2, 0.3]  # Sparsity levels
    measurement_rates = [0.5, 0.7, 0.9]  # m/n
    matrix_types = ['orthogonal', 'gaussian', 'heavy-tailed']
    methods = ['KAMP', 'AMP', 'DistributedKAMP']
    num_trials = 10

    out_dir = 'experiments/results/exp1_matrix_stability'
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'exp1_results.csv')
    jsonl_path = os.path.join(out_dir, 'exp1_results.jsonl')
    errors_log = os.path.join(out_dir, 'errors.log')

    # Remove previous logs if necessary (optional)
    # if os.path.exists(csv_path): os.remove(csv_path)

    for sparsity in tqdm(sparsities, desc="Sparsity"):
        k = int(sparsity * n)
        for rate in tqdm(measurement_rates, desc="Measurement Rate", leave=False):
            m = int(rate * n)
            for matrix_type in tqdm(matrix_types, desc="Matrix Type", leave=False):
                for method in tqdm(methods, desc="Method", leave=False):
                    nmse_list = []
                    for trial in range(num_trials):
                        # Create trial ID for tracking
                        trial_id = f"{method}_m{m}_n{n}_k{k}_mt_{matrix_type}_t{trial}"
                        try:
                            # Generate sparse signal as column vector (n,1)
                            x = np.zeros((n, 1))
                            support = np.random.choice(n, k, replace=False)
                            x[support, 0] = np.random.randn(k)

                            # Generate measurement matrix (m x n)
                            A = generate_measurement_matrix(matrix_type, n, m)

                            # Generate measurements y as column vector (m,1)
                            noise = 0.01 * np.random.randn(m, 1)
                            y = A @ x + noise

                            # Run solver (may raise)
                            x_est = run_solver(method, A, y)

                            # Compute metrics (flattened)
                            metrics = calculate_compressive_sensing_metrics(x.flatten(), np.asarray(x_est).flatten())
                            nmse_list.append(metrics.get('nmse', np.nan))

                            # Save single trial result incrementally
                            row = {
                                'trial_id': trial_id,
                                'method': method,
                                'matrix_type': matrix_type,
                                'sparsity': sparsity,
                                'measurement_rate': rate,
                                'nmse': metrics.get('nmse', None),
                                "rmse": metrics['rmse'],
                                "nmse": metrics['nmse'],
                                "snr": metrics['snr'],
                                "peak_snr": metrics['peak_snr']
                            }
                            # append to jsonl
                            with open(jsonl_path, 'a') as f:
                                f.write(pd.Series(row).to_json() + '\n')

                        except Exception as e:
                            tb = traceback.format_exc()
                            errors[trial_id] = str(e)
                            with open(errors_log, 'a') as f:
                                f.write(f"{trial_id}: {e}\n{tb}\n")
                            # continue next trial
                            continue

                    avg_nmse = float(np.nanmean(nmse_list)) if nmse_list else np.nan
                    results.append({
                        'method': method,
                        'matrix_type': matrix_type,
                        'sparsity': sparsity,
                        'measurement_rate': rate,
                        'avg_nmse': avg_nmse
                    })
                    # Save cumulative CSV incrementally
                    pd.DataFrame(results).to_csv(csv_path, index=False)

    return results, errors

def plot_comparison(results, out_dir='experiments/results/exp1_matrix_stability'):
    df = pd.DataFrame(results)
    methods = sorted(df['method'].unique())
    matrix_types = sorted(df['matrix_type'].unique())
    # grouped bar chart: x=matrix_types, bars per method
    x = np.arange(len(matrix_types))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8,5))
    for i, method in enumerate(methods):
        vals = []
        for mt in matrix_types:
            row = df[(df['method']==method) & (df['matrix_type']==mt)]
            vals.append(row['avg_nmse'].mean() if not row.empty else np.nan)
        ax.bar(x + i*width, vals, width, label=method)
    ax.set_xticks(x + width*(len(methods)-1)/2)
    ax.set_xticklabels(matrix_types)
    ax.set_ylabel('Average NMSE')
    ax.set_title('Solver comparison across measurement matrix types')
    ax.legend()
    plot_path = os.path.join(out_dir, 'method_comparison.png')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

def main():
    results, errors = run_experiment()
    out_dir = 'experiments/results/exp1_matrix_stability'
    pd.DataFrame(results).to_csv(os.path.join(out_dir, 'exp1_results_summary.csv'), index=False)
    # save errors summary
    import json
    with open(os.path.join(out_dir, 'errors_summary.json'), 'w') as f:
        json.dump(errors, f, indent=2)
    plot_comparison(results, out_dir=out_dir)
    print("Done.")

if __name__ == '__main__':
    main()