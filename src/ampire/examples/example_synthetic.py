import numpy as np
import matplotlib.pyplot as plt
from ampire.core.kamp import KAMP
from ampire.distributed.distributed_kamp import DistributedKAMP
from ampire.utils.visualization import plot_signal_comparison
from ampire.utils.metrics import calculate_metrics, log_results
import csv
import json
import time
from pathlib import Path
from tqdm import tqdm


def get_sinusoidal_data(m=400, n=50, sigma2=0.01, random_state=42):
    """Generate sinusoidal data for testing."""
    np.random.seed(random_state)
    A = np.random.randn(m, n)
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    t = np.linspace(0, 1, n)
    x_true = np.sin(2 * np.pi * 3 * t).reshape(n, 1)
    y = A @ x_true + np.sqrt(sigma2) * np.random.randn(m, 1)
    return A, y, x_true

def plot_node_estimates(x_true, node_estimates, max_iter, title_prefix):
    """Plot individual node estimates and overlay plot with minimal code."""
    num_nodes = len(node_estimates)
    # Individual node plots
    fig, axes = plt.subplots(2, (num_nodes + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    for i, x_node in enumerate(node_estimates):
        nmse_node = np.linalg.norm(x_node - x_true) ** 2 / np.linalg.norm(x_true) ** 2
        axes[i].plot(x_true.flatten(), 'bo-', label='True Signal')
        axes[i].plot(x_node.flatten(), 'rx--', label=f'Node {i} (max_iter={max_iter[i]})')
        axes[i].set_xlabel("Index")
        axes[i].set_ylabel("Amplitude")
        axes[i].set_title(f"Node {i}\nNMSE: {nmse_node:.4f}")
        axes[i].legend()
        axes[i].grid(True)
    for i in range(num_nodes, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    
    # Overlay plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_true.flatten(), 'b-o', label='True Signal', linewidth=2)
    colors = ['r', 'g', 'm', 'c', 'k']
    markers = ['x', 's', '^', 'd', '*']
    for i, x_node in enumerate(node_estimates):
        nmse_node = np.linalg.norm(x_node - x_true) ** 2 / np.linalg.norm(x_true) ** 2
        plt.plot(x_node.flatten(), f'{colors[i % len(colors)]}{markers[i % len(markers)]}--', 
                 label=f'Node {i} (max_iter={max_iter[i]}, NMSE: {nmse_node:.4f})')
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.title(f"{title_prefix} Overlay of True and Estimated Signals")
    plt.legend()
    plt.grid(True)
    plt.show()

def run_synthetic_example():
    """Original synthetic test with moderate dimensions and noise."""
    m, n = 100, 50
    alpha, tau = 0.5, 0.1
    noise_level = 0.1
    sparsity = 0.2
    
    rng = np.random.RandomState(42)
    A = rng.randn(m, n) / np.linalg.norm(rng.randn(m, n), axis=0, keepdims=True)
    sparse_mask = (rng.random(n) < sparsity).reshape(n, 1)
    x_true = rng.randn(n, 1) * sparse_mask
    y = A @ x_true + noise_level * rng.randn(m, 1)
    
    kamp = KAMP(alpha=alpha, tau=tau, max_iter=100)
    print(f"\nSynthetic Example: A shape: {A.shape}, y shape: {y.shape}, x_true shape: {x_true.shape}")
    kamp.fit(A, y)
    print(f"Internal A shape: {kamp.A.shape}, y shape: {kamp.y.shape}")
    x_est = kamp.solve()
    
    plot_signal_comparison(x_true, x_est, "KAMP Synthetic Signal Recovery")
    nmse = np.linalg.norm(x_est - x_true) ** 2 / np.linalg.norm(x_true) ** 2
    print(f"NMSE: {nmse:.4f}")

def run_high_dimensionality_example():
    """Test with larger dimensions to evaluate scalability."""
    m, n = 500, 200
    alpha, tau = 0.5, 0.1
    noise_level = 0.1
    sparsity = 0.2
    
    rng = np.random.RandomState(42)
    A = rng.randn(m, n) / np.linalg.norm(rng.randn(m, n), axis=0, keepdims=True)
    sparse_mask = (rng.random(n) < sparsity).reshape(n, 1)
    x_true = rng.randn(n, 1) * sparse_mask
    y = A @ x_true + noise_level * rng.randn(m, 1)
    
    kamp = KAMP(alpha=alpha, tau=tau, max_iter=100)
    print(f"\nHigh Dimensionality Example: A shape: {A.shape}, y shape: {y.shape}, x_true shape: {x_true.shape}")
    kamp.fit(A, y)
    print(f"Internal A shape: {kamp.A.shape}, y shape: {kamp.y.shape}")
    x_est = kamp.solve()
    
    plot_signal_comparison(x_true, x_est, "KAMP High Dimensionality Recovery")
    nmse = np.linalg.norm(x_est - x_true) ** 2 / np.linalg.norm(x_true) ** 2
    print(f"NMSE: {nmse:.4f}")

def run_high_noise_example():
    """Test with high noise to evaluate robustness."""
    m, n = 100, 50
    alpha, tau = 0.5, 0.2
    noise_level = 0.5
    sparsity = 0.2
    
    rng = np.random.RandomState(42)
    A = rng.randn(m, n) / np.linalg.norm(rng.randn(m, n), axis=0, keepdims=True)
    sparse_mask = (rng.random(n) < sparsity).reshape(n, 1)
    x_true = rng.randn(n, 1) * sparse_mask
    y = A @ x_true + noise_level * rng.randn(m, 1)
    
    kamp = KAMP(alpha=alpha, tau=tau, max_iter=100)
    print(f"\nHigh Noise Example: A shape: {A.shape}, y shape: {y.shape}, x_true shape: {x_true.shape}")
    kamp.fit(A, y)
    print(f"Internal A shape: {kamp.A.shape}, y shape: {kamp.y.shape}")
    x_est = kamp.solve()
    
    plot_signal_comparison(x_true, x_est, "KAMP High Noise Recovery")
    nmse = np.linalg.norm(x_est - x_true) ** 2 / np.linalg.norm(x_true) ** 2
    print(f"NMSE: {nmse:.4f}")

def run_varying_sparsity_example():
    """Test with low and high sparsity levels."""
    m, n = 100, 50
    alpha, tau = 0.5, 0.1
    noise_level = 0.1
    sparsity_levels = [0.05, 0.5]
    
    rng = np.random.RandomState(42)
    A = rng.randn(m, n) / np.linalg.norm(rng.randn(m, n), axis=0, keepdims=True)
    
    for sparsity in sparsity_levels:
        sparse_mask = (rng.random(n) < sparsity).reshape(n, 1)
        x_true = rng.randn(n, 1) * sparse_mask
        y = A @ x_true + noise_level * rng.randn(m, 1)
        
        kamp = KAMP(alpha=alpha, tau=tau, max_iter=100)
        print(f"\nVarying Sparsity Example (sparsity={sparsity}): A shape: {A.shape}, y shape: {y.shape}, x_true shape: {x_true.shape}")
        kamp.fit(A, y)
        print(f"Internal A shape: {kamp.A.shape}, y shape: {kamp.y.shape}")
        x_est = kamp.solve()
        
        plot_signal_comparison(x_true, x_est, f"KAMP Sparsity {sparsity} Recovery")
        nmse = np.linalg.norm(x_est - x_true) ** 2 / np.linalg.norm(x_true) ** 2
        print(f"NMSE (sparsity={sparsity}): {nmse:.4f}")

def run_phase_transition_experiment(
    n: int = 500,
    lam1: float = 0.01,
    delta_list=(0.2, 0.4, 0.6, 0.8),
    rho_list=(0.05, 0.1, 0.2),
    trials: int = 3,
    max_iter: int = 50,
    snr_db: float = 40.0,
    success_threshold: float = 1e-2,
    output_dir: str = "results/synthetic_phase"
):
    """
    Run a synthetic phase transition experiment using KAMP and a baseline AMP.

    Parameters
    ----------
    n : int, default 500
        Dimension of the unknown signal.
    lam1 : float, default 0.01
        L1 regularisation weight (tau in KAMP).
    delta_list : iterable of float
        Sampling ratios m/n to evaluate.
    rho_list : iterable of float
        Sparsity ratios k/n to evaluate.
    trials : int, default 3
        Number of Monte‑Carlo trials per (delta, rho) point.
    max_iter : int, default 50
        Maximum number of KAMP iterations.
    snr_db : float, default 40.0
        Signal‑to‑noise ratio in decibels.
    success_threshold : float, default 1e-2
        NMSE threshold below which a recovery is considered successful.
    output_dir : str, default "results/synthetic_phase"
        Directory where CSV and JSONL results will be saved.

    Notes
    -----
    Results are saved as <output_dir>/phase_transition_results.csv and
    <output_dir>/phase_transition_results.jsonl.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def generate_problem(delta, rho):
        m = int(delta * n)
        A = np.random.randn(m, n) / np.sqrt(m)
        k = max(1, int(rho * n))
        x_true = np.zeros((n, 1))
        idx = np.random.choice(n, size=k, replace=False)
        x_true[idx] = np.random.randn(k, 1)
        y0 = A @ x_true
        sigma = np.linalg.norm(y0) / np.sqrt(m) * 10 ** (-snr_db / 20)
        y = y0 + sigma * np.random.randn(m, 1)
        return A, x_true, y

    results = []
    algorithms = ("KF-AMP", "AMP")
    for algo in algorithms:
        for delta in tqdm(delta_list, desc=f"Delta ({algo})"):
            for rho in tqdm(rho_list, desc="  Rho", leave=False):
                nmse_sum = 0.0
                success_count = 0
                time_sum = 0.0
                sparsity_sum = 0.0
                for _ in range(trials):
                    A, x_true, y = generate_problem(delta, rho)
                    start = time.time()
                    kamp = KAMP(alpha=0.5, tau=lam1, max_iter=max_iter)
                    kamp.fit(A, y)
                    if algo == "AMP":
                        kamp.Q = np.zeros((kamp.n, kamp.n))  # emulate classic AMP
                    x_est = kamp.solve()
                    elapsed = time.time() - start
                    # Compute NMSE
                    nmse_val = float(np.mean((x_true.flatten() - x_est.flatten()) ** 2) /
                                     np.mean(x_true.flatten() ** 2))
                    nmse_sum += nmse_val
                    if nmse_val < success_threshold:
                        success_count += 1
                    time_sum += elapsed
                    # Compute sparsity
                    sparsity_val = float(np.count_nonzero(np.abs(x_est) > 1e-3) / x_est.size)
                    sparsity_sum += sparsity_val
                results.append({
                    "algo": algo,
                    "delta": delta,
                    "rho": rho,
                    "nmse": nmse_sum / trials,
                    "success": success_count / trials,
                    "time": time_sum / trials,
                    "sparsity": sparsity_sum / trials,
                })
    # Save results
    csv_path = Path(output_dir) / "phase_transition_results.csv"
    jsonl_path = Path(output_dir) / "phase_transition_results.jsonl"
    with csv_path.open("w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    with jsonl_path.open("w") as f_jsonl:
        for row in results:
            f_jsonl.write(json.dumps(row) + "\n")
    print(f"Saved results to {csv_path} and {jsonl_path}")


def run_image_example():
    """Simulate sparse signal recovery for a small image patch."""
    patch_size = 16
    n = patch_size * patch_size
    m = 2 * n
    alpha, tau = 0.5, 0.1
    noise_level = 0.1
    sparsity = 0.1
    
    rng = np.random.RandomState(42)
    A = rng.randn(m, n) / np.linalg.norm(rng.randn(m, n), axis=0, keepdims=True)
    x_true = rng.randn(n, 1) * (rng.random(n) < sparsity).reshape(n, 1)
    y = A @ x_true + noise_level * rng.randn(m, 1)
    
    kamp = KAMP(alpha=alpha, tau=tau, max_iter=100)
    print(f"\nImage Example: A shape: {A.shape}, y shape: {y.shape}, x_true shape: {x_true.shape}")
    kamp.fit(A, y)
    print(f"Internal A shape: {kamp.A.shape}, y shape: {kamp.y.shape}")
    x_est = kamp.solve()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(x_true.reshape(patch_size, patch_size), cmap='gray')
    plt.title("True Image Patch")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(x_est.reshape(patch_size, patch_size), cmap='gray')
    plt.title("Estimated Image Patch")
    plt.axis('off')
    plt.show()
    
    nmse = np.linalg.norm(x_est - x_true) ** 2 / np.linalg.norm(x_true) ** 2
    print(f"Image NMSE: {nmse:.4f}")

def run_audio_example():
    """Simulate sparse signal recovery for a short audio segment."""
    n = 100
    m = 200
    alpha, tau = 0.5, 0.1
    noise_level = 0.1
    sparsity = 0.15
    
    rng = np.random.RandomState(42)
    A = rng.randn(m, n) / np.linalg.norm(rng.randn(m, n), axis=0, keepdims=True)
    x_true = rng.randn(n, 1) * (rng.random(n) < sparsity).reshape(n, 1)
    y = A @ x_true + noise_level * rng.randn(m, 1)
    
    kamp = KAMP(alpha=alpha, tau=tau, max_iter=100)
    print(f"\nAudio Example: A shape: {A.shape}, y shape: {y.shape}, x_true shape: {x_true.shape}")
    kamp.fit(A, y)
    print(f"Internal A shape: {kamp.A.shape}, y shape: {kamp.y.shape}")
    x_est = kamp.solve()
    
    plt.figure(figsize=(10, 4))
    plt.plot(x_true.flatten(), 'b-', label='True Audio Signal')
    plt.plot(x_est.flatten(), 'r--', label='Estimated Audio Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('KAMP Audio Signal Recovery')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    nmse = np.linalg.norm(x_est - x_true) ** 2 / np.linalg.norm(x_true) ** 2
    print(f"Audio NMSE: {nmse:.4f}")

def run_distributed_example():
    """Test Distributed KAMP with uniform and varying node_max_iter on a random DAG."""
    m, n = 400, 50
    num_nodes = 5
    alpha, tau = 0.9, 0.1
    uniform_max_iter = 100
    varying_max_iter = [50, 100, 150, 200, 250]
    num_triggers = 10
    noise_level = 0.01
    random_state = 42
    
    # Generate sinusoidal data
    A, y, x_true = get_sinusoidal_data(m=m, n=n, sigma2=noise_level, random_state=random_state)
    
    # Split data across nodes
    rng = np.random.RandomState(random_state)
    indices = np.array_split(np.arange(m), num_nodes)
    A_list = [A[idx, :] for idx in indices]
    y_list = [y[idx, :] for idx in indices]
    
    # Create DAG
    G = DistributedKAMP.create_dag(num_nodes, edge_prob=0.3, random_state=random_state)
    
    # Helper function to run and visualize DistributedKAMP
    def run_and_visualize(node_max_iter, title_prefix, plot_filename):
        dkamp = DistributedKAMP(
            alpha=alpha, 
            tau=tau, 
            node_max_iter=node_max_iter, 
            num_triggers=num_triggers, 
            graph=G, 
            A_list=A_list, 
            y_list=y_list, 
            random_state=random_state
        )
        print(f"\n{title_prefix} Example: Initializing DistributedKAMP...")
        dkamp.fit()
        print(f"{title_prefix} Example: Fitting completed, retrieving estimates...")
        x_est = dkamp.solve()
        
        # Debug: Verify node estimates
        node_estimates = dkamp.get_node_estimates()
        for i, x_node in enumerate(node_estimates):
            print(f"Node {i} estimate shape: {x_node.shape if x_node is not None else 'None'}")
        
        # Compute residuals and metrics
        residuals = np.linalg.norm(y - A @ x_est, axis=1).reshape(-1, 1)
        anomalies = np.zeros_like(residuals)  # Placeholder for anomaly labels
        metrics = calculate_metrics(anomalies.flatten(), residuals.flatten())
        
        # Log results
        results = {f'Distributed KAMP ({title_prefix})': {'auc_roc': metrics.get('auc_roc', 0.0), 'time': 0.0}}
        log_results(results)
        
        # Plot global estimate
        print(f"\n{title_prefix} Example: x_true shape: {x_true.shape}, x_est shape: {x_est.shape}")
        plot_signal_comparison(x_true, x_est, f"Distributed KAMP ({title_prefix}) Recovery")
        nmse = np.linalg.norm(x_est - x_true) ** 2 / np.linalg.norm(x_true) ** 2
        print(f"Global NMSE ({title_prefix}): {nmse:.4f}")
        
        # Plot node estimates and overlay
        max_iter_list = [node_max_iter] * num_nodes if isinstance(node_max_iter, int) else node_max_iter
        plot_node_estimates(x_true, node_estimates, max_iter_list, title_prefix)
        
        # Visualize and save DAG
        print(f"{title_prefix} Example: Saving DAG plot to {plot_filename}...")
        dkamp.plot_graph(show=False)
        plt.savefig(plot_filename)
        plt.close()
    
    # Run tests
    print("\nTesting Distributed KAMP with Uniform node_max_iter...")
    run_and_visualize(uniform_max_iter, "Uniform", "dag_plot_uniform.png")
    
    print("\nTesting Distributed KAMP with Varying node_max_iter...")
    run_and_visualize(varying_max_iter, "Varying", "dag_plot_varying.png")

if __name__ == "__main__":
    #print("Running Synthetic Example...")
    #run_synthetic_example()
    #print("\nRunning High Dimensionality Example...")
    #run_high_dimensionality_example()
    #print("\nRunning High Noise Example...")
    #run_high_noise_example()
    #print("\nRunning Varying Sparsity Example...")
    #run_varying_sparsity_example()
    #print("\nRunning Image Example...")
    #run_image_example()
    #print("\nRunning Audio Example...")
    #run_audio_example()
    #print("\nRunning Distributed Example...")
    #run_distributed_example()
    print("\nRun phase transition experiment with default parameters...")
    run_phase_transition_experiment()