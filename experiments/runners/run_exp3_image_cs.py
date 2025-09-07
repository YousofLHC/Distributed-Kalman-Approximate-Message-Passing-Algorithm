import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import image as mpimg

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ampire.core.kamp import KAMP
from ampire.core.amp import AMP
from ampire.distributed.distributed_kamp import DistributedKAMP
from ampire.utils.metrics import calculate_ssim

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
        # Use sophisticated DAG topology (best from exp5)
        num_nodes = 5  # Use 5 nodes for comparison
        G = DistributedKAMP.create_dag(num_nodes, edge_prob=0.4, random_state=42)

        # For image CS, split measurements across nodes
        A_list = []
        y_list = []
        measurements_per_node = A.shape[0] // num_nodes

        for i in range(num_nodes):
            start_idx = i * measurements_per_node
            end_idx = (i + 1) * measurements_per_node if i < num_nodes - 1 else A.shape[0]
            A_node = A[start_idx:end_idx]
            y_node = y[start_idx:end_idx]
            A_list.append(A_node)
            y_list.append(y_node)

        dkamp = DistributedKAMP(alpha=0.5, tau=0.1, node_max_iter=100, num_triggers=10, graph=G, A_list=A_list, y_list=y_list)
        dkamp.fit()

        # Plot and save topology
        out_dir = 'experiments/results/exp3_image_cs'
        os.makedirs(out_dir, exist_ok=True)

        # Save adjacency matrix
        adj_matrix = nx.to_numpy_array(G)
        adj_path = os.path.join(out_dir, 'topology_dag_adjacency.npy')
        np.save(adj_path, adj_matrix)

        # Plot topology
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
               node_size=500, font_size=16, font_weight='bold',
               arrows=True, arrowstyle='->', arrowsize=20)
        plt.title('Distributed KAMP Topology: DAG (5 nodes)')
        plot_path = os.path.join(out_dir, 'topology_dag_graph.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        x_hat = dkamp.solve()
    else:
        raise ValueError(f"Unknown method {method}")
    return x_hat

def run_experiment():
    results = []

    # Load image using matplotlib's built-in test image
    # Since scipy.misc.face is deprecated, we'll use a simple test image
    # Create a simple test image (you can replace this with actual image loading)
    h, w = 64, 64  # Reduced size for memory efficiency
    x_center, y_center = h // 2, w // 2

    # Create a simple circular pattern as test image
    y_coords, x_coords = np.ogrid[:h, :w]
    mask = (x_coords - x_center)**2 + (y_coords - y_center)**2 < (min(h, w) // 4)**2
    image = np.zeros((h, w))
    image[mask] = 1.0

    # Add some noise for more realistic testing
    np.random.seed(42)
    image += 0.1 * np.random.randn(h, w)
    image = np.clip(image, 0, 1)

    n = h * w
    x = image.flatten()

    # Sampling rates
    sampling_rates = [0.1, 0.2, 0.3, 0.5, 0.7]
    methods = ['AMP', 'KAMP', 'DistributedKAMP']

    for rate in tqdm(sampling_rates, desc="Sampling Rate"):
        m = int(rate * n)

        # Generate random measurement matrix
        A = np.random.randn(m, n) / np.sqrt(m)

        # Generate measurements
        y = A @ x + 0.01 * np.random.randn(m)
        y = y.reshape(-1, 1)  # Ensure 2D

        for method in methods:
            # Run solver
            x_hat = run_solver(method, A, y)

            # Reshape to image
            image_hat = x_hat.reshape(h, w)

            # Compute PSNR
            psnr = peak_signal_noise_ratio(image, image_hat, data_range=image.max() - image.min())

            # Compute SSIM
            ssim = calculate_ssim(image, image_hat)

            results.append({
                'sampling_rate': rate,
                'method': method,
                'psnr': psnr,
                'ssim': ssim
            })

            # Save results immediately after each experiment
            df_temp = pd.DataFrame(results)
            temp_csv_path = 'experiments/results/exp3_image_cs/exp3_results_partial.csv'
            os.makedirs(os.path.dirname(temp_csv_path), exist_ok=True)
            df_temp.to_csv(temp_csv_path, index=False)

    return results

def main():
    results = run_experiment()

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = 'experiments/results/exp3_image_cs/exp3_results.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    # Save to Excel
    excel_path = 'experiments/results/exp3_image_cs/exp3_results.xlsx'
    df.to_excel(excel_path, index=False)

    # Plot results for each method
    methods = df['method'].unique()
    fig, axes = plt.subplots(len(methods), 1, figsize=(10, 6 * len(methods)))
    if len(methods) == 1:
        axes = [axes]
    for i, method in enumerate(methods):
        ax = axes[i]
        subset = df[df['method'] == method]
        ax.plot(subset['sampling_rate'], subset['psnr'], label='PSNR')
        ax.plot(subset['sampling_rate'], subset['ssim'], label='SSIM')
        ax.set_xlabel('Sampling Rate')
        ax.set_ylabel('Metric Value')
        ax.set_title(f'Image CS Reconstruction - {method}')
        ax.legend()
    plot_path = 'experiments/results/exp3_image_cs/exp3_plot.png'
    plt.savefig(plot_path)
    plt.close()

    print(f"Results saved to {csv_path} and {excel_path}, plot to {plot_path}")

if __name__ == '__main__':
    main()