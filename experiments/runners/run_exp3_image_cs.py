import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ampire.core.kamp import KAMP
from ampire.core.amp import AMP
from ampire.distributed.distributed_kamp import DistributedKAMP
from ampire.utils.metrics import calculate_ssim
from experiments.datasets.image_loaders import load_lena  # Assuming this function exists

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
        # Single-node graph
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

    # Load image
    image = load_lena()  # Assume grayscale, shape (h, w)
    h, w = image.shape
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