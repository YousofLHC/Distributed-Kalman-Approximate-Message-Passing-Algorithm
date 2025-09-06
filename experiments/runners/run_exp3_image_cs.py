import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio
import sys
import os
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ampire.core.kamp import KAMP
from ampire.utils.metrics import calculate_ssim
from experiments.datasets.image_loaders import load_lena  # Assuming this function exists

def run_experiment():
    results = []

    # Load image
    image = load_lena()  # Assume grayscale, shape (h, w)
    h, w = image.shape
    n = h * w
    x = image.flatten()

    # Sampling rates
    sampling_rates = [0.1, 0.2, 0.3, 0.5, 0.7]

    for rate in tqdm(sampling_rates, desc="Sampling Rate"):
        m = int(rate * n)

        # Generate random measurement matrix
        A = np.random.randn(m, n) / np.sqrt(m)

        # Generate measurements
        y = A @ x + 0.01 * np.random.randn(m)
        y = y.reshape(-1, 1)  # Ensure 2D for KAMP

        # Run KAMP
        kamp = KAMP(alpha=0.5, tau=0.1, max_iter=100)
        kamp.fit(A, y)
        x_hat = kamp.solve()

        # Reshape to image
        image_hat = x_hat.reshape(h, w)

        # Compute PSNR
        psnr = peak_signal_noise_ratio(image, image_hat, data_range=image.max() - image.min())

        # Compute SSIM
        ssim = calculate_ssim(image, image_hat)

        results.append({
            'sampling_rate': rate,
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

    print(f"Results saved to {csv_path} and {excel_path}")

if __name__ == '__main__':
    main()