import matplotlib.pyplot as plt
import numpy as np

def plot_signal_comparison(x_true: np.ndarray, x_est: np.ndarray, title: str = "Signal Comparison"):
    """
    Plot true vs. estimated signals.

    Parameters
    ----------
    x_true : np.ndarray
        True signal.
    x_est : np.ndarray
        Estimated signal.
    title : str, default="Signal Comparison"
        Plot title.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(x_true.flatten(), 'bo-', label='True Signal')
    plt.plot(x_est.flatten(), 'rx--', label='Estimated Signal')
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()