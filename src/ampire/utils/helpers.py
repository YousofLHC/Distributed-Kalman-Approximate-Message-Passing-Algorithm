import numpy as np
import os
import glob
import json
import time
import logging
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from sklearn.metrics import mean_squared_error
from ampire.core.amp import AMP
from ampire.core.kamp import KAMP
from ampire.distributed.distributed_kamp import DistributedKAMP
from ampire.network.graph import MyGraph
from ampire.utils.metrics import safe_divide, DEFAULT_EPS
def generate_synthetic_data(num_samples, num_features, sparsity, noise_std, random_state=42, is_2d=False, shape_2d=None):
    """Generate synthetic data for sparse signal recovery."""
    rng = np.random.RandomState(random_state)
    if is_2d:
        x_true = rng.randn(*shape_2d)
        mask = rng.rand(*shape_2d) < sparsity
        x_true[~mask] = 0
        x_true_flat = x_true.flatten()
        num_features = x_true_flat.shape[0]
    else:
        x_true = rng.randn(num_features)
        mask = rng.rand(num_features) < sparsity
        x_true[~mask] = 0
        x_true_flat = x_true
        x_true = None

    A = rng.randn(num_samples, num_features)
    noise = rng.normal(0, noise_std, num_samples)
    y = A @ x_true_flat + noise
    return A, y, x_true_flat, x_true


def create_measurement_matrix(m, n, matrix_type, random_state=None):
    """
    Generate a measurement matrix of specified type and dimensions.

    Parameters
    ----------
    m : int
        Number of measurements (rows of the matrix).
    n : int
        Number of features (columns of the matrix).
    matrix_type : str
        Type of measurement matrix to generate. Options are:
        - 'gaussian': Gaussian random matrix with entries ~ N(0, 1/m).
        - 'heavy': Heavy-tailed random matrix with entries from a t-distribution (df=2) scaled by 1/sqrt(m).
        - 'orthogonal': Orthogonal matrix derived from QR decomposition, with m rows randomly selected and scaled.
    random_state : int or None, optional
        Seed for the random number generator to ensure reproducibility. Default is None.

    Returns
    -------
    ndarray
        Measurement matrix of shape (m, n).

    Raises
    ------
    ValueError
        If matrix_type is not one of 'gaussian', 'heavy', or 'orthogonal'.
        If m or n is not a positive integer.
        If m > n for orthogonal matrix type (since orthogonal requires selecting m rows from n).
    """
    if not isinstance(m, int) or m <= 0:
        raise ValueError("Number of measurements (m) must be a positive integer.")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Number of features (n) must be a positive integer.")
    
    rng = np.random.default_rng(random_state)

    if matrix_type == "gaussian":
        return rng.randn(m, n) / np.sqrt(m)
    elif matrix_type == "heavy":
        return rng.standard_t(df=2, size=(m, n)) / np.sqrt(m)
    elif matrix_type == "orthogonal":
        if m > n:
            raise ValueError("For orthogonal matrix type, m must be less than or equal to n.")
        B = rng.randn(n, n)
        Q, _ = np.linalg.qr(B)
        idx = rng.choice(n, size=m, replace=False)
        return Q[idx, :] * np.sqrt(n / m)
    else:
        raise ValueError("Unknown matrix type. Choose from 'gaussian', 'heavy', or 'orthogonal'.")

def load_graph(adj_file):
    """Load graph from adjacency matrix file."""
    try:
        adj = np.loadtxt(adj_file)
        G = MyGraph()
        G.add_nodes_from(range(adj.shape[0]))
        for i, j in zip(*np.where(adj == 1)):
            G.add_edge(i, j)
        return G, adj.shape[0]
    except Exception as e:
        logging.error(f"Error loading topology {adj_file}: {e}")
        print(f"Warning: Error loading topology {adj_file}: {e}")
        return None, None
    
def evaluate_model(model, name, x_true, A, y):
    """Evaluate a model and print results."""
    try:
        model.fit(A, y)
        x_est = model.solve().flatten()
        mse = mean_squared_error(x_true, x_est)
        print(f"\n=== Testing {name} ===")
        print(f"{name} MSE: {mse:.6f}")
        print(f"{name} Recovered Signal (first 10): {x_est[:10]}")
        print(f"True Signal (first 10): {x_true[:10]}")
        return x_est, mse
    except Exception as e:
        logging.error(f"Error evaluating {name}: {e}")
        print(f"Warning: Error evaluating {name}: {e}")
        return None, float('inf')
    
def save_plot(fig, filepath, title, xlabel, ylabel, log_scale=False):
    """Save a matplotlib plot."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if log_scale:
        plt.yscale('log')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()