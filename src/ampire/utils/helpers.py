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
        return rng.normal(0, 1 / np.sqrt(m), size=(m, n))
    elif matrix_type == "heavy":
        return rng.standard_t(df=2, size=(m, n)) / np.sqrt(m)
    elif matrix_type == "orthogonal":
        if m > n:
            raise ValueError("For orthogonal matrix type, m must be less than or equal to n.")
        B = rng.normal(0, 1, size=(n, n))
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
    
    
    
from   pathlib               import Path # to create directory
import numpy                 as     np
from   sklearn.base          import TransformerMixin
from   sklearn               import preprocessing
from   sklearn               import model_selection
from   pathlib               import Path
import os
import __main__ as main
#from   sklearn.preprocessing import *
import re
import pickle
import itertools
from itertools import pairwise
def find_almost_boundaries(z,quantile):
    #remove quantile*z, the rests are near boundary z
    val = np.quantile(z,quantile) 
    count, edges = np.histogram(z)
    
    bin = next(((edges[i-1],edges[-1]) for i,edge in enumerate(edges) if edge>val),None)# select 1-quantile of data
    lower, upper = bin
    mask = (np.logical_and(z>=lower, z<upper).flatten())
    return mask

def write_report(obj, path):
    #suffix = '-poly' if 'poly' in main.__file__ else ''
    empty = []
    
    suffix = re.findall(pattern="run_(.*?).py", string=main.__file__)
    suffix = "" if suffix==empty else '_'+suffix[0]

    with open(f'{path}{suffix}.pkl','wb') as f:
        pickle.dump(obj=obj, file=f, protocol=pickle.HIGHEST_PROTOCOL)

def extract_directory_name_from_file_path(file_path:str, path_seprator:str='/'):
    path           = file_path.split('.')[-2]
    directory_name = path.split(path_seprator)[-1]
    return directory_name

def create_directory(directory:str, des:str='Results/'):
    dir_path = os.path.join(des, directory)
    return Path(dir_path).mkdir(parents=True, exist_ok=True)

def create_report_file(directory:str, file:str, des:str='results/')->Path:
    file_path = os.path.join(des, directory, file)
    Path(file_path).touch(exist_ok=True)
    return file_path


def make_inlier_outlier_label(
                                    labels:np.ndarray[int],
                                    inlier:int=0,
                                    target:int=+1
                                )->np.ndarray[int]:
    """
    Returns binary class of labels target(inlier)(+1), non-target(-1)
    if target=0, output labels is 0 for targets and +1 for non-targets(anomalies)
    
    Parameters
    -----------
        labels   : np.ndarray[int]
            1-D array of (binary/~multi~) class labels
        inlier : int 
            inlier label (class) in labels array
        target   : int
            convert inlier labels to target
            (default is +1)
    
    Returns
    ----------
        np.ndarray[int]
        binary class labels target(+1), non-target(-1)
    """
    output         = np.zeros(shape=labels.shape, dtype=int)
    mask           = labels==inlier
    target_is_zero = target==0
    output[~mask]  = +1 if target_is_zero else -target #non-targets
    output[mask]   = target
    return output


def preprocess(scaler:TransformerMixin,data:np.ndarray, *args, **kwargs):
    transformer_class_name = scaler.__qualname__
    
    transformer_object     = getattr(preprocessing,transformer_class_name)(*args, **kwargs)

    return transformer_object.fit_transform(data)

def train_test_split(
                        X:np.ndarray,
                        y:np.ndarray,
                        target:int=+1,
                        *args,
                        **kwargs,
                    )-> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    mask_target  = y==target
    mask_target  = mask_target.flatten()
    X_target     = X[mask_target,:]
    X_non_target = X[~mask_target,:]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                        X_target,
                                        y[mask_target],
                                        *args,
                                        **kwargs,
                                        )
    X_test = np.concatenate((X_non_target, X_test))
    y_test = np.concatenate((y[~mask_target],y_test))

    return X_train, X_test, y_train, y_test
    
    

def parameter_permutation(params): #params Dict[List]
    keys, values = zip(*params.items())
    permutation  = [
        dict(zip(keys, v))
        for v in itertools.product(*values)
    ]
    return permutation # List[Dict]