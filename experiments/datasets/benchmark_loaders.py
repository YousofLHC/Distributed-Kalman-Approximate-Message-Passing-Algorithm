import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
from sklearn.datasets import make_classification
try:
    from data import DataReader
    USE_DATAREADER = True
except ImportError:
    USE_DATAREADER = False

def load_odds_dataset(name):
    """
    Load ODDS dataset from Information-Fusion.

    Parameters:
        name (str): Dataset name (e.g., 'wine', 'lympho').

    Returns:
        tuple: (X, y) where X is features, y is labels.
    """
    file_path = Path(__file__).parent.parent.parent / 'data' / 'benchmark' / 'Information-Fusion' / f'{name}.mat'
    if file_path.exists():
        if USE_DATAREADER:
            dr = DataReader(str(file_path))
            data = dr()
            X = data['X']
            y = data['y'].flatten()
        else:
            data = loadmat(file_path)
            X = data['X']
            y = data['y'].flatten()
        # Convert to binary: 1 for anomaly, 0 for normal
        y = (y == 1).astype(int)
        return X, y
    else:
        # Fallback to synthetic
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
        y = np.where(np.random.rand(n_samples) < 0.05, 1, 0)
        return X, y

def load_keel_dataset(name):
    """
    Load KEEL dataset.

    Parameters:
        name (str): Dataset name (e.g., 'iris0', 'winequality-white-9_vs_4').

    Returns:
        tuple: (X, y) where X is features, y is labels.
    """
    file_path = Path(__file__).parent.parent.parent / 'data' / 'benchmark' / 'KEEL' / f'{name}.csv'
    if file_path.exists():
        if USE_DATAREADER:
            dr = DataReader(str(file_path))
            df = dr()
        else:
            df = pd.read_csv(file_path)
        # Assume last column is label, others are features
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        # Convert to binary: 1 for anomaly, 0 for normal
        y = (y == 1).astype(int) if y.dtype == int else (y == 'anomaly').astype(int)
        return X, y
    else:
        # Fallback to synthetic
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
        y = np.where(np.random.rand(n_samples) < 0.05, 1, 0)
        return X, y