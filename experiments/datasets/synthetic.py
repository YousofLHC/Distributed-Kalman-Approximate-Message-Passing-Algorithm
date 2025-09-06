import pandas as pd
import numpy as np
from pathlib import Path

def load_ring_dataset():
    """
    Load the ring synthetic dataset.

    Returns:
        tuple: (X, y) where X is features, y is labels (1 for normal, -1 for anomaly).
    """
    file_path = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / 'ring.csv'
    df = pd.read_csv(file_path)
    X = df[['x', 'y']].values
    y = df['class'].values
    return X, y

def load_two_circle_dataset():
    """
    Load the two circle synthetic dataset.

    Returns:
        tuple: (X, y) where X is features, y is labels (1 for normal, -1 for anomaly).
    """
    file_path = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / 'twoCircle.csv'
    df = pd.read_csv(file_path)
    X = df[['x', 'y']].values
    y = df['class'].values
    return X, y

def load_synthetic_dataset(name):
    """
    Load a synthetic dataset by name.

    Parameters:
        name (str): Dataset name ('ring' or 'twoCircle').

    Returns:
        tuple: (X, y) where X is features, y is labels.
    """
    if name == 'ring':
        return load_ring_dataset()
    elif name == 'twoCircle':
        return load_two_circle_dataset()
    else:
        raise ValueError(f"Unknown synthetic dataset: {name}")