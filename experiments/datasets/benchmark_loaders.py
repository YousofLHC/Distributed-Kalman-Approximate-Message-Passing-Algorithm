import numpy as np
from sklearn.datasets import make_classification

def load_odds_dataset(name):
    """
    Load or generate ODDS dataset.

    Parameters:
        name (str): Dataset name.

    Returns:
        tuple: (X, y) where X is features, y is labels.
    """
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_informative = 5
    n_redundant = 2
    n_clusters_per_class = 1

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=n_clusters_per_class,
        random_state=42
    )
    # Make it anomaly detection: few anomalies
    y = np.where(np.random.rand(n_samples) < 0.05, 1, 0)  # 5% anomalies
    return X, y

def load_keel_dataset(name):
    """
    Load or generate KEEL dataset.

    Parameters:
        name (str): Dataset name.

    Returns:
        tuple: (X, y) where X is features, y is labels.
    """
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_informative = 5
    n_redundant = 2
    n_clusters_per_class = 1

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=n_clusters_per_class,
        random_state=42
    )
    # Make it anomaly detection: few anomalies
    y = np.where(np.random.rand(n_samples) < 0.05, 1, 0)  # 5% anomalies
    return X, y