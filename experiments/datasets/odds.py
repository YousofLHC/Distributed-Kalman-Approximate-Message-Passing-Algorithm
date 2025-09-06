import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler

def load_odds_dataset(name):
    if name.lower() == "kddcup99":
        data = fetch_kddcup99(subset="SA", percent10=True, shuffle=True, random_state=0)
        X = data.data
        # Basic hashing trick for categorical → numeric if needed
        X = np.array([
            np.fromiter((hash(v) if isinstance(v, (bytes, str)) else v for v in row), dtype=np.float64)
            for row in X
        ])
        y = (data.target != b'normal.').astype(int)
        X = StandardScaler().fit_transform(X)
        return X, y
    raise NotImplementedError("Add more ODDS loaders as needed (Arrhythmia, SECOM, ...).")
