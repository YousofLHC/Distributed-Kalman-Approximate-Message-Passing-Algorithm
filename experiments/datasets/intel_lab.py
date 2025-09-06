import numpy as np
import pandas as pd
from pathlib import Path

def load_intel_timeseries():
    # Expect CSVs under data/intel_lab/*.csv (e.g., temperature per node)
    root = Path("data/intel_lab")
    files = sorted(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError("Place Intel Lab CSVs under data/intel_lab/")

    data = {}
    for f in files:
        df = pd.read_csv(f)
        node = int(f.stem.split("_")[-1])
        vals = df["temperature"].astype(float).values
        win = 5
        if len(vals) <= win:
            continue
        X = np.vstack([vals[i-win:i] for i in range(win, len(vals))])
        y = vals[win:]
        A_i = X  # simple AR(1..win) design; can be refined
        data[node] = (A_i, y)
    return data
