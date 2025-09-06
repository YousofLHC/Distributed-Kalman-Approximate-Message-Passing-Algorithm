from experiments.datasets.intel_lab import load_intel_timeseries

data = load_intel_timeseries()
print("Loaded data for nodes:", list(data.keys()))
for node, (A, y) in data.items():
    print(f"Node {node}: A shape {A.shape}, y shape {y.shape}")