import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from ampire.core.amp import AMP
from ampire.core.kamp import KAMP
from ampire.distributed import DistributedKAMP
from ampire.network import create_tree_with_leaf_connections

# Set random seed for reproducibility
np.random.seed(42)

# ----------------------- Common parameters ----------------
n = 100  # Dimension of the signal
k = 10   # Sparsity level (number of non-zero elements)
m = 50   # Number of measurements (for single node)
sigma = 0.01  # Noise level
num_nodes = 5 # Number of nodes for DistributedKAMP
alpha = 0.5     # AMP/KAMP parameter
tau =  0.1         # AMP/KAMP parameter
max_iter =100    # Maximum iterations
tol =1e-8        # Tolerance
node_max_iter = 20 # Node iterations for DistributedKAMP
num_triggers = 50 # Triggers for DistributedKAMP

# --------------------- Generate Data --------------
# Generate sparse true signal x_true
x_true = np.zeros(n)
non_zero_indices = np.random.choice(n, k, replace=False)
x_true[non_zero_indices] = np.random.normal(1, 0.5, k)  # Non-zero values ~ N(1, 0.5)

# Generate measurement matrix A (Gaussian)
A = np.random.normal(0, 1/np.sqrt(m), (m, n))

# Generate observations y = A @ x_true + noise
noise = np.random.normal(0, sigma, m)
y = A @ x_true + noise

# --------------- Setup output directory ----------------
plot_dir = 'src/ampire/examples/results/plots'
os.makedirs(plot_dir, exist_ok=True)


# ----------------------- Helper function to evaluate and print results ----------------------
def evaluate_model(model, name, x_true, A, y):
    model.fit(A, y)
    x_est = model.solve().flatten()
    mse = mean_squared_error(x_true, x_est)
    print(f"\n=== Testing {name} ===")
    print(f"{name} MSE: {mse:.6f}")
    print(f"{name} Recovered Signal (first 10): {x_est[:10]}")
    print(f"True Signal (first 10): {x_true[:10]}")
    return x_est, mse


# ------------------- Test AMP -------------------
amp = AMP(alpha=0.5, tau=0.1, max_iter=100, tol=1e-8)
x_amp, mse_amp = evaluate_model(amp, "AMP", x_true, A, y)

# ------------------- Test KAMP -------------------
kamp = KAMP(alpha=0.5, tau=0.1, max_iter=100, tol=1e-8)
x_kamp, mse_kamp = evaluate_model(kamp, "KAMP", x_true, A, y)


# ------------------- Test DistributedKAMP -------------------
# Partition data into 3 nodes (m_per_node ≈ m/num_nodes)
m_per_node = m // num_nodes
A_list = [A[i*m_per_node:(i+1)*m_per_node] for i in range(num_nodes)]
y_list = [y[i*m_per_node:(i+1)*m_per_node] for i in range(num_nodes)]

# Create graph using create_tree_with_leaf_connections
graph = create_tree_with_leaf_connections(
    num_nodes=num_nodes,
    leaf_edge_prob=0.5,
    back_edge_prob=0.3,
    rng=np.random.RandomState(42)
)
nx.set_edge_attributes(graph, 0.5, 'weight')

print("\n=== Testing DistributedKAMP with create_tree_with_leaf_connections topology ===")
print(f"Graph edges: {list(graph.edges(data=True))}")
dkamp = DistributedKAMP(
    alpha=0.5,
    tau=0.1,
    node_max_iter=20,
    num_triggers=50,
    graph=graph,
    A_list=A_list,
    y_list=y_list,
    random_state=42,
    record_history=True,
    just_dag=False,  # Allow cycles due to back edges
    verbose=False
)

dkamp.fit()
x_dkamp = dkamp.solve().flatten()
mse_dkamp = mean_squared_error(x_true, x_dkamp)
print(f"DistributedKAMP MSE: {mse_dkamp:.6f}")
print(f"DistributedKAMP Global Recovered Signal (first 10): {x_dkamp[:10]}")
print(f"True Signal (first 10): {x_true[:10]}")

# Node-wise evaluation
node_estimates = dkamp.get_node_estimates()
node_mses = []
for i, x_node in enumerate(node_estimates):
    x_node = x_node.flatten()
    mse_node = mean_squared_error(x_true, x_node)
    node_mses.append(mse_node)
    print(f"Node {i} MSE: {mse_node:.6f}")
    print(f"Node {i} Recovered Signal (first 10): {x_node[:10]}")

# ------------------- Plotting -------------------
def save_plot(fig, filename, title, xlabel, ylabel, log_scale=False):
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True)
    if log_scale:
        plt.yscale('log')
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()
# Plot 1: Signal Recovery Comparison
plt.figure(figsize=(12, 6))
plt.plot(x_true, label='True Signal', color='black', linewidth=2, marker='o', markersize=4)
plt.plot(x_amp, label='AMP', linestyle='--', marker='s', markersize=4)
plt.plot(x_kamp, label='KAMP', linestyle='-.', marker='^', markersize=4)
plt.plot(x_dkamp, label='DistributedKAMP (Global)', linestyle=':', marker='d', markersize=4)
save_plot(plt, 'signal_recovery_comparison.png', 'Signal Recovery Comparison', 'Index', 'Signal Value')

# Plot 2: Node-wise Signal Recovery for DistributedKAMP
plt.figure(figsize=(12, 6))
plt.plot(x_true, label='True Signal', color='black', linewidth=2, marker='o', markersize=4)
for i, x_node in enumerate(node_estimates):
    x_node = x_node.flatten()
    plt.plot(x_node, label=f'DistributedKAMP Node {i}', linestyle='--', marker='s', markersize=4)
save_plot(plt, 'dkamp_nodewise_recovery.png', 'DistributedKAMP: Node-wise Signal Recovery', 'Index', 'Signal Value')

# Plot 3: Consensus Error History for DistributedKAMP
plt.figure(figsize=(10, 5))
plt.plot(dkamp.history, label='Consensus Error', color='blue', marker='o', markersize=4)
save_plot(plt, 'dkamp_consensus_error.png', 'DistributedKAMP: Consensus Error Over Triggers', 
          'Trigger Iteration', 'Consensus Error (Variance)', log_scale=True)

# Plot 4: Graph Visualization for DistributedKAMP
dkamp.plot_graph(show=False)
plt.savefig(os.path.join(plot_dir, 'dkamp_graph.png'))
plt.close()

# Save results
results = {
    'Centralized AMP MSE': mse_amp,
    'Centralized KAMP MSE': mse_kamp,
    'Distributed KAMP MSE': mse_dkamp,
    'Node MSEs': node_mses,
    'Consensus Error History (last 5)': dkamp.history[-5:]
}
print("\n=== Comparison ===")
with open(os.path.join(plot_dir, 'mse_results.txt'), 'w') as f:
    for key, value in results.items():
        print(f"{key}: {value}")
        f.write(f"{key}: {value}\n")

print(f"\nPlots and results saved to {plot_dir}")