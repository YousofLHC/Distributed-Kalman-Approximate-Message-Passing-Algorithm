import numpy as np
from ..distributed.distributed_kamp import DistributedKAMP
from ..network.graph import MyGraph, Node
from ..core.kamp import KAMP

def create_synthetic_data(num_nodes=5, n=10, m=20):
    """Create synthetic data for testing."""
    np.random.seed(42)
    x_true = np.random.randn(n)
    A_list = []
    y_list = []
    for i in range(num_nodes):
        A = np.random.randn(m, n)
        noise = 0.1 * np.random.randn(m)
        y = A @ x_true + noise
        A_list.append(A)
        y_list.append(y)
    return A_list, y_list, x_true

def test_distributed_kamp():
    """Test the refactored DistributedKAMP with MyGraph."""
    print("Testing DistributedKAMP with MyGraph...")

    # Create synthetic data
    num_nodes = 5
    A_list, y_list, x_true = create_synthetic_data(num_nodes)

    # Create graph using MyGraph
    graph = MyGraph()
    graph.add_nodes_from(range(num_nodes))
    # Add some edges
    graph.add_edge(0, 1, weight=0.5)
    graph.add_edge(1, 2, weight=0.7)
    graph.add_edge(2, 3, weight=0.6)
    graph.add_edge(3, 4, weight=0.8)
    graph.add_edge(0, 4, weight=0.4)

    # Initialize DistributedKAMP
    dkamp = DistributedKAMP(
        alpha=0.5,
        tau=0.1,
        node_max_iter=10,
        num_triggers=20,
        graph=graph,
        A_list=A_list,
        y_list=y_list,
        random_state=42,
        verbose=True
    )

    # Fit the model
    dkamp.fit()

    # Get results
    x_global = dkamp.solve()
    node_estimates = dkamp.get_node_estimates()

    print(f"True signal norm: {np.linalg.norm(x_true):.4f}")
    print(f"Global estimate norm: {np.linalg.norm(x_global):.4f}")
    print(f"Estimation error: {np.linalg.norm(x_global - x_true):.4f}")

    # Show edge weights
    print("\nEdge weights:")
    for u, v, data in graph.edges(data=True):
        print(f"Edge {u} -> {v}: weight = {data['weight']:.2f}")

    # Show information passing (node estimates)
    print("\nNode estimates:")
    for i, x in enumerate(node_estimates):
        print(f"Node {i}: norm = {np.linalg.norm(x):.4f}")

    # Plot the graph
    print("\nPlotting the graph...")
    dkamp.plot_graph(show=False)  # Set show=False to avoid blocking
    import matplotlib.pyplot as plt
    plt.savefig('distributed_kamp_graph.png')
    print("Graph plot saved as 'distributed_kamp_graph.png'")

    # Save adjacency matrix
    print("\nSaving adjacency matrix...")
    adj_matrix = graph.save_adjacency_matrix('distributed_kamp_adjacency')
    print("Adjacency matrix:")
    print(adj_matrix)

    return dkamp

if __name__ == "__main__":
    dkamp = test_distributed_kamp()