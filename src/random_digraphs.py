import networkx as nx
import numpy as np
from graph import MyGraph, Node
from kamp import KAMP
import matplotlib.pyplot as plt

def create_strongly_connected_graph(num_nodes    : int,
                                 edge_prob     : float = 0.5,
                                 self_loop_prob: float = 0.0,
                                 random_state  : int   = None) -> MyGraph:
    """
    Create a strongly connected directed acyclic graph (DAG).

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edge_prob : float, default=0.5
        Probability of edge creation.
    random_state : int, optional
        Random seed.

    Returns
    -------
    MyGraph
        Generated DAG.
    """
    rng = np.random.RandomState(random_state)
    
    # Generate a random directed graph
    G = nx.gnp_random_graph(n=num_nodes, p=edge_prob, directed=True, seed=rng)
    
    # Ensure the graph is strongly connected by adding edges if necessary
    while not nx.is_strongly_connected(G):
        u, v = rng.choice(range(num_nodes), size=2, replace=False)
        G.add_edge(u, v)
    
    if self_loop_prob:
        for i in range(num_nodes):
            if rng.random() < self_loop_prob:
                G.add_edge(i, i)
    
    return G
# -------------- Grok version ---------
    G = MyGraph()
    nodes = [Node(estimator=KAMP(alpha=0.5, tau=0.1, max_iter=100)) for _ in range(num_nodes)]
    G.add_nodes_from(nodes)
    
    # Create a DAG by ensuring edges point from lower to higher indexed nodes
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < edge_prob:
                G.add_edge(nodes[i], nodes[j], weight=rng.random())
    
    return G


if __name__ == "__main__":
    for i in range(1):
        G_nx = create_strongly_connected_graph(5, 0.1,42)
        nx.draw(G_nx, with_labels=True, 
                node_color='skyblue', font_weight='bold', node_size=800, font_size=12)
        plt.show()
        adj_matrix = nx.to_numpy_array(G_nx)
        print('#'*45)
        print((adj_matrix))
        print('#'*45)