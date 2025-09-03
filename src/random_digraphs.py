import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def create_strongly_connected_graph(num_nodes, self_loop_prob=0.5, random_state=None):
    # Create a random number generator with the specified seed
    rng = np.random.RandomState(random_state)
    
    # Generate a random directed graph
    G = nx.gnp_random_graph(num_nodes, 0.5, directed=True, seed=rng)
    
    # Ensure the graph is strongly connected by adding edges if necessary
    while not nx.is_strongly_connected(G):
        u, v = rng.choice(range(num_nodes), size=2, replace=False)
        G.add_edge(u, v)
    
    # Add self-loops randomly with probability self_loop_prob
    for i in range(num_nodes):
        if rng.random() < self_loop_prob:
            G.add_edge(i, i)
    
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
    