# src/ampire/tests/test_topology_plots.py
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
sys.path.insert(0, 'src')
from ampire.network.topology_generators import (
    create_dag,
    create_directed_with_cycles,
    create_directed_with_cycles_and_loops,
    create_star_with_extra_edges,
    create_leafs_to_final_node,
    create_tree_with_leaf_connections,
    create_bidirectional
)

def plot_graph(G, title, filename):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True, arrowstyle='->', arrowsize=20)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# List of generators
generators = [
    ('DAG', create_dag),
    ('Directed with Cycles', create_directed_with_cycles),
    ('Directed with Cycles and Loops', create_directed_with_cycles_and_loops),
    ('Star with Extra Edges', create_star_with_extra_edges),
    ('Leafs to Final Node', create_leafs_to_final_node),
    ('Tree with Leaf Connections', create_tree_with_leaf_connections),
    ('Bidirectional', create_bidirectional)
]

# Different parameter sets to generate variety
params_list = [
    {'num_nodes': 10, 'edge_prob': 0.3},
    {'num_nodes': 10, 'edge_prob': 0.7},
    {'num_nodes': 15, 'edge_prob': 0.5},
    {'num_nodes': 15, 'edge_prob': 0.9},
    {'num_nodes': 20, 'edge_prob': 0.4},
    {'num_nodes': 20, 'edge_prob': 0.8},
]

# Additional params for specific generators
extra_params = {
    'create_directed_with_cycles_and_loops': {'self_loop_prob': 0.1},
    'create_star_with_extra_edges': {'extra_edge_prob': 0.2},
    'create_tree_with_leaf_connections': {'leaf_edge_prob': 0.3, 'back_edge_prob': 0.1},
    'create_bidirectional': {'bidirectional_prob': 0.3}
}

# Create directory for plots if not exists
plots_dir = 'topology_plots'
os.makedirs(plots_dir, exist_ok=True)

graph_count = 0
for name, func in generators:
    func_name = func.__name__
    for params in params_list:
        # Merge with extra params if available
        full_params = params.copy()
        if func_name in extra_params:
            full_params.update(extra_params[func_name])
        try:
            G = func(**full_params)
            title = f"{name} - Nodes: {full_params.get('num_nodes', 'N/A')}, Edge Prob: {full_params.get('edge_prob', 'N/A')}"
            filename = os.path.join(plots_dir, f"graph_{graph_count:02d}_{name.replace(' ', '_')}.png")
            plot_graph(G, title, filename)
            graph_count += 1
            if graph_count >= 20:
                break
        except Exception as e:
            print(f"Error generating {name}: {e}")
    if graph_count >= 20:
        break

print(f"Generated {graph_count} graphs. Plots saved in '{plots_dir}' directory.")