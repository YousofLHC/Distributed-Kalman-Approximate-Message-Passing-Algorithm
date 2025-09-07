import sys
import numpy as np
from pathlib import Path
import os
import networkx as nx

# Add the src directory to sys.path to enable imports from the ampire package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from ampire.distributed.distributed_kamp import DistributedKAMP
from ampire.network.graph import MyGraph, Node
from ampire.core.kamp import KAMP
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from ampire.network.topology_generators import (
    create_dag, create_directed_with_cycles, create_directed_with_cycles_and_loops,
    create_star_with_extra_edges, create_leafs_to_final_node, create_tree_with_leaf_connections,
    create_bidirectional
)

def plot_graph(G, title, filename):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True, arrowstyle='->', arrowsize=20)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def save_adjacency_matrix(G, filename):
    adj = nx.to_numpy_array(G, dtype=int)
    np.savetxt(filename, adj, fmt='%d')

def generate_topology_plots():
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
                plot_filename = os.path.join(plots_dir, f"graph_{graph_count:02d}_{name.replace(' ', '_')}.png")
                matrix_filename = os.path.join(plots_dir, f"graph_{graph_count:02d}_{name.replace(' ', '_')}_adj.txt")
                plot_graph(G, title, plot_filename)
                save_adjacency_matrix(G, matrix_filename)
                graph_count += 1
                if graph_count >= 20:
                    break
            except Exception as e:
                print(f"Error generating {name}: {e}")
        if graph_count >= 20:
            break

    print(f"Generated {graph_count} graphs. Plots and adjacency matrices saved in '{plots_dir}' directory.")

def load_topologies(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)['topologies']
    
def get_graph(topology_spec, num_nodes, rng):
    t = topology_spec['type']
    params = topology_spec.get('params', {})
    if t == 'dag':
        return MyGraph(create_dag(num_nodes, **params, rng=rng))
    elif t == 'directed_with_cycles':
        return MyGraph(create_directed_with_cycles(num_nodes, **params, rng=rng))
    elif t == 'directed_with_cycles_and_loops':
        return MyGraph(create_directed_with_cycles_and_loops(num_nodes, **params, rng=rng))
    elif t == 'star_with_extra_edges':
        return MyGraph(create_star_with_extra_edges(num_nodes, **params, rng=rng))
    elif t == 'leafs_to_final_node':
        return MyGraph(create_leafs_to_final_node(num_nodes, **params, rng=rng))
    elif t == 'tree_with_leaf_connections':
        return MyGraph(create_tree_with_leaf_connections(num_nodes, **params, rng=rng))
    elif t == 'bidirectional':
        return MyGraph(create_bidirectional(num_nodes, **params, rng=rng))
    else:
        raise ValueError(f"Unknown topology type: {t}")


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

def main():
    # Generate topology plots
    generate_topology_plots()

    config_path = Path(__file__).parent.parent.parent.parent / 'configs' / 'topologies.yaml'
    topologies = load_topologies(str(config_path))
    node_counts = [5, 10, 15, 25, 30, 50, 60, 100, 150, 200]
    n = 70  # dimension of the signal
    m = 800  # number of samples per measurement
    results = []
    for num_nodes in tqdm(node_counts, desc="Processing node counts"):
        print(f"\nProcessing {num_nodes} nodes...")
        # generate synthetic data for this num_nodes
        A_list, y_list, x_true = create_synthetic_data(num_nodes=num_nodes, n=n, m=m)
        for topo in topologies:
            print(f"  Topology: {topo['name']}")
            rng = np.random.RandomState(42)
            graph = get_graph(topo, num_nodes, rng)
            dkamp = DistributedKAMP(alpha=0.5, tau=0.1,
                            node_max_iter=10,
                            num_triggers=20,
                            graph=graph,
                            A_list=A_list,
                            y_list=y_list,
                            random_state=42,
                            verbose=False,
                            record_history=True)
            dkamp.fit()
            # save per-trigger history
            history_file = f"logs/{topo['name']}_nodes{num_nodes}_history.json"
            Path('logs').mkdir(exist_ok=True)
            dkamp.save_history(history_file)
            # save graph figure
            figure_path = f"graphs/{topo['name']}_nodes{num_nodes}.png"
            Path('graphs').mkdir(exist_ok=True)
            dkamp.plot_graph(show=False)
            plt.savefig(figure_path)
            # save adjacency matrix
            adj_path = f"adjacency/{topo['name']}_nodes{num_nodes}.npy"
            Path('adjacency').mkdir(exist_ok=True)
            np.save(adj_path, graph.get_adjacency_matrix())
            # compute summary metrics
            report = dkamp.report()
            print(f"    Consensus error: {report['consensus_error']}")
            print(f"    Bytes: {report['bytes']}")
            results.append({
                'topology': topo['name'],
                'num_nodes': num_nodes,
                'consensus_error': report['consensus_error'],
                'bytes': report['bytes']
            })
    # choose the three best topologies by consensus error
    results_sorted = sorted(results, key=lambda r: r['consensus_error'])
    best_three = results_sorted[:3]
    with open('best_topologies.json', 'w') as f:
        import json
        json.dump(best_three, f, indent=2)
    print("\nBest topologies saved to best_topologies.json")

if __name__ == "__main__":
    main()