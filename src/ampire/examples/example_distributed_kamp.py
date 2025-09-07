import sys
import numpy as np
from pathlib import Path

# Add the src directory to sys.path to enable imports from the ampire package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from ampire.distributed.distributed_kamp import DistributedKAMP
from ampire.network.graph import MyGraph, Node
from ampire.core.kamp import KAMP
import yaml
from ampire.network.topology_generators import (
    create_dag, create_directed_with_cycles, create_directed_with_cycles_and_loops,
    create_star_with_extra_edges, create_leafs_to_final_node, create_tree_with_leaf_connections,
    create_bidirectional
)

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
    topologies = load_topologies('configs/topologies.yaml')
    node_counts = [5, 10, 15, 25, 30, 50, 60, 100, 150, 200]
    max_num_nodes = max(node_counts)
    n = 700  # dimension of the signal
    m = 5000  # number of samples per measurement
    # generate synthetic data once for all topologies
    A_list_full, y_list_full, x_true = create_synthetic_data(num_nodes=max_num_nodes, n=n, m=m)
    results = []
    for topo in topologies:
        for num_nodes in node_counts:
            if num_nodes > m:
                continue  # skip if more nodes than samples
            # use the pre-generated synthetic data
            A_list = A_list_full[:num_nodes]
            y_list = y_list_full[:num_nodes]
            # create graph using topology
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
            import matplotlib.pyplot as plt
            plt.savefig(figure_path)
            # save adjacency matrix
            adj_path = f"adjacency/{topo['name']}_nodes{num_nodes}.npy"
            Path('adjacency').mkdir(exist_ok=True)
            np.save(adj_path, graph.get_adjacency_matrix())
            # compute summary metrics
            report = dkamp.report()
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

if __name__ == "__main__":
    main()