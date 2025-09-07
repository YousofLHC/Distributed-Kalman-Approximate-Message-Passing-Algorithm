# src/ampire/network/topology_generators.py
import networkx as nx
import numpy as np

def create_dag(num_nodes: int, edge_prob: float = 0.3, rng=None) -> nx.DiGraph:
    rng = rng or np.random.RandomState()
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < edge_prob:
                G.add_edge(i, j)
    return G

def create_directed_with_cycles(num_nodes: int, edge_prob: float = 0.3, rng=None) -> nx.DiGraph:
    rng = rng or np.random.RandomState()
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and rng.random() < edge_prob:
                G.add_edge(i, j)
    return G

def create_directed_with_cycles_and_loops(num_nodes: int, edge_prob: float = 0.3,
                                          self_loop_prob: float = 0.05, rng=None) -> nx.DiGraph:
    G = create_directed_with_cycles(num_nodes, edge_prob, rng)
    rng = rng or np.random.RandomState()
    for i in range(num_nodes):
        if rng.random() < self_loop_prob:
            G.add_edge(i, i)
    return G

def create_star_with_extra_edges(num_nodes: int, extra_edge_prob: float = 0.1, rng=None) -> nx.DiGraph:
    """A star directed outward from node 0; outer nodes may connect among themselves."""
    rng = rng or np.random.RandomState()
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    # star edges: from center (0) to all others
    for j in range(1, num_nodes):
        G.add_edge(0, j)
    # possible extra edges between outer nodes
    for i in range(1, num_nodes):
        for j in range(1, num_nodes):
            if i != j and rng.random() < extra_edge_prob:
                G.add_edge(i, j)
    return G

def create_leafs_to_final_node(num_nodes: int) -> nx.DiGraph:
    """A tree where every leaf connects to a final sink node (last index)."""
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    # build a simple directed tree (chain) from 0 to n-2
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1)
    # connect all leaves (here just node n-2) to the final node n-1
    # if you want multiple leaves, make a branching structure; here for simplicity:
    for i in range(num_nodes - 1):
        G.add_edge(i, num_nodes - 1)
    return G

def create_tree_with_leaf_connections(num_nodes: int, leaf_edge_prob: float = 0.2,
                                      back_edge_prob: float = 0.1, rng=None) -> nx.DiGraph:
    """
    Build a directed tree (root 0). Leaves may connect among themselves (leaf_edge_prob)
    and may have edges back toward ancestors (back_edge_prob).
    """
    rng = rng or np.random.RandomState()
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    # create a simple tree: parent i/2 -> i
    for i in range(1, num_nodes):
        G.add_edge((i - 1) // 2, i)
    # identify leaves
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
    # add random edges among leaves
    for i in leaves:
        for j in leaves:
            if i != j and rng.random() < leaf_edge_prob:
                G.add_edge(i, j)
    # add random back edges from leaves up to ancestors
    for i in leaves:
        ancestor = (i - 1) // 2
        if ancestor >= 0 and rng.random() < back_edge_prob:
            G.add_edge(i, ancestor)
    return G

def create_bidirectional(num_nodes: int, bidirectional_prob: float = 0.2, rng=None) -> nx.DiGraph:
    """
    Create a directed graph where some pairs of nodes have edges in both directions.
    """
    rng = rng or np.random.RandomState()
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    # random edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and rng.random() < bidirectional_prob:
                G.add_edge(i, j)
                G.add_edge(j, i)
    return G
