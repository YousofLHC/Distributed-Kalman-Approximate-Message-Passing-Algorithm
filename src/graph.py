import networkx as nx
import numpy as np
from itertools import count
import matplotlib.pyplot as plt

class Node:
    """Node class for graph-based distributed computation."""
    node_index = count(0)
    
    def __init__(self, estimator=None, name: str = None, **kwargs):
        self.estimator = estimator
        self._id = next(self.node_index) # first node_index then node_index+=1
        self.name = f"node_{self._id}" if name is None else name
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __hash__(self):
        return self._id

    def __repr__(self):
        return self.name # f"{self.__class__.__qualname__}({self._id})"

    def __getattr__(self, attr): # Delegate to estimator if the attribute is not found on Node
        if hasattr(self.estimator, attr):
            return getattr(self.estimator, attr)
        raise AttributeError(f"Node and its `estimator` have no attribute '{attr}'")
    
    
    @property
    def index(self):
         return self._id
    @property 
    def id(self):
        return self._id

class MyGraph(nx.DiGraph):
    """Directed acyclic graph for distributed computation."""
    
    def total_in_degree(self, node: Node, inplace: bool = True) -> 'MyGraph':
        """
        Aggregate estimates and covariances from predecessors.

        Parameters
        ----------
        node : Node
            Target node.
        inplace : bool, default=True
            Update node attributes if True.

        Returns
        -------
        MyGraph
            Updated graph.
        """
        total_x = 0
        total_P = 0
        if self.in_degree(node) == 0:
            self.nodes[node]['x'] = self.nodes[node].get('x', node.x)
            self.nodes[node]['P'] = self.nodes[node].get('P', node.P)
            return self
        
        # Normalize incoming weights
        total_weight = self.in_degree(node, weight='weight')
        if total_weight > 0:
            for j in self.predecessors(node):
                self[node][j]['weight'] /= total_weight
        
        # Aggregate messages
        for u, v, data in self.in_edges(node, data=True):
            w = data.get('weight', 0)
            total_x += w * data.get('x', 0)
            total_P += w * data.get('P', u.P)
        
        if inplace:
            self.nodes[node]['x'] = total_x
            self.nodes[node]['P'] = total_P
        return self

    def trigger(self, node: Node, inplace: bool = True, **attr) -> 'MyGraph':
        """
        Trigger local computation at the node.

        Parameters
        ----------
        node : Node
            Target node.
        inplace : bool, default=True
            Update node attributes if True.

        Returns
        -------
        MyGraph
            Updated graph.
        """
        self.total_in_degree(node, inplace, **attr)
        node.solve()
        return self

    def plot(self, show: bool = True):
        """Visualize the graph."""
        pos = nx.spring_layout(self, seed=42)
        weights = np.array([d.get('weight', 0) for _, _, d in self.edges(data=True)])
        cmap_edges = plt.cm.viridis
        norm_edges = plt.Normalize(vmin=weights.min(), vmax=weights.max())
        edge_colors = cmap_edges(norm_edges(weights))
        widths = 1 + 3 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
        
        nx.draw_networkx_nodes(self, pos, node_size=800, node_color='skyblue')
        nx.draw_networkx_labels(self, pos, {n: n.name for n in self.nodes}, font_size=10)
        nx.draw_networkx_edges(self, pos, width=widths, edge_color=edge_colors, 
                              arrowstyle='-|>', arrowsize=16)
        edge_labels = {(u, v): f"{d.get('weight', 0):.2f}" for u, v, d in self.edges(data=True)}
        nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels, font_color="firebrick")
        
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm_edges), label="Edge Weight")
        plt.title("Distributed KAMP Network")
        plt.axis("off")
        if show:
            plt.show()