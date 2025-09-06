import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Union
from ..core.kamp import KAMP
from ..network.graph import MyGraph, Node
from ..network.random_digraphs import create_strongly_connected_graph

class DistributedKAMP(KAMP):
    """
    Distributed Kalman Approximate Message Passing (DKAMP) for sparse signal recovery
    on a directed acyclic graph (DAG) with per-node measurement matrices and a shared unknown vector.
    Inherits from KAMP to reuse single-node functionality.
    """
    def __init__(self, alpha: float, tau: float, node_max_iter: Union[int, List[int]],
                 num_triggers: int, graph: nx.DiGraph, A_list: List[np.ndarray],
                 y_list: List[np.ndarray], random_state: int = None, verbose: bool = False):
        """
        Initialize DistributedKAMP.
        
        Args:
            alpha: Step size for KAMP updates.
            tau: Threshold for soft thresholding.
            node_max_iter: Maximum iterations for each node's KAMP algorithm (int or list of ints).
            num_triggers: Number of random node triggers.
            graph: nx.DiGraph representing the DAG, determines number of nodes.
            A_list: List of measurement matrices [A_1, ..., A_N].
            y_list: List of observation vectors [y_1, ..., y_N].
            random_state: Random seed for reproducibility.
        """
        super().__init__(alpha=alpha, tau=tau, max_iter=1, verbose=verbose)
        self.num_triggers = num_triggers
        self.graph = graph
        self.A_list = A_list
        self.y_list = y_list
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        # Validate inputs
        self.nodes = list(graph.nodes)
        self.num_nodes = len(self.nodes)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Graph must be a directed acyclic graph (DAG).")
        if len(A_list) != self.num_nodes or len(y_list) != self.num_nodes:
            raise ValueError("A_list and y_list must match the number of nodes in the graph.")
        self.n = A_list[0].shape[1]
        if not all(A.shape[1] == self.n for A in A_list):
            raise ValueError("All measurement matrices must have the same number of columns.")
        if not all(y.shape[0] == A.shape[0] for A, y in zip(A_list, y_list)):
            raise ValueError("Observation vectors must match measurement matrix rows.")
        
        # Handle node_max_iter (single int or list)
        if isinstance(node_max_iter, int):
            self.node_max_iter = [node_max_iter] * self.num_nodes
        elif isinstance(node_max_iter, list) and len(node_max_iter) == self.num_nodes:
            self.node_max_iter = node_max_iter
        else:
            raise ValueError("node_max_iter must be an int or a list matching the number of nodes.")
        
        # Initialize node estimators with varying max_iter
        self.node_estimators = [
            Node(estimator=KAMP(alpha=alpha, tau=tau, max_iter=max_iter), name=str(i))
            for i, max_iter in enumerate(self.node_max_iter)
        ]
        self.my_graph = MyGraph(verbose=False)
        self._initialize_graph()

    def _initialize_graph(self):
        """Initialize MyGraph with nodes and edges, assigning random weights."""
        # Fit each node with its data subset
        for i, node in enumerate(self.node_estimators):
            node.fit(self.A_list[i], self.y_list[i])
        # Add all nodes to the graph
        for node in self.node_estimators:
            self.my_graph.add_node(node, x=node.x, P=node.P)
        # Create edges with initialized x
        for u, v in self.graph.edges():
            weight = np.clip(self.rng.normal(loc=0.5, scale=0.1), 0, 1)
            self.my_graph.add_edge(
                self.node_estimators[u],
                self.node_estimators[v],
                x=self.node_estimators[u].x,
                weight=weight
            )

    def fit(self):
        """Run distributed KAMP with random node triggering."""
        # Perform random node triggering
        for _ in range(self.num_triggers):
            selected_node = self.rng.choice(self.node_estimators)
            self.my_graph.trigger(selected_node, inplace=True)

        # Update local estimates based on graph messages
        for node in self.node_estimators:
            self.my_graph.total_in_degree(node, inplace=True)

        return self

    def solve(self) -> np.ndarray:
        """
        Return the global estimate by averaging node estimates.
        
        Returns:
            np.ndarray: Global estimated signal.
        """
        valid_estimates = [node.x for node in self.node_estimators if node.x is not None]
        if not valid_estimates:
            raise ValueError("No valid estimates found.")
        return np.mean(valid_estimates, axis=0)

    def get_node_estimates(self) -> List[np.ndarray]:
        """
        Return the list of local estimates from all nodes.
        
        Returns:
            List[np.ndarray]: List of node estimates.
        """
        return [node.x for node in self.node_estimators]

    def get_covariances(self) -> List[np.ndarray]:
        """
        Return the list of covariance matrices from all nodes.
        
        Returns:
            List[np.ndarray]: List of covariance matrices.
        """
        return [node.P for node in self.node_estimators if hasattr(node, 'P')]

    def plot_graph(self, show: bool = True):
        """Visualize the graph using MyGraph.plot."""
        self.my_graph.plot(show=show)

    @classmethod
    def from_partition(cls, data: Dict[int, Tuple[np.ndarray, np.ndarray]], topology: str,
                      max_iters: int = 50, consensus_tol: float = 1e-3,
                      alpha: float = 0.5, tau: float = 0.1, num_triggers: int = 100,
                      random_state: int = None):
        """
        Create DistributedKAMP from partitioned data.

        Args:
            data: Dict of {node_id: (A_i, y_i)}
            topology: 'dag', 'cycle', 'selfloop', 'mixed'
            max_iters: Maximum iterations per node
            consensus_tol: Consensus tolerance (not used in current implementation)
            alpha: Step size
            tau: Threshold
            num_triggers: Number of random triggers
            random_state: Random seed

        Returns:
            DistributedKAMP instance
        """
        nodes = sorted(data.keys())
        num_nodes = len(nodes)
        A_list = [data[node][0] for node in nodes]
        y_list = [data[node][1] for node in nodes]

        # Create graph based on topology
        if topology == 'dag':
            graph = cls.create_dag(num_nodes, random_state=random_state)
        elif topology == 'cycle':
            graph = nx.DiGraph()
            graph.add_nodes_from(range(num_nodes))
            for i in range(num_nodes):
                graph.add_edge(i, (i + 1) % num_nodes)
        elif topology == 'selfloop':
            graph = nx.DiGraph()
            graph.add_nodes_from(range(num_nodes))
            for i in range(num_nodes):
                graph.add_edge(i, i)
        elif topology == 'mixed':
            graph = cls.create_dag(num_nodes, random_state=random_state)
            # Add some cycles or self-loops if needed
        else:
            raise ValueError(f"Unknown topology: {topology}")

        return cls(alpha=alpha, tau=tau, node_max_iter=max_iters,
                  num_triggers=num_triggers, graph=graph,
                  A_list=A_list, y_list=y_list, random_state=random_state)

    def report(self) -> Dict:
        """
        Generate a report with metrics.

        Returns:
            Dict with consensus_error, nmse_global, bytes, iters
        """
        x_global = self.solve()
        # Simple consensus error (variance of node estimates)
        node_estimates = self.get_node_estimates()
        consensus_error = np.var([np.linalg.norm(x - x_global) for x in node_estimates])

        # NMSE global (assuming true signal is x_global for simplicity)
        nmse_global = 0.0  # Placeholder

        # Bytes: rough estimate
        bytes_used = sum(A.nbytes + y.nbytes for A, y in zip(self.A_list, self.y_list))

        return {
            'consensus_error': consensus_error,
            'nmse_global': nmse_global,
            'bytes': bytes_used,
            'iters': self.num_triggers
        }

    @staticmethod
    def create_dag(num_nodes: int, edge_prob: float = 0.3, random_state: int = None) -> nx.DiGraph:
        """
        Create a random DAG for distributed KAMP.

        Args:
            num_nodes: Number of nodes in the graph.
            edge_prob: Probability of edge creation.
            random_state: Random seed for reproducibility.

        Returns:
            nx.DiGraph: Directed acyclic graph.
        """
        rng = np.random.RandomState(random_state)
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if rng.random() < edge_prob:
                    G.add_edge(i, j)
        return G