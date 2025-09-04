import numpy as np
import networkx as nx
from itertools import count
from typing import List, Union
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from kamp import KAMP  # فرض می‌کنیم KAMP در kamp.py پیاده‌سازی شده است

class Node:
    """Node class for graph-based distributed computation."""
    node_index = count(0)  # Counter for unique node IDs
    
    def __init__(self, estimator=None, name: str = None, **kwargs):
        """Initialize a Node with an estimator and optional attributes."""
        self.estimator = estimator
        self._id = next(self.node_index)  # Assign unique ID
        self.name = f"node_{self._id}" if name is None else name
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __hash__(self):
        """Hash based on unique ID."""
        return self._id

    def __repr__(self):
        """Representation as node name."""
        return self.name

    def __getattr__(self, attr):
        """Delegate attribute access to estimator if not found on Node."""
        if hasattr(self.estimator, attr):
            return getattr(self.estimator, attr)
        raise AttributeError(f"Node and its estimator have no attribute '{attr}'")

    @property
    def index(self):
        """Get node index."""
        return self._id

    @property
    def id(self):
        """Get node ID (alias for index)."""
        return self._id

class MyGraph(nx.DiGraph):
    """Directed acyclic graph for distributed computation with visualization support."""

    def __init__(self, *args, **kwargs):
        """Initialize graph with triggered nodes tracking."""
        super().__init__(*args, **kwargs)
        self.triggered_nodes = set()  # Set to track triggered nodes

    def compute_sparsity(self, node: Node) -> int:
        """Compute sparsity: number of near-zero elements in vector x."""
        if node.x is None:
            return 0
        return np.sum(np.isclose(node.x, 0, atol=1e-5))  # Tolerance for numerical zero

    def update_node_attributes(self, triggered_nodes: set):
        """Update node attributes for animation (color based on trigger, size based on sparsity)."""
        for node in self.nodes:
            sparsity = self.compute_sparsity(node)
            self.nodes[node]['sparsity'] = sparsity
            self.nodes[node]['color'] = '#00FF00' if node in triggered_nodes else '#808080'  # Green for triggered, gray for dim
            self.nodes[node]['size'] = 300 + 50 * sparsity  # Size scales with sparsity

    def update_edge_attributes(self):
        """Update edge color and thickness based on weight and source sparsity."""
        for u, v, data in self.edges(data=True):
            weight = data.get('weight', 0.5)
            sparsity_u = self.nodes[u].get('sparsity', 0)
            data['thickness'] = 1 + 3 * weight  # Thickness based on weight
            norm_sparsity = sparsity_u / max(1, len(u.x)) if u.x is not None else 0
            rgb_color = plt.cm.RdBu(norm_sparsity)  # Get RGB tuple
            data['color'] = mcolors.to_hex(rgb_color)  # Convert to hex for Manim
            data['weight_label'] = f"{weight:.2f}"  # Store weight as string for label

    def __rshift__(self, node):
        """Normalize incoming weights (>> operator)."""
        if node not in self.nodes:
            raise nx.NetworkXError(f"Node {node} is not in the graph.")
        total_weight = self.in_degree(node, weight='weight')
        if total_weight > 0:
            for j in self.predecessors(node):
                self[j][node]['weight'] /= total_weight
        return self.predecessors(node)

    def __lshift__(self, node):
        """Normalize outgoing weights (<< operator)."""
        if node not in self.nodes:
            raise nx.NetworkXError(f"Node {node} is not in the graph.")
        total_weight = self.out_degree(node, weight='weight')
        if total_weight > 0:
            for i in self.successors(node):
                self[node][i]['weight'] /= total_weight
        return self.successors(node)

    def add_edges_from_list(self, edges):
        """Add edges from a list of (node1, node2, weight) tuples."""
        for node1, node2, weight in edges:
            self.add_edge(node1, node2, x=node1.x, weight=weight)

    def total_in_degree(self, node: Node, inplace: bool = True, **attr) -> 'MyGraph':
        """Aggregate estimates and covariances from predecessors."""
        total_x = 0
        total_P = 0
        if self.in_degree(node) == 0:
            self.nodes[node]['x'] = self.nodes[node].get('x', node.x)
            self.nodes[node]['P'] = self.nodes[node].get('P', node.P)
            return self

        self >> node  # Normalize incoming weights
        for u, v, data in self.in_edges(node, data=True):
            w = data.get('weight', 0)
            total_x += w * data.get('x', 0)
            total_P += w * data.get('P', u.P)

        if inplace:
            self.nodes[node]['x'] = total_x
            self.nodes[node]['P'] = total_P
        return self

    def trigger(self, node: Node, inplace: bool = True, **attr) -> 'MyGraph':
        """Trigger local computation at the node."""
        self.total_in_degree(node, inplace, **attr)
        node.solve()
        self.triggered_nodes.add(node)  # Mark as triggered
        self.update_node_attributes(self.triggered_nodes)
        self.update_edge_attributes()
        return self

    def plot(self, show: bool = True):
        """Static plot of the graph for debugging."""
        pos = nx.spring_layout(self, seed=42)
        weights = np.array([d.get('weight', 0) for _, _, d in self.edges(data=True)])
        cmap_edges = plt.cm.viridis
        norm_edges = plt.Normalize(vmin=weights.min() if weights.size > 0 else 0, 
                                 vmax=weights.max() if weights.size > 0 else 1)
        edge_colors = cmap_edges(norm_edges(weights)) if weights.size > 0 else ['black']
        widths = (1 + 3 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-10) 
                 if weights.size > 0 else [1])

        nx.draw_networkx_nodes(self, pos, node_size=800, node_color='skyblue')
        nx.draw_networkx_labels(self, pos, {n: n.name for n in self.nodes}, font_size=10)
        nx.draw_networkx_edges(self, pos, width=widths, edge_color=edge_colors, 
                              arrowstyle='-|>', arrowsize=16)
        edge_labels = {(u, v): f"{d.get('weight', 0):.2f}" for u, v, d in self.edges(data=True)}
        nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels, font_color="firebrick")
        
        if weights.size > 0:
            plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm_edges), label="Edge Weight")
        plt.title("Distributed KAMP Network")
        plt.axis("off")
        if show:
            plt.show()

class DistributedKAMP(KAMP):
    """Distributed Kalman Approximate Message Passing (DKAMP) for sparse signal recovery."""
    def __init__(self, alpha: float, tau: float, node_max_iter: Union[int, List[int]], 
                 num_triggers: int, graph: nx.DiGraph, A_list: List[np.ndarray], 
                 y_list: List[np.ndarray], random_state: int = None):
        """Initialize DistributedKAMP."""
        super().__init__(alpha=alpha, tau=tau, max_iter=1)
        self.num_triggers = num_triggers
        self.graph = graph
        self.A_list = A_list
        self.y_list = y_list
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        self.nodes = list(graph.nodes)
        self.num_nodes = len(self.nodes)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Graph must be a directed acyclic graph (DAG).")
        if len(A_list) != self.num_nodes or len(y_list) != self.num_nodes:
            raise ValueError("A_list and y_list must match the number of nodes.")
        self.n = A_list[0].shape[1]
        if not all(A.shape[1] == self.n for A in A_list):
            raise ValueError("All measurement matrices must have same columns.")
        if not all(y.shape[0] == A.shape[0] for A, y in zip(A_list, y_list)):
            raise ValueError("Observation vectors must match matrix rows.")
        
        if isinstance(node_max_iter, int):
            self.node_max_iter = [node_max_iter] * self.num_nodes
        elif isinstance(node_max_iter, list) and len(node_max_iter) == self.num_nodes:
            self.node_max_iter = node_max_iter
        else:
            raise ValueError("node_max_iter must be int or list matching nodes.")
        
        self.node_estimators = [
            Node(estimator=KAMP(alpha=alpha, tau=tau, max_iter=max_iter), name=str(i))
            for i, max_iter in enumerate(self.node_max_iter)
        ]
        self.my_graph = MyGraph()
        self._initialize_graph()

    def _initialize_graph(self):
        """Initialize MyGraph with nodes and edges, assigning random weights."""
        for i, node in enumerate(self.node_estimators):
            node.fit(self.A_list[i], self.y_list[i])
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
        for _ in range(self.num_triggers):
            selected_node = self.rng.choice(self.node_estimators)
            self.my_graph.trigger(selected_node, inplace=True)
        for node in self.node_estimators:
            self.my_graph.total_in_degree(node, inplace=True)

    def solve(self) -> np.ndarray:
        """Return global estimate by averaging node estimates."""
        valid_estimates = [node.x for node in self.node_estimators if node.x is not None]
        if not valid_estimates:
            raise ValueError("No valid estimates found.")
        return np.mean(valid_estimates, axis=0)

    def get_node_estimates(self) -> List[np.ndarray]:
        """Return list of local estimates from all nodes."""
        return [node.x for node in self.node_estimators]

    def get_covariances(self) -> List[np.ndarray]:
        """Return list of covariance matrices from all nodes."""
        return [node.P for node in self.node_estimators if hasattr(node, 'P')]

    def animate_fit(self):
        """Animate the fitting process using Manim."""
        #from animation import DistributedKAMPAnimation
        self.fit()
        scene = DistributedKAMPAnimation(self.my_graph, self.num_triggers)
        scene.render()

    @staticmethod
    def create_dag(num_nodes: int, edge_prob: float = 0.3, random_state: int = None) -> nx.DiGraph:
        """Create a random DAG for distributed KAMP."""
        rng = np.random.RandomState(random_state)
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if rng.random() < edge_prob:
                    G.add_edge(i, j)
        return G
    
    
from manim import *
import networkx as nx
import numpy as np
import matplotlib.colors as mcolors

class DistributedKAMPAnimation(Scene):
    """Manim scene for animating DistributedKAMP graph updates, inspired by Mutual Information style."""

    def __init__(self, graph: 'MyGraph', num_triggers: int, **kwargs):
        """Initialize the scene with graph and number of triggers."""
        super().__init__(**kwargs)
        self.graph = graph
        self.num_triggers = num_triggers
        self.pos = nx.spring_layout(graph, seed=42)  # Fixed positions
        self.graph_mobjects = None
        self.hist_mobjects = None
        self.edge_label_mobjects = VGroup()  # To store edge weight labels

    def construct(self):
        """Main animation: build initial graph/histogram animatedly and animate triggers."""
        self.build_graph_animated()
        self.build_histogram_animated()
        self.wait(1)

        for _ in range(self.num_triggers):
            node = np.random.choice(list(self.graph.nodes))
            self.graph.trigger(node)
            self.animate_trigger(node)
            self.update_graph_animation()
            self.update_histogram_animation()
            self.wait(1)

    def build_graph_animated(self):
        """Build the graph animatedly: fade in nodes, then grow edges."""
        nodes = VGroup()
        for node in self.graph.nodes:
            color = self.graph.nodes[node]['color']
            size = self.graph.nodes[node].get('size', 300) / 1000
            dot = Circle(radius=size, fill_color=color, fill_opacity=0.8, stroke_color=WHITE)
            dot.move_to(self.pos_to_manim(self.pos[node]))
            label = Text(node.name, font_size=24).next_to(dot, DOWN, buff=0.1)
            nodes.add(dot, label)

        self.play(FadeIn(nodes, run_time=1.5))  # Fade in all nodes first

        edges = VGroup()
        edge_labels = VGroup()
        for u, v, data in self.graph.edges(data=True):
            start = self.pos_to_manim(self.pos[u])
            end = self.pos_to_manim(self.pos[v])
            arrow = Arrow(start, end, color=data['color'], stroke_width=data['thickness'] * 0.5, buff=0.2)
            self.play(GrowArrow(arrow, run_time=0.5))  # Grow each edge like a wipe effect
            edges.add(arrow)
            
            # Add weight label
            mid_point = (start + end) / 2
            weight_label = Text(data['weight_label'], font_size=18, color=WHITE).next_to(mid_point, UP * 0.2)
            self.play(FadeIn(weight_label, run_time=0.3))
            edge_labels.add(weight_label)

        group = VGroup(nodes, edges)
        group.to_edge(LEFT)
        self.add(group)
        self.graph_mobjects = group
        self.edge_label_mobjects = edge_labels

    def build_histogram_animated(self):
        """Build histogram animatedly with labels."""
        sparsities = [self.graph.nodes[n].get('sparsity', 0) for n in self.graph.nodes]
        max_sp = max(sparsities) + 1 if sparsities else 1
        axes = Axes(
            x_range=[-0.5, len(sparsities) - 0.5, 1],
            y_range=[0, max_sp, 1],
            x_length=5, y_length=3,
            axis_config={"color": BLUE, "include_tip": False},
            tips=False
        )
        # Add y-axis label
        y_label = Text("Number of Zeros", font_size=24, color=BLUE).next_to(axes, LEFT)
        
        labels = [Text(f"{node.name}", font_size=20) for node in self.graph.nodes]
        x_labels = VGroup(*[labels[i].next_to(axes.c2p(i, 0), DOWN) for i in range(len(sparsities))])
        
        bars = VGroup()
        for i, val in enumerate(sparsities):
            bar = Rectangle(
                height=0,  # Start with height 0 for growth
                width=0.8,
                fill_color=BLUE,
                fill_opacity=0.8,
                stroke_color=WHITE
            )
            bar.move_to(axes.c2p(i, 0), DOWN)
            bars.add(bar)

        group = VGroup(axes, y_label, x_labels, bars).to_edge(RIGHT)
        self.play(FadeIn(group, run_time=1.5))
        
        # Animate bar growth
        animations = []
        for i, val in enumerate(sparsities):
            new_height = val / max_sp * axes.y_length
            animations.append(bars[i].animate.stretch_to_fit_height(new_height).align_to(axes.c2p(i, 0), DOWN))
        self.play(*animations, run_time=1)
        
        self.hist_mobjects = group

    def animate_trigger(self, node):
        """Animate the trigger: path-following dot along incoming edges, activate node."""
        animations = []
        for u, v in self.graph.in_edges(node):  # Use self.graph.in_edges instead of self.in_edges
            edge_index = list(self.graph.edges()).index((u, v))
            edge_path = self.graph_mobjects[1][edge_index]  # Get edge mobject
            dot = Dot(color=YELLOW, radius=0.05)
            self.add(dot)
            animations.append(MoveAlongPath(dot, edge_path, run_time=1))  # Dot moves along edge (wipe-like)
            self.play(*animations)
            self.remove(dot)  # Remove dot after animation
            animations = []  # Reset for next edge
        
        # Activate node (e.g., scale up briefly)
        dot_index = list(self.graph.nodes).index(node) * 2  # Dot is even index in nodes group
        node_dot = self.graph_mobjects[0][dot_index]
        self.play(node_dot.animate.scale(1.2).set_color(YELLOW), run_time=0.5)
        self.play(node_dot.animate.scale(1/1.2).set_color(self.graph.nodes[node]['color']), run_time=0.5)

    def update_graph_animation(self):
        """Animate graph updates with smooth transitions."""
        animations = []
        for i, node in enumerate(self.graph.nodes):
            dot = self.graph_mobjects[0][2 * i]
            new_color = self.graph.nodes[node]['color']
            new_radius = self.graph.nodes[node].get('size', 300) / 1000
            animations.append(dot.animate.set_fill(color=new_color).set_radius(new_radius))

        for i, (u, v, data) in enumerate(self.graph.edges(data=True)):
            edge = self.graph_mobjects[1][i]
            new_color = data['color']
            new_width = data['thickness'] * 0.5
            animations.append(edge.animate.set_color(new_color).set_stroke(width=new_width))
            
            # Update weight label with ReplacementTransform
            mid_point = (self.pos_to_manim(self.pos[u]) + self.pos_to_manim(self.pos[v])) / 2
            new_label = Text(data['weight_label'], font_size=18, color=WHITE).next_to(mid_point, UP * 0.2)
            old_label = self.edge_label_mobjects[i]
            animations.append(ReplacementTransform(old_label, new_label))
            self.edge_label_mobjects[i] = new_label  # Update stored label

        self.play(*animations, run_time=1.5)

    def update_histogram_animation(self):
        """Animate histogram updates with growing/shrinking bars."""
        new_sparsities = [self.graph.nodes[n].get('sparsity', 0) for n in self.graph.nodes]
        max_sp = max(new_sparsities) + 1 if new_sparsities else 1
        animations = []
        for i, new_val in enumerate(new_sparsities):
            bar = self.hist_mobjects[3][i]  # Bars are index 3 in group (axes=0, y_label=1, x_labels=2, bars=3)
            new_height = new_val / max_sp * self.hist_mobjects[0].y_length
            animations.append(bar.animate.stretch_to_fit_height(new_height).align_to(self.hist_mobjects[0].c2p(i, 0), DOWN))
        self.play(*animations, run_time=1)

    def pos_to_manim(self, pos):
        """Convert NetworkX position to Manim coordinates."""
        return np.array([pos[0] * 4, pos[1] * 3, 0])
    
    


if __name__ == "__main__":
    # Create a random DAG graph with 5 nodes
    num_nodes = 5
    graph = DistributedKAMP.create_dag(num_nodes, edge_prob=0.5, random_state=30)

    # Generate sample data
    n = 10  # Signal dimension
    m = 5   # Measurements per node
    true_signal = np.random.randn(n)  # True signal
    A_list = [np.random.randn(m, n) for _ in range(num_nodes)]  # Measurement matrices
    y_list = [A @ true_signal + 0.1 * np.random.randn(m) for A in A_list]  # Observations

    # Initialize DistributedKAMP
    dkamp = DistributedKAMP(
        alpha=0.1,
        tau=0.5,
        node_max_iter=10,
        num_triggers=5,
        graph=graph,
        A_list=A_list,
        y_list=y_list,
        random_state=42
    )

    # Run animation
    dkamp.animate_fit()

    # Get results
    global_estimate = dkamp.solve()
    print("Global estimated signal:", global_estimate)
    node_estimates = dkamp.get_node_estimates()
    print("Node estimates:", node_estimates)