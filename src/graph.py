import networkx as nx
import numpy as np
from itertools import count
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class Node:
    node_index = count(0)
    def __init__(self, estimator=None, name=None, **kwargs):
        self.estimator = estimator
        self._id = next(self.node_index)
        self.name = "node" if name is None else name
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __hash__(self):
        return self._id
    
    def __repr__(self):
        if self.name is None:
            return f"N_{self._id}"
        else:
            return self.name 
    
    def __getattr__(self, attr):
        if hasattr(self.estimator, attr):
            return getattr(self.estimator, attr)
        raise AttributeError(f"Node and its estimator don't have any {attr}.")
    
    @property
    def index(self):
        return self._id
    
    @property 
    def id(self):
        return self._id

class MyGraph(nx.DiGraph):
    def trigger(self, node, inplace=True, **attr):
        self.total_in_degree(node, inplace, **attr)
        node.solve()
        return self

    def total_in_degree(self, node, inplace=True, **attr):
        if node not in self.nodes:
            raise nx.NetworkXError(f"Node {node} is not in the graph.")
        
        print('#'*30, f"total_in_degree({node}), {node}.x={node.x}, P_{node}={np.ravel(node.P)}", "#"*30)
        total_x = 0
        total_P = 0
        if self.in_degree(node) == 0:
            self.nodes[node]['x'] = self.nodes[node].get('x', node.x)
            self.nodes[node]['P'] = self.nodes[node].get('P', node.P)
            return self.nodes[node]['x']
        
        print(' '*10, 'normalize in_weights', ' '*10)
        self >> node  # Normalize in-degree weights
        for u, v, data in self.in_edges(node, data=True):
            print('j-->i', f"{u}-->{v}")
            print(' '*25, self.nodes(data=True), ' '*25)
            w = data.get('weight', 0)
            print(f"w={w}, x_{u}={data.get('x', 0)}")
            print(f"P_{u}\n{data.get('P', u.P)}")
            total_x += w * data.get('x', 0)
            total_P += w * data.get('P', u.P)
        if inplace:
            print(f"{node}.total_x={total_x},\n total_P=\n{total_P}")
            print('#'*25,'\n')
            self.nodes[node]['x'] = total_x
            self.nodes[node]['P'] = total_P
            print(f"{node}.total_x={self.nodes[node]['x']},\n {node}.total_P=\n{self.nodes[node]['P']}")
        return self

    def __lshift__(self, node):
        if node not in self.nodes:
            raise nx.NetworkXError(f"Node {node} is not in the graph.")
        total_weight = self.out_degree(node, weight='weight')
        if total_weight > 0:
            for i in self.successors(node):
                self[node][i]['weight'] /= total_weight
        return self.successors(node)
    
    def __rshift__(self, node):
        if node not in self.nodes:
            raise nx.NetworkXError(f"Node {node} is not in the graph.")
        total_weight = self.in_degree(node, weight='weight')
        if total_weight > 0:
            for j in self.predecessors(node):
                self[j][node]['weight'] /= total_weight
        return self.predecessors(node)

    def add_edges_from_list(self, edges):
        for node1, node2, weight in edges:
            self.add_edge(node1, node2, x=node1.x, weight=weight)

    def plot(self, seed=42, show=True):
        # موقعیت گره‌ها با layout پویا و یکنواخت
        pos = nx.kamada_kawai_layout(self)
    
        # اندازه گره‌ها بر اساس درجه
        node_degrees = np.array([self.degree(n) for n in self.nodes()])
        node_sizes = 300 + 80 * node_degrees
    
        # رنگ گره‌ها: colormap پیوسته یا categorical
        cmap_nodes = plt.cm.Paired
        norm_nodes = mcolors.Normalize(vmin=node_degrees.min(), vmax=node_degrees.max())
        node_colors = [cmap_nodes(norm_nodes(d)) for d in node_degrees]
    
        # رسم گره‌ها
        nx.draw_networkx_nodes(self, pos, node_size=node_sizes, node_color=node_colors, edgecolors='black', linewidths=0.8)
    
        # برچسب گره‌ها
        labels = {node: node.name for node in self.nodes()}
        nx.draw_networkx_labels(self, pos, labels, font_size=11, font_color="black", font_weight="bold")
    
        # رنگ و ضخامت یال‌ها بر اساس وزن
        weights = np.array([d.get('weight', 0.5) for u, v, d in self.edges(data=True)])
        widths = 1.5 + 3 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
        cmap_edges = plt.cm.viridis
        norm_edges = mcolors.Normalize(vmin=weights.min(), vmax=weights.max())
        edge_colors = cmap_edges(norm_edges(weights))
    
        # رسم یال‌ها
        edges = nx.draw_networkx_edges(
            self, pos,
            width=widths,
            edge_color=edge_colors,
            arrowstyle='-|>',
            arrowsize=16,
            connectionstyle='arc3,rad=0.1'
        )
    
        # برچسب وزن یال‌ها
        edge_labels = {(u, v): f"{d.get('weight', 0):.2f}" for u, v, d in self.edges(data=True)}
        nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels, font_color="firebrick", font_size=9)
    
        # colorbar برای وزن یال‌ها
        sm = plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm_edges)
        sm.set_array(weights)
        ax = plt.gca()
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Edge Weight", fontsize=11)
    
        # تنظیمات نهایی
        ax.set_facecolor("#f5f5f5")
        ax.set_title("Dynamic Network Visualization", fontsize=15, fontweight="bold", pad=15)
        ax.axis("off")
    
        if show:
            plt.tight_layout()
            plt.show()
            
            
            
            
            
            

# ---------------------- Grok code for myGraph ---------------
# class Node:
#     """Node class for graph-based distributed computation."""
#     node_index = count(0)
    
#     def __init__(self, estimator=None, name: str = None, **kwargs):
#         self.estimator = estimator
#         self._id = next(self.node_index) # first node_index then node_index+=1
#         self.name = f"node_{self._id}" if name is None else name
#         for key, value in kwargs.items():
#             setattr(self, key, value)

#     def __hash__(self):
#         return self._id

#     def __repr__(self):
#         return self.name # f"{self.__class__.__qualname__}({self._id})"

#     def __getattr__(self, attr): # Delegate to estimator if the attribute is not found on Node
#         if hasattr(self.estimator, attr):
#             return getattr(self.estimator, attr)
#         raise AttributeError(f"Node and its `estimator` have no attribute '{attr}'")
    
    
#     @property
#     def index(self):
#          return self._id
#     @property 
#     def id(self):
#         return self._id

# class MyGraph(nx.DiGraph):
#     """Directed acyclic graph for distributed computation."""
    
#     def __rshift__(self, node):
#         """Normalize incoming weights."""
#         total_weight = self.in_degree(node, weight='weight')
#         for j in self.predecessors(node):
#             self[j][node]['weight']/=total_weight
    
#     def __lshift__(self, node):
#         """Normalize outgoing weights."""
#         total_weight = self.out_degree(node, weight='weight')
#         for i in self.successors(node):
#             self[node][i]['weight']/=total_weight
#         return self.successors(node)
    
#     def add_edges_from_list(self, edges):
#         for node1, node2, weight in edges:
#             self.add_edge(node1, node2, x=node1.x, weight=weight)
    
#     def total_in_degree(self, node: Node, inplace: bool = True, **attr) -> 'MyGraph':
#         """
#         Aggregate estimates and covariances from predecessors.

#         Parameters
#         ----------
#         node : Node
#             Target node.
#         inplace : bool, default=True
#             Update node attributes if True.

#         Returns
#         -------
#         MyGraph
#             Updated graph.
#         """
#         total_x = 0
#         total_P = 0
#         if self.in_degree(node) == 0:
#             self.nodes[node]['x'] = self.nodes[node].get('x', node.x)
#             self.nodes[node]['P'] = self.nodes[node].get('P', node.P)
#             return self # self.nodes[node]['x']
        
#         #--------------- Normalize incoming -------------- 
#         self >> node
#         # total_weight = self.in_degree(node, weight='weight')
#         # if total_weight > 0:
#         #     for j in self.predecessors(node):
#         #         self[node][j]['weight'] /= total_weight
        
#         #---------------- Aggregate messages ---------------
#         for u, v, data in self.in_edges(node, data=True, **attr):
#             w = data.get('weight', 0)
#             total_x += w * data.get('x', 0)
#             total_P += w * data.get('P', u.P)
        
#         if inplace:
#             self.nodes[node]['x'] = total_x
#             self.nodes[node]['P'] = total_P
#         return self

#     def trigger(self, node: Node, inplace: bool = True, **attr) -> 'MyGraph':
#         """
#         Trigger local computation at the node.

#         Parameters
#         ----------
#         node : Node
#             Target node.
#         inplace : bool, default=True
#             Update node attributes if True.

#         Returns
#         -------
#         MyGraph
#             Updated graph.
#         """
#         self.total_in_degree(node, inplace, **attr)
#         node.solve()
#         return self

#     def plot(self, show: bool = True):
#         """Visualize the graph."""
#         pos = nx.spring_layout(self, seed=42)
#         weights = np.array([d.get('weight', 0) for _, _, d in self.edges(data=True)])
#         cmap_edges = plt.cm.viridis
#         norm_edges = plt.Normalize(vmin=weights.min(), vmax=weights.max())
#         edge_colors = cmap_edges(norm_edges(weights))
#         widths = 1 + 3 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
        
#         nx.draw_networkx_nodes(self, pos, node_size=800, node_color='skyblue')
#         nx.draw_networkx_labels(self, pos, {n: n.name for n in self.nodes}, font_size=10)
#         nx.draw_networkx_edges(self, pos, width=widths, edge_color=edge_colors, 
#                               arrowstyle='-|>', arrowsize=16)
#         edge_labels = {(u, v): f"{d.get('weight', 0):.2f}" for u, v, d in self.edges(data=True)}
#         nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels, font_color="firebrick")
        
#         plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm_edges), label="Edge Weight")
#         plt.title("Distributed KAMP Network")
#         plt.axis("off")
#         if show:
#             plt.show()