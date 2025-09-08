"""
Ampire: Distributed Kalman Approximate Message Passing Algorithm

This package provides implementations for Kalman-based Approximate Message Passing (KAMP)
and its distributed version for sparse signal recovery.
"""

from .core import KAMP, EnetConvexHull, ThresholdFinder
from .distributed import DistributedKAMP
from .network import MyGraph, Node, create_strongly_connected_graph
from .utils import calculate_metrics, log_results, plot_signal_comparison

__version__ = "0.9.0"
__all__ = [
    "KAMP", "DistributedKAMP", "EnetConvexHull", "ThresholdFinder",
    "MyGraph", "Node", "create_strongly_connected_graph",
    "calculate_metrics", "log_results", "plot_signal_comparison"
]