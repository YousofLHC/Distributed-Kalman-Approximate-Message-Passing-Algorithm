# Ampire
Ampire: Distributed Kalman Approximate Message Passing Algorithm

A Python package implementing distributed Kalman-based Approximate Message Passing (DKAMP) for sparse signal recovery over directed acyclic graphs (DAGs), enabling efficient decentralized estimation with shared unknown vectors across nodes.

## Installation

Install from PyPI:
```bash
pip install ampire
```

Or install from source:
```bash
git clone https://github.com/YousfLHC/ampire.git
cd ampire
pip install -e .
```

## Usage

```python
from ampire import KAMP, DistributedKAMP

# Example usage
kamp = KAMP(alpha=0.5, tau=0.1, max_iter=100)
# ... fit and solve
```


















## File Structure
ampire/
├── src/
│   └── ampire/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── kamp.py              # Kalman-based AMP implementation
│       │   ├── enet_convex_hull.py  # Elastic Net-based convex hull classifier
│       │   └── threshold_finder.py  # Threshold finding for anomaly detection
│       ├── network/
│       │   ├── __init__.py
│       │   ├── graph.py             # Graph utilities (MyGraph, Node classes)
│       │   └── random_digraphs.py   # Random DAG generation
│       ├── distributed/
│       │   ├── __init__.py
│       │   └── distributed_kamp.py  # Distributed KAMP implementation
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── metrics.py           # Evaluation metrics (e.g., AUC-ROC, PR)
│       │   └── visualization.py     # Plotting and visualization utilities
│       ├── examples/
│       │   ├── __init__.py
│       │   └── example_*.py         # Example scripts
│       ├── tests/
│       │   ├── __init__.py
│       │   └── test_*.py            # Test files
│       └── docs/
│           ├── api.md               # API documentation
│           └── tutorial.md          # Usage tutorial
├── setup.py                    # Package installation script
├── MANIFEST.in                 # Files to include in distribution
├── README.md                   # Repository description
├── LICENSE                     # License (e.g., MIT)
└── requirements.txt            # Dependencies
