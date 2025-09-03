# Distributed-Kalman-Approximate-Message-Passing-Algorithm
DKAMP: A distributed Kalman-based Approximate Message Passing (AMP) implementation for sparse signal recovery over directed acyclic graphs (DAGs), enabling efficient decentralized estimation with shared unknown vectors across nodes.


















# File Strucutre
dkamp/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── enet_convex_hull.py  # Elastic Net-based convex hull classifier
│   │   ├── kamp.py              # Kalman-based AMP implementation
│   │   └── threshold_finder.py   # Threshold finding for anomaly detection
│   ├── network/
│   │   ├── __init__.py
│   │   ├── graph.py             # Graph utilities (MyGraph, Node classes)
│   │   └── random_digraphs.py   # Random DAG generation
│   ├── distributed/
│   │   ├── __init__.py
│   │   └── dkamp.py            # Distributed KAMP implementation
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py          # Evaluation metrics (e.g., AUC-ROC, PR)
│       └── visualization.py    # Plotting and visualization utilities
├── tests/
│   ├── __init__.py
│   ├── test_enet_convex_hull.py
│   ├── test_kamp.py
│   ├── test_dkamp.py
│   └── test_graph.py
├── examples/
│   ├── example_synthetic.py    # Synthetic data example for DKAMP
│   └── example_anomaly_detection.py  # Anomaly detection use case
├── docs/
│   ├── api.md                 # API documentation
│   └── tutorial.md           # Usage tutorial
├── requirements.txt           # Dependencies
├── README.md                 # Repository description
├── LICENSE                   # License (e.g., MIT)
├── setup.py                  # Package installation script
└── main.py                   # Entry point for running DKAMP
