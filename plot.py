import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import make_moons
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import networkx as nx
from ampire.distributed import DistEnetConvexHull, DistThresholdFinder
from ampire.utils import DataReader
import os
import logging
from pathlib import Path

# Define base output directory
BASE_OUTPUT_DIR = Path('ThesisExperiments/2DPlots/kamp_distenetconvexhull')

# Set up logging
def setup_logging():
    """Ensure the logging directory exists and configure logging."""
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        filename=BASE_OUTPUT_DIR / 'grid_search.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Call setup_logging before any logging operations
setup_logging()

def load_data(data_info, two_moon):
    """
    Load and scale dataset.
    
    Parameters
    ----------
    data_info : dict
        Dictionary containing 'path' and 'gamma' for the dataset.
    two_moon : ndarray
        Fallback two_moon dataset.
    
    Returns
    -------
    ndarray
        Scaled dataset.
    """
    try:
        path = data_info['path']
        if path is not None:
            X = DataReader(file=path)()
            logging.info(f"Loaded dataset from {path}, shape: {X.shape}")
        else:
            X = two_moon
            logging.info(f"Using generated two_moon dataset, shape: {X.shape}")
        X = preprocessing.MinMaxScaler(feature_range=(-5, 5)).fit_transform(X)
        return X
    except Exception as e:
        logging.error(f"Failed to load dataset {data_info.get('path', 'two_moon')}: {e}")
        raise

def create_complete_dag(num_nodes, random_state=None):
    """
    Create a complete DAG for a given number of nodes with no isolated nodes.
    
    Parameters
    ----------
    num_nodes : int
        Number of nodes in the DAG (minimum 2).
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    nx.DiGraph
        Complete directed acyclic graph with no isolated nodes.
    """
    if num_nodes < 2:
        raise ValueError("Number of nodes must be at least 2")
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    rng = np.random.RandomState(random_state)
    
    # Create a fully connected DAG (no isolated nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            G.add_edge(i, j, weight=rng.uniform(0.1, 0.9))
    
    # Verify no isolated nodes
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        logging.error(f"Graph contains isolated nodes: {isolated_nodes}")
        raise ValueError(f"Graph contains isolated nodes: {isolated_nodes}")
    
    logging.info(f"Created graph with {num_nodes} nodes, {len(G.edges)} edges")
    return G

def plot_decision_boundary(algorithm, X, title, output_path, X_scatter=None, n=100):
    """
    Plot and save the decision boundary for a given algorithm.
    
    Parameters
    ----------
    algorithm : estimator object
        The fitted estimator with a predict method.
    X : ndarray
        Input data for fitting and plotting.
    title : str
        Title of the plot.
    output_path : Path
        Path to save the plot.
    X_scatter : ndarray, optional
        Data points to scatter. If None, uses X.
    n : int
        Number of points in the mesh grid.
    """
    try:
        algorithm.fit(X)
        axis = 5  # Data scaled to [-5, 5]
        lin = np.linspace(-axis, axis, n)
        x_mesh, y_mesh = np.meshgrid(lin, lin)
        X_grid = np.c_[x_mesh.ravel(), y_mesh.ravel()]
        
        # Predict on the grid
        Z = algorithm.predict(X_grid)
        Z = Z.reshape(x_mesh.shape)
        
        # Create contour plot
        plt.figure(figsize=(7, 7))
        plt.contour(x_mesh, y_mesh, Z, levels=[0], colors='black')
        
        # Scatter the data points
        if X_scatter is None:
            X_scatter = X
        plt.scatter(X_scatter[:, 0], X_scatter[:, 1], s=20, edgecolor='k', color='red')
        
        plt.xlabel('x axis', fontsize=15)
        plt.ylabel('y axis', fontsize=15)
        plt.title(title)
        plt.axis('square')
        
        # Save the plot
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved plot: {output_path}")
    except Exception as e:
        logging.error(f"Failed to plot decision boundary for {title}: {e}")
        raise

def perform_grid_search(X, base_params, graph, num_nodes, data_name, graph_desc, output_dir):
    """
    Perform grid search over hyperparameters and return the best model and score.
    
    Parameters
    ----------
    X : ndarray
        Input data.
    base_params : dict
        Base parameters for DistEnetConvexHull.
    graph : nx.DiGraph
        Graph for DistributedKAMP.
    num_nodes : int
        Number of nodes in the graph.
    data_name : str
        Name of the dataset.
    graph_desc : str
        Description of the graph configuration.
    output_dir : Path
        Directory to save plots.
    
    Returns
    -------
    tuple
        Best model, best score, best parameters.
    """
    param_grid = {
        'landa1': [0.1, 0.5, 0.9],
        'alpha': [0.3, 0.5, 0.7],
        'tau': [0.05, 0.1, 0.2],
        'gamma': [base_params['gamma'], base_params['gamma'] * 0.5, base_params['gamma'] * 2.0]
    }
    
    best_score = float('inf')
    best_model = None
    best_params = None
    total_combinations = len(list(ParameterGrid(param_grid)))
    
    for idx, params in enumerate(ParameterGrid(param_grid)):
        try:
            logging.info(f"Processing sample {idx}/{total_combinations-1} for {data_name}/{graph_desc}")
            # Update parameters
            algo_params = base_params.copy()
            algo_params.update(params)
            algo_params['graph'] = graph
            algo_params['just_dag'] = True  # Always True since minimum nodes is 2
            algo_params['num_triggers'] = 100  # Always 100 since minimum nodes is 2
            
            # Validate graph
            if len(graph.nodes) != num_nodes:
                logging.error(f"Graph node count ({len(graph.nodes)}) does not match num_nodes ({num_nodes})")
                raise ValueError(f"Graph node count ({len(graph.nodes)}) does not match num_nodes ({num_nodes})")
            
            # Initialize and fit model
            algo = DistEnetConvexHull(**algo_params)
            algo.fit(X)
            
            # Log internal data structures
            if hasattr(algo, 'A_list') and hasattr(algo, 'y_list'):
                logging.info(f"A_list length: {len(algo.A_list)}, y_list length: {len(algo.y_list)}, graph nodes: {len(graph.nodes)}, num_triggers: {algo_params['num_triggers']}")
            
            # Find threshold
            threshold_finder = DistThresholdFinder(algo, X)
            _, _, thrs = threshold_finder.find(outs='max')
            algo.thr = thrs[0] if len(thrs) > 0 else 1e-5
            
            # Compute score (negative mean decision function score on inliers)
            scores = algo.decision_function(X)
            score = -np.mean(scores)
            
            # Plot decision boundary
            param_str = f"landa1_{params['landa1']}_alpha_{params['alpha']}_tau_{params['tau']}_gamma_{params['gamma']:.3f}"
            algo_name = f"DistEnetConvexHull_{graph_desc}_{param_str}"
            title = f"{algo_name} | {data_name}"
            output_path = output_dir / f"{algo_name}-{data_name}.png"
            plot_decision_boundary(algo, X, title, output_path)
            
            # Update best model if score is better
            if score < best_score:
                best_score = score
                best_model = algo
                best_params = params
                # Save best plot
                best_output_path = output_dir / f"best_DistEnetConvexHull_{graph_desc}-{data_name}.png"
                plot_decision_boundary(algo, X, f"Best {title}", best_output_path)
                
            logging.info(f"Evaluated params {params} for {data_name}/{graph_desc}: score={score:.4f}")
        
        except Exception as e:
            logging.error(f"Error with params {params} for {data_name}/{graph_desc}: {e}")
            continue
    
    if best_model is None:
        logging.error(f"No valid model found for {data_name}/{graph_desc}")
        return None, float('inf'), None
    
    return best_model, best_score, best_params

def main():
    """Main function to run grid search and generate plots."""
    # Configuration
    random_state = 42
    two_moon = make_moons(n_samples=150, noise=0.0, random_state=random_state)[0] - np.array([0.5, 0.25])
    
    # Base parameters for DistEnetConvexHull
    base_params = {
        'landa1': 0.5,
        'alpha': 0.5,
        'tau': 0.1,
        'node_max_iter': 50,
        'num_triggers': 100,
        'metric': 'rbf',
        'gamma': 'scale',  # Will be overridden by dataset-specific gamma
        'random_state': random_state,
        'verbose': False
    }
    
    # Define datasets with hardcoded paths
    datasets = {
        'pentagon': {
            'gamma': 0.5,
            'path': 'data/synthetic/pentagon.csv'
        },
        'moon': {
            'gamma': 0.1,
            'path': 'data/synthetic/moon.csv'
        },
        'moon_circle': {
            'gamma': 0.1,
            'path': 'data/synthetic/moon-circle.csv'
        },
        'disconnected': {
            'gamma': 0.75,
            'path': 'data/synthetic/disconnected.csv'
        },
        'non_convex_triangle': {
            'gamma': 0.1,
            'path': 'data/synthetic/non-convex-triangle.csv'
        },
        'tri': {
            'gamma': 0.5,
            'path': 'data/synthetic/tri.csv'
        },
        'two-moon': {
            'gamma': 1.2,
            'path': None
        }
    }
    
    # Define graph configurations
    graph_configs = [
        {'num_nodes': 2, 'description': 'dag_2_nodes'},
        {'num_nodes': 3, 'description': 'dag_3_nodes'},
        {'num_nodes': 4, 'description': 'dag_4_nodes'},
        {'num_nodes': 5, 'description': 'dag_5_nodes'}
    ]
    
    # Iterate over datasets
    for data_name, data_info in tqdm(datasets.items(), desc="Processing datasets"):
        # Load and scale data
        try:
            X = load_data(data_info, two_moon)
        except Exception as e:
            logging.error(f"Skipping dataset {data_name} due to loading error")
            continue
        
        # Create output directory for dataset
        dataset_dir = BASE_OUTPUT_DIR / data_name
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Iterate over graph configurations
        for graph_config in tqdm(graph_configs, desc=f"Graphs for {data_name}"):
            num_nodes = graph_config['num_nodes']
            graph_desc = graph_config['description']
            
            # Create graph
            try:
                graph = create_complete_dag(num_nodes, random_state)
                logging.info(f"Created graph for {data_name}/{graph_desc}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
            except Exception as e:
                logging.error(f"Failed to create graph for {data_name}/{graph_desc}: {e}")
                continue
            
            # Create output directory for graph configuration
            output_dir = dataset_dir / graph_desc
            os.makedirs(output_dir, exist_ok=True)
            
            # Perform grid search
            base_params['gamma'] = data_info['gamma']
            best_model, best_score, best_params = perform_grid_search(
                X, base_params, graph, num_nodes, data_name, graph_desc, output_dir
            )
            
            if best_model is None:
                logging.warning(f"No valid model found for {data_name}/{graph_desc}")
            else:
                logging.info(f"Best params for {data_name}/{graph_desc}: {best_params}, score={best_score:.4f}")

if __name__ == "__main__":
    main()