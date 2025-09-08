import numpy as np
from scipy.linalg import solve_triangular
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics.pairwise import pairwise_kernels
from ampire.distributed import DistributedKAMP
from ampire.network.graph import MyGraph
import networkx as nx
from tqdm import tqdm
import logging
import os
import matplotlib.pyplot as plt

def ensure_fitted(func):
    """
    Decorator to ensure the model is fitted before method execution.
    """
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "is_fitted_"):
            raise ValueError(f"This {self.__class__.__name__} instance is not fitted yet. Call `fit` before using this method.")
        return func(self, *args, **kwargs)
    return wrapper

class DistEnetConvexHull(BaseEstimator, OutlierMixin):
    """
    One-Class Classifier for Anomaly Detection using Elastic Net and Convex Hull with DistributedKAMP solver.

    Parameters
    ----------
    landa1 : float (default=0.5)
        The weight of L1 regularization. Must be in the range [0,1].
    target : int (default=1)
        The label of the target class (e.g., 1 for inliers).
    lb : ndarray of shape (n_samples, ) or None
        Lower bound for optimization variables. Defaults to zeros.
    alpha : float (default=0.5)
        Damping factor for DistributedKAMP, in [0,1].
    tau : float (default=0.1)
        Thresholding parameter for DistributedKAMP.
    node_max_iter : int or list (default=50)
        Maximum iterations for each node's KAMP algorithm.
    num_triggers : int (default=100)
        Number of random node triggers for DistributedKAMP.
    graph : MyGraph or None (default=None)
        NetworkX DiGraph for DistributedKAMP. If None, a default DAG is created.
    metric : str, callable (default='rbf')
        The kernel metric used for pairwise similarity computation.
    only_target : bool (default=True)
        Whether to use only the target class samples for training.
    thr : float (default=1.0)
        The decision threshold for classifying anomalies.
    kernel_params : dict (default=None)
        Additional parameters for kernel computation.
    degree : int (default=3)
        Degree for poly kernels.
    gamma : {'scale', 'auto'}, float (default='scale')
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    coef0 : float (default=1.0)
        Independent term for 'poly' and 'sigmoid' kernels.
    random_state : int or None (default=None)
        Random seed for reproducibility.
    just_dag : bool (default=False)
        If True, enforce that graph must be a DAG.
    verbose : bool (default=False)
        Whether to enable verbose logging.
    """
    def __init__(self, landa1=0.5, target=1, lb=None, alpha=0.5, tau=0.1,
                 node_max_iter=50, num_triggers=100, graph=None, metric='rbf',
                 only_target=True, thr=1.0, kernel_params=None, degree=3,
                 gamma='scale', coef0=1.0, random_state=None, just_dag=False,
                 verbose=False):
        self.landa1 = landa1
        self.landa2 = 1 - self.landa1
        self.target = target
        self.lb = lb
        self.alpha = alpha
        self.tau = tau
        self.node_max_iter = node_max_iter
        self.num_triggers = num_triggers
        self.graph = graph
        self.metric = metric
        self.only_target = only_target
        self.thr = thr
        self.kernel_params = kernel_params if kernel_params else {}
        self.random_state = random_state
        self.just_dag = just_dag
        self.verbose = verbose

        # Initialize kernel-specific parameters
        if self.metric in {'poly', 'rbf', 'sigmoid'} and 'gamma' not in self.kernel_params:
            self.kernel_params['gamma'] = gamma
        if self.metric == 'poly':
            if 'degree' not in self.kernel_params:
                self.kernel_params['degree'] = degree
            if 'coef0' not in self.kernel_params:
                self.kernel_params['coef0'] = coef0
        if 'coef0' not in self.kernel_params and self.metric == 'sigmoid':
            self.kernel_params['coef0'] = coef0

        # Setup logging
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, 'dist_enet_convex_hull.log'),
            level=logging.DEBUG if verbose else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _validate_params(self):
        """Validate input parameters."""
        if not isinstance(self.landa1, (int, float)) or not (0 <= self.landa1 <= 1):
            raise ValueError(f"landa1 ({self.landa1}) must be in [0,1].")
        if not isinstance(self.target, int):
            raise ValueError(f"target ({self.target}) must be an integer.")
        if self.lb is not None and not isinstance(self.lb, np.ndarray):
            raise ValueError(f"lb must be a numpy array or None. Got {type(self.lb)}.")
        if not (0 <= self.alpha <= 1):
            raise ValueError(f"alpha ({self.alpha}) must be in [0,1].")
        if not isinstance(self.node_max_iter, (int, list)):
            raise ValueError(f"node_max_iter must be an int or list. Got {type(self.node_max_iter)}.")
        if not isinstance(self.num_triggers, int) or self.num_triggers <= 0:
            raise ValueError(f"num_triggers ({self.num_triggers}) must be a positive integer.")
        if self.graph is not None and not isinstance(self.graph, (nx.DiGraph, MyGraph)):
            raise ValueError(f"graph must be a NetworkX DiGraph or MyGraph. Got {type(self.graph)}.")
        if not isinstance(self.metric, (str, callable)):
            raise ValueError(f"metric ({self.metric}) must be a string or callable.")
        if not isinstance(self.only_target, bool):
            raise ValueError(f"only_target ({self.only_target}) must be a boolean.")
        if not isinstance(self.thr, (int, float)) or self.thr <= 0:
            raise ValueError(f"thr ({self.thr}) must be a positive float.")
        if self.random_state is not None and not isinstance(self.random_state, int):
            raise ValueError(f"random_state must be an integer or None. Got {type(self.random_state)}.")
        if not isinstance(self.just_dag, bool):
            raise ValueError(f"just_dag ({self.just_dag}) must be a boolean.")

    def _validate_kernel_params(self):
        """Validate kernel-specific parameters."""
        if self.metric is None:
            raise ValueError("The kernel metric must be specified.")
        if self.metric in {'poly', 'rbf', 'sigmoid'}:
            gamma = self.kernel_params.get('gamma', 'scale')
            if not (isinstance(gamma, (int, float)) or gamma in {'scale', 'auto'}):
                raise ValueError(f"Invalid gamma value: {gamma}. Must be 'scale', 'auto', or a number.")
        if self.metric == 'poly':
            degree = self.kernel_params.get('degree', None)
            if degree is None or not isinstance(degree, int) or degree <= 0:
                raise ValueError(f"Invalid degree value: {degree}. Must be a positive integer.")
        if self.metric in {'poly', 'sigmoid'}:
            coef0 = self.kernel_params.get('coef0', None)
            if coef0 is not None and not isinstance(coef0, (int, float)):
                raise ValueError(f"Invalid coef0 value: {coef0}. Must be a number.")

    def _adjust_kernel(self, X, Y=None):
        """Compute adjusted pairwise kernel similarity matrix."""
        if Y is None:
            Y = X
        G = pairwise_kernels(X, Y, metric=self.metric, **self.kernel_params)
        G_sum = G.sum()
        row_sum = G.sum(axis=1, keepdims=True)
        col_sum = G.sum(axis=0, keepdims=True)
        return G - (row_sum + col_sum) / X.shape[0] + G_sum / (X.shape[0] ** 2)

    def _calculate_P(self, X):
        """Compute optimization matrix P."""
        BTB = self._adjust_kernel(X)
        P = (self.landa2 * np.identity(X.shape[0])) + BTB
        return P

    def _create_default_graph(self, num_nodes):
        """Create a default DAG for DistributedKAMP."""
        rng = np.random.RandomState(self.random_state)
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if rng.random() < 0.9:  # Edge probability
                    G.add_edge(i, j)
        return G

    def _convert_qp_to_kamp(self, P, q):
        """
        Convert QP problem (min x^T P x + q^T x) to KAMP format (y = A @ x).
        Returns A_list, y_list for DistributedKAMP.
        """
        try:
            # Perform Cholesky decomposition: P = L @ L.T
            L = np.linalg.cholesky(P)
            # Compute y = -0.5 * L^(-1) @ q
            y = -0.5 * solve_triangular(L, q, lower=True)
            A = L.T  # Measurement matrix
        except np.linalg.LinAlgError:
            logging.error("Cholesky decomposition failed. Using fallback method.")
            # Fallback: Use square root of eigenvalues
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.clip(eigvals, 0, None)  # Ensure non-negative
            L = eigvecs @ np.diag(np.sqrt(eigvals))
            y = -0.5 * solve_triangular(L, q, lower=False, check_finite=False)
            A = L.T

        # Normalize A and y to prevent numerical issues
        A_norm = np.linalg.norm(A, ord='fro')
        y_norm = np.linalg.norm(y)
        scale_A = max(A_norm, 1e-10)
        scale_y = max(y_norm, 1e-10)
        A = A / scale_A
        y = y / scale_y

        # Distribute data across nodes
        num_nodes = min(5, A.shape[0])  # Use up to 5 nodes or fewer if A is small
        A_list = np.array_split(A, num_nodes, axis=0)
        y_list = np.array_split(y, num_nodes, axis=0)

        return A_list, y_list, scale_A * scale_y

    def fit(self, X, y=None):
        """
        Fit the DistEnetConvexHull model to the given data.
        """
        # Validate parameters and inputs
        self._validate_params()
        self._validate_kernel_params()
        if y is not None:
            X, y = check_X_y(X, y, accept_sparse=False, ensure_2d=True, dtype=np.float64)
            self.return_label = True
            self.classes_ = np.unique(y)  # Required for scikit-learn compatibility
            mask = (y == self.target)
            self.X_target = X[mask, :] if self.only_target else X
        else:
            X = check_array(X, ensure_2d=True, dtype=np.float64)
            self.X_target = X

        # Compute the kernel matrix
        self.G = pairwise_kernels(self.X_target, metric=self.metric, **self.kernel_params)
        self.n, self.m = self.X_target.shape

        # Compute the optimization matrix
        self.P = self._calculate_P(self.X_target)
        # Initialize lower bounds
        self.lb = np.zeros((self.n, 1)) if self.lb is None else self.lb
        self.is_fitted_ = True

        # Initialize graph if not provided
        if self.graph is None:
            self.graph = self._create_default_graph(num_nodes=5)

        return self

    @ensure_fitted
    def __calculate_z__(self, sample):
        """
        Calculate the z-value for a given sample using DistributedKAMP.
        """
        # Compute kernel similarity between X_target and the sample
        Ky = self._adjust_kernel(self.X_target, sample)
        h = Ky - (Ky.sum() + self.G.sum(axis=1, keepdims=True)) / self.X_target.shape[0]
        h += self.G.sum() / (self.X_target.shape[0] ** 2)
        q = self.landa1 * np.ones((self.n, 1)) - 2 * h

        # Convert QP to KAMP format
        A_list, y_list, scale = self._convert_qp_to_kamp(self.P, q)

        # Initialize DistributedKAMP
        dk = DistributedKAMP(
            alpha=self.alpha,
            tau=self.tau,
            node_max_iter=self.node_max_iter,
            num_triggers=self.num_triggers,
            graph=self.graph,
            A_list=A_list,
            y_list=y_list,
            random_state=self.random_state,
            just_dag=self.just_dag,
            verbose=self.verbose
        )

        # Fit and solve
        dk.fit()
        x = dk.solve()

        # Rescale x to account for normalization
        x = x * scale

        # Apply lower bound constraint
        x = np.maximum(x, self.lb)

        # Compute z-value
        z_value = self.landa1 * x.sum() + self.landa2 * np.linalg.norm(x)
        logging.debug(f"z_value={z_value:.4f}, x_norm={np.linalg.norm(x):.2e}")
        return z_value

    @ensure_fitted
    def predict(self, X):
        """Predict whether a sample is an inlier or outlier."""
        scores = self.decision_function(X)
        predictions = np.where(scores <= self.thr, 1, -1)
        return predictions

    @ensure_fitted
    def decision_function(self, X):
        """Compute anomaly scores for each sample."""
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        scores = np.array([self.__calculate_z__(sample.reshape(1, -1)) for sample in X])
        return scores

    @ensure_fitted
    def plot_contour(self, X=None, n_grid=50, title="DistEnetConvexHull Contour Plot", show=True):
        """Plot the decision boundary using a contour plot."""
        if self.m != 2:
            raise ValueError("Contour plot is only supported for 2D data.")
        X_plot = self.X_target if X is None else check_array(X, ensure_2d=True, dtype=np.float64)
        x_min, x_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
        y_min, y_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, n_grid),
                                     np.linspace(y_min, y_max, n_grid))
        z_grid = np.zeros((n_grid, n_grid))
        for i in tqdm(range(n_grid), desc="Computing contour"):
            for j in range(n_grid):
                sample = np.array([x_grid[i, j], y_grid[i, j]]).reshape(1, -1)
                z_grid[i, j] = self.__calculate_z__(sample)
        fig, ax = plt.subplots(figsize=(10, 6))
        contour = ax.contour(x_grid, y_grid, z_grid, levels=[self.thr], colors='red', linestyles='--')
        ax.contourf(x_grid, y_grid, z_grid, levels=50, cmap='coolwarm', alpha=0.3)
        ax.clabel(contour, inline=True, fontsize=10, fmt='thr=%.2f' % self.thr)
        if X is not None:
            predictions = self.predict(X)
            ax.scatter(X[predictions == 1, 0], X[predictions == 1, 1], c='blue', label='Inliers', alpha=0.5)
            ax.scatter(X[predictions == -1, 0], X[predictions == -1, 1], c='red', label='Outliers', marker='x')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        if show:
            plt.show()
        return fig

    def set_kernel_params(self, **params):
        """Set additional parameters for kernel computations."""
        self.kernel_params.update(params)

    def get_params(self, deep=True):
        """Get the parameters for this estimator."""
        return {
            'landa1': self.landa1,
            'target': self.target,
            'lb': self.lb,
            'alpha': self.alpha,
            'tau': self.tau,
            'node_max_iter': self.node_max_iter,
            'num_triggers': self.num_triggers,
            'graph': self.graph,
            'metric': self.metric,
            'only_target': self.only_target,
            'thr': self.thr,
            'kernel_params': self.kernel_params,
            'degree': self.kernel_params.get('degree', None),
            'gamma': self.kernel_params.get('gamma', None),
            'coef0': self.kernel_params.get('coef0', None),
            'random_state': self.random_state,
            'just_dag': self.just_dag,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        kernel_params = params.pop('kernel_params', {})
        for key in ['degree', 'gamma', 'coef0']:
            if key in params:
                kernel_params[key] = params.pop(key)
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter `{key}` for estimator `{self.__class__.__name__}`.")
        self.kernel_params.update(kernel_params)
        return self