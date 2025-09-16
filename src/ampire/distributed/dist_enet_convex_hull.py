from copy import deepcopy
import numpy as np
from scipy.linalg import solve_triangular
from sklearn.base import BaseEstimator, ClassifierMixin
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
    """Decorator to ensure the model is fitted before method execution."""
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "is_fitted_"):
            raise ValueError(f"This {self.__class__.__name__} instance is not fitted yet. Call `fit` before using this method.")
        return func(self, *args, **kwargs)
    return wrapper

class DistEnetConvexHull(BaseEstimator, ClassifierMixin):
    """
    One-Class Classifier for Anomaly Detection using Elastic Net and Convex Hull with DistributedKAMP solver.
    """
    def __init__(self, landa1=0.5, target=1, lb=None, alpha=0.5, tau=0.1,
                 node_max_iter=50, num_triggers=100, graph=None, metric='rbf',
                 only_target=True, thr=1.0, kernel_params=None, degree=3,
                 gamma='scale', coef0=1.0, random_state=None, just_dag=False,
                 verbose=False):
        self._estimator_type = "classifier"
        self.landa1 = landa1
        self.landa2 = 1 - self.landa1
        self.target = target
        self.lb = deepcopy(lb) if lb is not None else None  # Deep copy lb
        self.alpha = alpha
        self.tau = tau
        self.node_max_iter = node_max_iter
        self.num_triggers = num_triggers
        self.graph = deepcopy(graph) if graph is not None else None  # Deep copy graph
        self.metric = metric
        self.only_target = only_target
        self.thr = thr
        self.kernel_params = deepcopy(kernel_params) if kernel_params else {}  # Deep copy
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.random_state = random_state
        self.just_dag = just_dag
        self.verbose = verbose

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
        if not isinstance(self.thr, (int, float)) or self.thr < 0:  # Allow thr=0
            raise ValueError(f"thr ({self.thr}) must be a non-negative float.")
        if self.random_state is not None and not isinstance(self.random_state, int):
            raise ValueError(f"random_state must be an integer or None. Got {type(self.random_state)}.")
        if not isinstance(self.just_dag, bool):
            raise ValueError(f"just_dag ({self.just_dag}) must be a boolean.")

    def _validate_kernel_params(self, X=None):
        """Validate and adjust kernel-specific parameters."""
        kernel_params = self.kernel_params.copy()  # Work on a copy
        if self.metric in {'poly', 'rbf', 'sigmoid'} and 'gamma' not in kernel_params:
            if self.gamma == 'scale' and X is not None:
                kernel_params['gamma'] = 1.0 / (X.shape[1] * X.var()) if X.shape[0] > 0 else 1.0
            elif self.gamma == 'auto':
                kernel_params['gamma'] = 1.0 / X.shape[1] if X is not None and X.shape[0] > 0 else 1.0
            else:
                kernel_params['gamma'] = self.gamma
        if self.metric == 'poly':
            kernel_params['degree'] = kernel_params.get('degree', self.degree)
            kernel_params['coef0'] = kernel_params.get('coef0', self.coef0)
        if self.metric == 'sigmoid':
            kernel_params['coef0'] = kernel_params.get('coef0', self.coef0)
        logging.debug(f"Validated kernel_params: {kernel_params}")
        return kernel_params

    def _adjust_kernel(self, X, Y=None):
        """Compute adjusted pairwise kernel similarity matrix."""
        kernel_params = self._validate_kernel_params(X)
        if Y is None:
            Y = X
        G = pairwise_kernels(X, Y, metric=self.metric, **kernel_params)
        G_sum = G.sum()
        row_sum = G.sum(axis=1, keepdims=True)
        col_sum = G.sum(axis=0, keepdims=True)
        return G - (row_sum + col_sum) / X.shape[0] + G_sum / (X.shape[0] ** 2)

    def _calculate_P(self, X):
        """Compute optimization matrix P."""
        BTB = self._adjust_kernel(X)
        P = (self.landa2 * np.identity(X.shape[0])) + BTB
        return P

    def _calculate_q(self, G: np.ndarray, Ky: np.ndarray) -> np.ndarray:
        """Calculate vector q for the quadratic problem (used for ThresholdFinder)."""
        n = G.shape[0]
        G_sum = G.sum()
        Ky_sum = Ky.sum()
        row_sum = G.sum(axis=1, keepdims=True)
        h = Ky - (Ky_sum + row_sum) / n + G_sum / (n ** 2)
        return self.landa1 * np.ones((n, 1)) - 2 * h

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
        """Convert QP problem to KAMP format (y = A @ x)."""
        try:
            L = np.linalg.cholesky(P)
            y = -0.5 * solve_triangular(L, q, lower=True)
            A = L.T
        except np.linalg.LinAlgError:
            logging.error("Cholesky decomposition failed. Using fallback method.")
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.clip(eigvals, 0, None)
            L = eigvecs @ np.diag(np.sqrt(eigvals))
            y = -0.5 * solve_triangular(L, q, lower=False, check_finite=False)
            A = L.T

        A_norm = np.linalg.norm(A, ord='fro')
        y_norm = np.linalg.norm(y)
        scale_A = max(A_norm, 1e-10)
        scale_y = max(y_norm, 1e-10)
        A = A / scale_A
        y = y / scale_y
        # Use the number of nodes from the graph
        num_nodes = len(self.graph.nodes) if self.graph is not None else 1
        if num_nodes == 1:
            A_list = [A]
            y_list = [y]
        else:
            A_list = np.array_split(A, num_nodes, axis=0)
            y_list = np.array_split(y, num_nodes, axis=0)
        if len(A_list) != num_nodes or len(y_list) != num_nodes:
            logging.error(f"Mismatch in A_list (len={len(A_list)}) and y_list (len={len(y_list)}) with graph nodes ({num_nodes})")
            raise ValueError(f"Mismatch in A_list and y_list lengths with graph nodes: {len(A_list)}, {len(y_list)}, {num_nodes}")
        logging.info(f"Converted QP to KAMP: num_nodes={num_nodes}, A_list length={len(A_list)}, y_list length={len(y_list)}")
        return A_list, y_list, scale_A * scale_y

        num_nodes = min(5, A.shape[0])
        A_list = np.array_split(A, num_nodes, axis=0)
        y_list = np.array_split(y, num_nodes, axis=0)

        return A_list, y_list, scale_A * scale_y

    def fit(self, X, y=None):
        """Fit the DistEnetConvexHull model to the given data."""
        self._validate_params()
        if y is not None:
            X, y = check_X_y(X, y, accept_sparse=False, ensure_2d=True, dtype=np.float64)
            self.return_label = True
            self.classes_ = np.unique(y)
            mask = (y == self.target)
            self.X_target = X[mask, :] if self.only_target else X
        else:
            X = check_array(X, ensure_2d=True, dtype=np.float64)
            self.X_target = X

        self.G = pairwise_kernels(self.X_target, metric=self.metric, **self._validate_kernel_params(self.X_target))
        self.n, self.m = self.X_target.shape
        self.P = self._calculate_P(self.X_target)
        self.lb = np.zeros((self.n, 1)) if self.lb is None else self.lb
        self.is_fitted_ = True

        if self.graph is None:
            self.graph = self._create_default_graph(num_nodes=5)

        return self

    @ensure_fitted
    def __calculate_z__(self, sample):
        Ky = self._adjust_kernel(self.X_target, sample)
        h = Ky - (Ky.sum() + self.G.sum(axis=1, keepdims=True)) / self.X_target.shape[0]
        h += self.G.sum() / (self.X_target.shape[0] ** 2)
        q = self.landa1 * np.ones((self.n, 1)) - 2 * h
        A_list, y_list, scale = self._convert_qp_to_kamp(self.P, q)
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
        dk.fit()
        x = dk.solve()
        logging.debug(f"x before scaling: {x.flatten()[:5]}, norm={np.linalg.norm(x):.2e}")
        x = x * scale
        logging.debug(f"x after scaling: {x.flatten()[:5]}, norm={np.linalg.norm(x):.2e}")
        x = np.maximum(x, self.lb)
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

    def get_params(self, deep=True):
        """Get the parameters for this estimator."""
        params = {
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
            'kernel_params': deepcopy(self.kernel_params) if deep else self.kernel_params,
            'degree': self.degree,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'random_state': self.random_state,
            'just_dag': self.just_dag,
            'verbose': self.verbose
        }
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            if key == 'kernel_params':
                self.kernel_params = deepcopy(value) if value is not None else {}
            elif key in ['degree', 'gamma', 'coef0']:
                self.__setattr__(key, value)
                if key not in self.kernel_params:
                    self.kernel_params[key] = value
            elif hasattr(self, key):
                self.__setattr__(key, value)
            else:
                raise ValueError(f"Invalid parameter `{key}` for estimator `{self.__class__.__name__}`.")
        return self