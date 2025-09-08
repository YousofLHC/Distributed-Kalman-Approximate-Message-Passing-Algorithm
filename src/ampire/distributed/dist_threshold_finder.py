from copy import deepcopy
import numpy as np
from scipy.linalg import solve_triangular
from sklearn.metrics.pairwise import pairwise_kernels
from ampire.distributed import DistributedKAMP
from ampire.distributed import DistEnetConvexHull
from tqdm import tqdm
import logging
import os

class DistThresholdFinder:
    """
    Find the optimal threshold for DistEnetConvexHull using DistributedKAMP.

    Parameters
    ----------
    model : DistEnetConvexHull
        The fitted DistEnetConvexHull model.
    X : ndarray of shape (n_samples, n_features)
        Training data.
    """
    def __init__(self, model, X):
        if not isinstance(model, DistEnetConvexHull):
            raise ValueError(f"Model must be an instance of DistEnetConvexHull. Got {type(model)}.")
        if not hasattr(model, "is_fitted_"):
            raise ValueError("Model is not fitted yet. Call `fit` on the model first.")
        self.model = model
        self.X = X
        self.XCopy = deepcopy(X)
        self.n, self.m = X.shape
        self.lb = np.zeros((self.n - 1, 1)) if model.lb is None else model.lb[:-1, :]
        
        # Setup logging
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, 'dist_threshold_finder.log'),
            level=logging.DEBUG if model.verbose else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _convert_qp_to_kamp(self, P, q):
        """
        Convert QP problem to KAMP format (y = A @ x).
        """
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

        num_nodes = min(5, A.shape[0])
        A_list = np.array_split(A, num_nodes, axis=0)
        y_list = np.array_split(y, num_nodes, axis=0)

        return A_list, y_list, scale_A * scale_y


    def find(self, outs='max', specific_index=None):
        """
        Find anomaly scores for training samples using leave-one-out cross-validation.

        Parameters:
        - outs: str or tuple of str, aggregation method(s) for z-scores (e.g., 'max', 'mean')
        - specific_index: int, optional, compute score for a single sample index

        Returns:
        - z: array, anomaly scores for all samples
        - aggregated z: max, mean, or tuple of aggregated scores
        """
        z = np.zeros((self.n, 1))
        indices = [specific_index] if specific_index is not None else range(self.n)

        for row in tqdm(indices, leave=False, total=len(indices), desc="Calculating z"):
            logging.info(f"Processing sample {row}/{len(indices)}")
            eliminated_X = np.delete(self.XCopy, row, axis=0)
            self.model._validate_kernel_params(X=eliminated_X)
            G = pairwise_kernels(eliminated_X, metric=self.model.metric, **self.model.kernel_params)
            P = self.model._calculate_P(eliminated_X)
            Ky = pairwise_kernels(eliminated_X, self.X[row].reshape((1, self.m)),
                                  metric=self.model.metric, **self.model.kernel_params)
            q = self.model._calculate_q(G, Ky)
            A_list, y_list, scale = self._convert_qp_to_kamp(P, q)
            dk = DistributedKAMP(
                alpha=self.model.alpha,
                tau=self.model.tau,
                node_max_iter=self.model.node_max_iter,
                num_triggers=self.model.num_triggers,
                graph=self.model.graph,
                A_list=A_list,
                y_list=y_list,
                random_state=self.model.random_state,
                just_dag=self.model.just_dag,
                verbose=self.model.verbose
            )
            dk.fit()
            x_opt = dk.solve()
            x_opt = x_opt * scale
            x_opt = np.maximum(x_opt, self.lb)
            z[row, 0] = self.model.landa1 * np.sum(x_opt) + self.model.landa2 * np.linalg.norm(x_opt)
            logging.debug(f"z[{row}]={z[row, 0]:.4f}, x_opt_norm={np.linalg.norm(x_opt):.2e}")

        if specific_index is not None:
            return z[specific_index, 0], z[specific_index, 0]

        if isinstance(outs, str):
            return z, getattr(np, outs)(z)
        return z, tuple(getattr(np, outs)(z) for func in outs)
   
