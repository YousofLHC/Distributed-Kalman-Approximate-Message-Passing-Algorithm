import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_array
from typing import Optional, Callable
from numpy.typing import NDArray

class AMP(BaseEstimator):
    """
    Approximate Message Passing (AMP) for compressed sensing with sub-differential denoising.

    Parameters
    ----------
    alpha : float
        Damping factor in [0, 1].
    tau : float
        Thresholding parameter.
    max_iter : int
        Maximum number of iterations.
    tol : float, default=1e-8
        Convergence tolerance.
    denoiser : callable, optional
        Custom denoising function, defaults to soft thresholding: sgn(r) * max(|r| - τ, 0).
    subdif_denoiser : callable, optional
        Derivative of denoising function, defaults to indicator: 1_{|r| > τ}.
    """
    def __init__(self,
                 alpha: float = 0.5,
                 tau: float = 0.1,
                 max_iter: int = 100,
                 tol: float = 1e-8,
                 denoiser: Optional[Callable[[NDArray, float], NDArray]] = None,
                 subdif_denoiser: Optional[Callable[[NDArray, float], NDArray]] = None):
        if not 0 <= alpha <= 1:
            raise ValueError("`alpha` must be in [0, 1]")
        if max_iter <= 0:
            raise ValueError("`max_iter` must be positive")
        if tol <= 0:
            raise ValueError("`tol` must be positive")

        self.alpha = alpha
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol
        self.denoiser = denoiser or (lambda r, tau: np.sign(r) * np.maximum(np.abs(r) - tau, 0))
        self.subdif_denoiser = subdif_denoiser or (lambda r, tau: np.squeeze((np.abs(r) > tau).astype(float)))

    def fit(self, A: NDArray, y: NDArray):
        """
        Initialize the model with sensing matrix and observations.

        Parameters
        ----------
        A : NDArray, shape (m, n)
            Sensing matrix.
        y : NDArray, shape (m,)
            Observations.

        Returns
        -------
        self : AMP
            Fitted estimator.
        """
        self.A = check_array(A, dtype=np.float64)
        self.y = check_array(y, ensure_2d=False, dtype=np.float64).reshape(-1, 1)
        self.m, self.n = self.A.shape
        
        # Normalize A and y to prevent large values
        A_norm = np.linalg.norm(self.A, ord='fro')
        y_norm = np.linalg.norm(self.y)
        self.scale_A = max(A_norm, 1e-10)
        self.scale_y = max(y_norm, 1e-10)
        self.A = self.A / self.scale_A
        self.y = self.y / self.scale_y
        
        self.x = np.zeros((self.n, 1), dtype=np.float64)
        self.is_fitted_ = True
        return self

    def solve(self) -> NDArray:
        """
        Solve the sparse signal recovery problem using AMP.

        Returns
        -------
        x : NDArray, shape (n, 1)
            Estimated signal.
        """
        check_is_fitted(self, "is_fitted_")
        x = self.x.copy()
        z = self.y.copy()
        for i in range(self.max_iter):
            x_prev = x.copy()
            # Compute r with clipping to prevent overflow
            r = x + np.clip(self.A.T @ z, -1e10, 1e10)
            x = np.clip(self.denoiser(r, self.tau), -1e10, 1e10)
            # Use subdif_denoiser to compute the derivative for Onsager correction
            subdif = self.subdif_denoiser(r, self.tau)
            # Compute the Onsager correction term with clipping
            onsager_term = np.clip(np.mean(subdif) * z, -1e10, 1e10)
            z = np.clip(self.y - self.A @ x + self.alpha * onsager_term, -1e10, 1e10)
            # Log debug information
            import logging
            logging.debug(f"Iteration {i}: x_norm={np.linalg.norm(x):.2e}, z_norm={np.linalg.norm(z):.2e}, r_norm={np.linalg.norm(r):.2e}")
            # Check for convergence
            diff_norm = np.linalg.norm(x - x_prev)
            prev_norm = np.linalg.norm(x_prev)
            if not np.isnan(diff_norm) and not np.isinf(diff_norm) and not np.isnan(prev_norm) and not np.isinf(prev_norm):
                if prev_norm > 1e-4 and diff_norm / prev_norm < self.tol and not np.any(np.isnan(x)) and not np.any(np.isinf(x)):
                    logging.debug(f"Converged at iteration {i}: diff_norm/prev_norm={diff_norm/prev_norm:.2e}")
                    break
            else:
                logging.debug(f"Invalid norms in AMP.solve: diff_norm={diff_norm:.2e}, prev_norm={prev_norm:.2e}")
            # Check for NaN/inf values in x
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                logging.debug(f"NaN or inf detected in x at iteration {i}: x_norm={np.linalg.norm(x):.2e}")
                break
        self.x = x * self.scale_A * self.scale_y  # Rescale x to original scale
        return self.x