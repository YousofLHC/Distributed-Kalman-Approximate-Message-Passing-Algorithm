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
        self.A = check_array(A)
        self.y = check_array(y, ensure_2d=False).reshape(-1, 1)
        self.m, self.n = self.A.shape
        self.x = np.zeros((self.n, 1))
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
        for _ in range(self.max_iter):
            x_prev = x.copy()
            r = x + self.A.T @ z
            x = self.denoiser(r, self.tau)
            # Use subdif_denoiser to compute the derivative for Onsager correction
            subdif = self.subdif_denoiser(r, self.tau)
            # Compute the Onsager correction term using the mean of the sub-differential
            onsager_term = np.mean(subdif) * z
            z = self.y - self.A @ x + self.alpha * onsager_term
            # Check for convergence
            diff_norm = np.linalg.norm(x - x_prev)
            prev_norm = np.linalg.norm(x_prev) + 1e-8
            if prev_norm > 0 and diff_norm / prev_norm < self.tol:
                break
            # Check for NaN/inf values
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                break
        self.x = x
        return x