import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from typing import Optional, Callable, Tuple
from numpy.typing import NDArray    

class KAMP(BaseEstimator):
    """
    Kalman-based Approximate Message Passing (KAMP) for compressed sensing.

    Parameters
    ----------
    alpha : float
        Damping factor in [0, 1].
    
    tau : float
        Thresholding parameter, must be in ℝ.
    
    max_iter : int
        Maximum number of iterations, must be positive.
    
    tol : float, default=1e-8
        Convergence tolerance.
    
    denoiser : callable, optional
        Custom denoising function, defaults to soft thresholding: sgn(r) * max(|r| - τ, 0).
    
    subdif_denoiser : callable, optional
        Derivative of denoising function, defaults to indicator: 1_{|r| > τ}.
    """
    def __init__(self,
                 alpha          : float,
                 tau            : float,
                 max_iter       : int,
                 tol            : float = 1e-8,
                 denoiser       : Optional[Callable[[NDArray, float], NDArray]] = None,
                 subdif_denoiser: Optional[Callable[[NDArray, float], NDArray]] = None): 
        
        if not 0 <= alpha <= 1:
            raise ValueError("`alpha` must be in [0, 1]")
        if max_iter <= 0:
            raise ValueError("`max_iter` must be positive")
        if tol <= 0:
            raise ValueError("`tol` must be positive")
        
        self.alpha           = alpha
        self.tau             = tau
        self.max_iter        = max_iter
        self.tol             = tol
        self.denoiser        = denoiser or (lambda r, tau: np.sign(r) * np.maximum(np.abs(r) - tau, 0))
        self.subdif_denoiser = subdif_denoiser or (lambda r, tau: np.squeeze((np.abs(r) > tau).astype(float)))

    def fit(self, A: NDArray, y: NDArray):
        """
        Initialize the model with sensing matrix and observations.

        Parameters
        ----------
        A : NDArray, shape (m, n)
            Sensing matrix.
        y : NDArray, shape (m, 1)
            Observations.

        Returns
        -------
        self : KAMP
            Fitted estimator.
        """
        self.A = check_array(A)
        self.y = check_array(y, ensure_2d=False).reshape(-1, 1)
        self.m, self.n = self.A.shape
        self.x = np.zeros((self.n, 1))
        self.P = np.eye(self.n)
        self.Q = np.eye(self.n)
        self.R = np.eye(self.m)
        self.is_fitted_ = True
        return self

    def _update_prior_estimation(self, r: NDArray) -> NDArray:
        """Compute prior estimate."""
        return self.denoiser(r, self.tau)

    def _update_jacobian(self, r: NDArray) -> NDArray:
        """Compute Jacobian of denoising function."""
        return np.diag(self.subdif_denoiser(r, self.tau).flatten()) @ (np.eye(self.n) - self.A.T @ self.A)

    def _update_prior_covariance(self, J: NDArray, P: NDArray, Q: NDArray) -> NDArray:
        """Update prior covariance."""
        return J @ P @ J.T + Q

    def _update_kalman_gain(self, P_: NDArray, R: NDArray) -> NDArray:
        """Compute Kalman gain."""
        return P_ @ self.A.T @ np.linalg.inv(self.A @ P_ @ self.A.T + R)

    def _update_prior_residual(self, x_: NDArray) -> NDArray:
        """Compute prior residual."""
        return self.y - self.A @ x_

    def _update_estimation(self, x_: NDArray, G: NDArray, r_: NDArray) -> NDArray:
        """Update state estimate."""
        return x_ + G @ r_

    def _update_covariance(self, G: NDArray, P_: NDArray) -> NDArray:
        """Update covariance matrix."""
        return (np.eye(self.n) - G @ self.A) @ P_

    def _update_process_noise_covariance(self, G: NDArray, r_: NDArray) -> NDArray:
        """Update process noise covariance."""
        return self.alpha * self.Q + (1 - self.alpha) * (G @ r_) @ (G @ r_).T

    def solve(self) -> NDArray:
        """
        Solve the sparse signal recovery problem.

        Returns
        -------
        x : NDArray, shape (n, 1)
            Estimated signal.
        """
        check_is_fitted(self, "is_fitted_")
        x_prev = np.copy(self.x)
        for _ in range(self.max_iter):
            # Prediction phase
            r = self.y - self.A @ self.x
            x_ = self._update_prior_estimation(r)
            J = self._update_jacobian(r)
            P_ = self._update_prior_covariance(J, self.P, self.Q)
            # Correction phase
            G = self._update_kalman_gain(P_, self.R)
            r_ = self._update_prior_residual(x_)
            self.x = self._update_estimation(x_, G, r_)
            self.P = self._update_covariance(G, P_)
            self.Q = self._update_process_noise_covariance(G, r_)
            # Check convergence
            if np.linalg.norm(self.x - x_prev) / (np.linalg.norm(x_prev) + 1e-10) < self.tol:
                break
            x_prev = np.copy(self.x)
        return self.x