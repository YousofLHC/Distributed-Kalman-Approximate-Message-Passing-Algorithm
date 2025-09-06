import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_array
from typing import Optional
import sys
import os

class AMP(BaseEstimator):
    """
    Approximate Message Passing (AMP) for compressed sensing.

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
        Custom denoising function.
    """
    def __init__(self, alpha: float = 0.5, tau: float = 0.1, max_iter: int = 100, tol: float = 1e-8, denoiser=None):
        self.alpha = alpha
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol
        self.denoiser = denoiser or (lambda r, tau: np.sign(r) * np.maximum(np.abs(r) - tau, 0))

    def fit(self, A, y):
        self.A = check_array(A)
        self.y = check_array(y, ensure_2d=False).reshape(-1, 1)
        self.m, self.n = self.A.shape
        self.x = np.zeros((self.n, 1))
        self.is_fitted_ = True
        return self

    def solve(self):
        check_is_fitted(self, "is_fitted_")
        x = self.x.copy()
        z = self.y.copy()
        for _ in range(self.max_iter):
            x_prev = x.copy()
            r = x + self.A.T @ z
            x = self.denoiser(r, self.tau)
            z = self.y - self.A @ x + z * (np.sum(x != 0) / self.m)
            if np.linalg.norm(x - x_prev) < self.tol:
                break
        return x