import numpy as np
from scipy.linalg import solve_triangular
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_array
from typing import Optional, Callable, Tuple
from numpy.typing import NDArray
from tqdm import tqdm
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
                  subdif_denoiser: Optional[Callable[[NDArray, float], NDArray]] = None,
                  verbose        : bool = False):

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
        self.verbose         = verbose
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
        #-------------------- Problem Data Information -------------
        
        self.A            = check_array(A) # Sensing matrix, A ∈ ℝ^{m×n}
        self.AT           = self.A.T
        self.y            = check_array(y, ensure_2d=False).reshape(-1, 1) # Observations, y ∈ ℝ^{m×1}
        self.m, self.n    = self.A.shape # (samples, features)
        self.sigma2       = min(self.m,self.n)/max(self.m,self.n)# Noise variance, σ^2 = min(m,n)/max(m,n)
        
        #-------------------- Algorithm Initial Step -------------

        self.x            = np.zeros((self.n, 1)) # Initialize signal estimate to zeros for stability
        self.I_n          = np.eye(self.n) # Identity matrix, I_n ∈ ℝ^{n×n}
        self.z            = self.y.copy() # Initial residual, z_{[0]} = y - A * x̂_{[0]}
        self._P           = self.sigma2 * self.I_n # Initial covariance matrix, smaller value for stability
        self.Q            = self.sigma2 * self.I_n # Initial process noise covariance, smaller value for stability
        self.R            = self.sigma2 * np.eye(self.m) # Measurement noise covariance, R= σ^2 * I_m
        self.is_fitted_   = True
        return self

    def __update_prior_estimation(self, r: NDArray) -> NDArray:
        """Compute prior estimate: x̂_{[t]}^{-} = η(r_{[t-1]}; τ).
        
        Parameters
        ----------
        r : NDArray, shape(n, 1)
            Residual.
        
        Returns
        -------
        NDArray, shape(n, 1)
            Prior estimate.
        """
        return self.denoiser(r, self.tau)

    def __update_jacobian(self, r: NDArray) -> NDArray:
        """Compute Jacobian of denoising function: J_η = diag(η'(r)) * (I_n - A^T * A).
        
        Parameters
        ---------
        r : NDArray, shape(n, 1)
            Residual.
        
        Returns
        -------
        NDArray, shape(n, n)
            Jacobian matrix.
        """
        d = self.subdif_denoiser(r, self.tau).flatten() # shape(n, )
        return np.diag(d) @ (self.I_n - self.AT @ self.A) # shape(n, n)

    def __update_prior_covariance(self, J: NDArray, P: NDArray, Q: NDArray) -> NDArray:
        """Update prior covariance matrix: P_{[t]}^{-} = J_η * P_{[t-1]} * J_η^T + Q_{[t-1]}.
        
        Parameters
        ----------
        J : NDArray, shape(n, n)
            Jacobian matrix.
        P : NDArray, shape(n, n)
            Previous covariance matrix.
        Q : NDArray, shape(n, n)
            Process noise covariance.
        
        Returns
        -------
        NDArray, shape(n, n)
            Updated prior covariance
        """
        P_ = J @ P @ J.T + Q  # n × n
        return np.clip(P_, -1e6, 1e6) # Clip to prevent overflow in large matrices

    def __update_kalman_gain(self, P_: NDArray, R: NDArray) -> NDArray:
        """Compute Kalman gain: G_{[t]} = P_{[t]}^{-} * A^T * (A * P_{[t]}^{-} * A^T + R)^{-1}.

        Parameters
        ----------
        P_ : NDArray, shape(n, n)
            Prior covariance matrix.
        R : NDArray, shape(m, m)
            Measurement noise covariance.

        Returns
        -------
        NDArray, shape(n, m)
            Kalman gain.
        """
        P_AT   = P_ @ self.AT                             # shape(n, m)
        matrix = self.A @ P_AT + R
        epsilon = 1e-6 *np.linalg.norm(matrix, ord=2)  # Small regularization to stabilize inverse
        matrix_reg = matrix + epsilon * np.eye(matrix.shape[0])
        # Use Cholesky decomposition for stable and fast inversion of PD matrix
        try:
            L = np.linalg.cholesky(matrix_reg)
            temp = solve_triangular(L, P_AT.T, lower=True)
            G_T = solve_triangular(L.T, temp, lower=False)
            return G_T.T  # shape(n, m)
        except np.linalg.LinAlgError:
            # Fallback to pinv if Cholesky fails (though unlikely with regularization)
            return P_AT @ np.linalg.pinv(matrix_reg)

    def __update_prior_residual(self, x_: NDArray) -> NDArray:
        """Compute prior residual: r_{[t]}^{-} = y - A * x̂_{[t]}^{-}.

        Parameters
        ----------
        x_ : NDArray, shape(n, 1)
            Prior estimate.

        Returns
        -------
        NDArray, shape(m, 1)
            Prior residual.
        """
        # Clip x_ to prevent overflow
        x_clipped = np.clip(x_, -1e10, 1e10)
        result = self.y - self.A @ x_clipped
        # Handle potential NaN/inf values
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
        return result

    def __update_estimation(self, x_: NDArray, G: NDArray, r_: NDArray) -> NDArray:
        """Update state estimate: x̂_{[t]} = x̂_{[t]}^{-} + G_{[t]} * (y - A * x̂_{[t]}^{-})."""
        return x_ + G @ r_ # n × 1

    def __update_covariance(self, G: NDArray, P_: NDArray) -> NDArray:
        """Update covariance matrix: P_{[t]} = (I_n - G_{[t]} * A) * P_{[t]}^{-}.
        
        Parameters
        ----------
        G : NDArray, shape(n, m)
            Kalamn gain.
        P_ : NDArray, shape(n, n)
            Prior covariance matrix.
        
        Returns
        -------
        NDArray, shape(n, n)
            Updated covariance.
        """
        P = (self.I_n - G @ self.A) @ P_
        return np.clip(P, -1e6, 1e6) # Clip to avoid overflow

    def __update_process_noise_covariance(self, G: NDArray, r_: NDArray) -> NDArray:
        """Update process noise covariance: Q_{[t]} = α * Q_{[t-1]} + (1-α) * (G_{[t]} * r_{[t]}^{-}) * (G_{[t]} * r_{[t]}^{-})^T.
        
        Parameters
        ----------
        G : NDArray, shape(n, m)
            Kalman gain.
        r_ : NDArray, shape(m, 1)
            Prior residual
            
        Returns
        -------
        NDArray, shape(n, n)
            Updated process noise covariance.
        """
        Gv_ = G @ r_            # shape(n, 1)
        Q   = self.alpha * self.Q + (1 - self.alpha) * (Gv_) @ (Gv_.T)
        return np.clip(Q, -1e6, 1e6) # Clip to avoid overflow

    
    def __update_r(self, x_prev: NDArray, z_prev: NDArray)-> NDArray:
        """Compute residual: r_{[t-1]} = x̂_{[t-1]} + A^T * z_{[t-1]}.

        Parameters
        ----------
        x_prev : NDArray, shape(n, 1)
            Previous estimate.
        z_prev : NDArray, shape(m, 1)
            Previous residual.

        Returns
        -------
        NDArray, shape(n, 1)
            Updated residual.
        """
        # Clip inputs to prevent overflow
        x_clipped = np.clip(x_prev, -1e10, 1e10)
        z_clipped = np.clip(z_prev, -1e10, 1e10)
        result = x_clipped + self.AT @ z_clipped
        # Handle potential NaN/inf values
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
        return result
    
    @property
    def P(self):
        return self._P
    @P.setter
    def P(self, value):
        self._P=value
    
    def solve(self) -> NDArray:
        """
        Solve the sparse signal recovery problem.

        Returns
        -------
        x : NDArray, shape (n, 1)
            Estimated signal.
        """
        check_is_fitted(self, "is_fitted_")
        for _ in (des:=tqdm(range(self.max_iter),ascii=True, leave=False,colour='blue')):#range(self.max_iter):
            des.set_description(f'\33[31m[Thresholding phase]\33[33m')
            
            x_prev = np.copy(self.x) # Store previous estimate,                                   x̂_{[t-1]}
            r      = self.__update_r(x_prev,self.z) # Compute r_{[t-1]}                          = x̂_{[t-1]} + A^T * z_{[t-1]}
            x_     = self.__update_prior_estimation(r) # Compute prior estimate,                   x̂_{[t]}^{-} = η(r_{[t-1]}; τ)
            J      = self.__update_jacobian(r) # Compute Jacobian,                                 J_η          = diag(η'(r)) * (I_n - A^T * A)
            P_     = self.__update_prior_covariance(J, self._P, self.Q) # Update prior covariance, P_{[t]}^{-}  = J_η * P_{[t-1]} * J_η^T + Q_{[t-1]}
            
            des.set_description(f'\33[32m[Correction]\33[33m')
            
            G       = self.__update_kalman_gain(P_, self.R) # Compute Kalman gain,          G_{[t]}     = P_{[t]}^{-} * A^T * (A * P_{[t]}^{-} * A^T + R)^{-1}
            r_      = self.__update_prior_residual(x_) # Compute prior residual,            r_{[t]}^{-} = y - A * x̂_{[t]}^{-}
            self.x  = self.__update_estimation(x_, G, r_) # Update estimate,                x̂_{[t]}    = x̂_{[t]}^{-} + G_{[t]} * (y - A * x̂_{[t]}^{-})
            self._P = self.__update_covariance(G, P_) # Update covariance,                  P_{[t]}     = (I_n - G_{[t]} * A) * P_{[t]}^{-}
            self.Q  = self.__update_process_noise_covariance(G, r_) # Update process noise, Q_{[t]}     = α * Q_{[t-1]} + (1-α) * (G_{[t]} * r_{[t]}^{-}) * (G_{[t]} * r_{[t]}^{-})^T
            self.z  = self.y - self.A @ self.x # Update residual,                          z_{[t]}     = y - A * x̂_{[t]}^{-}

            #--------------------------Check convergence------------------------------

            # Check for NaN/inf values
            if np.any(np.isnan(self.x)) or np.any(np.isinf(self.x)):
                if self.verbose:
                    print("Warning: NaN or inf values detected, stopping iteration")
                break

            diff_norm = np.linalg.norm(self.x - x_prev)
            prev_norm = np.linalg.norm(x_prev)
            if prev_norm > 0 and diff_norm / prev_norm < self.tol:
                break
        return self.x