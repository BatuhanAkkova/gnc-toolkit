"""
Ensemble Kalman Filter (EnKF) using Monte Carlo samples for covariance representation.
"""

import numpy as np
from typing import Callable, Any


class EnKF:
    """
    Ensemble Kalman Filter (EnKF).

    Uses an ensemble of states to represent the error covariance matrix.
    Highly efficient for high-dimensional systems (e.g., weather/climate models)
    where the full covariance matrix is too large to compute.

    Parameters
    ----------
    dim_x : int
        Dimension of the state vector.
    dim_z : int
        Dimension of the measurement vector.
    ensemble_size : int, optional
        Number of ensemble members (N). Default is 50.
    """

    def __init__(self, dim_x: int, dim_z: int, ensemble_size: int = 50):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.num_ensemble = ensemble_size

        # Ensemble of states: shape (dim_x, N)
        self.X = np.zeros((dim_x, self.num_ensemble))
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

    def initialize_ensemble(self, x_mean: np.ndarray, p_cov: np.ndarray) -> None:
        """
        Initialize the ensemble using a multivariate normal distribution.

        Parameters
        ----------
        x_mean : np.ndarray
            Mean initial state (dim_x,).
        p_cov : np.ndarray
            Initial state error covariance (dim_x, dim_x).
        """
        self.X = np.random.multivariate_normal(x_mean, p_cov, self.num_ensemble).T

    def predict(
        self,
        dt: float,
        fx_func: Callable,
        q_mat: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Predict step (Propagates each ensemble member).

        Parameters
        ----------
        dt : float
            Time step (s).
        fx_func : Callable
            Nonlinear state transition function $f(x, dt, **kwargs) \to x_{new}$.
        q_mat : np.ndarray, optional
            Process noise covariance (dim_x, dim_x). If None, uses `self.Q`.
        **kwargs : Any
            Additional arguments passed to transition function.
        """
        q_curr = q_mat if q_mat is not None else self.Q

        # Propagate each ensemble member
        for i in range(self.num_ensemble):
            # Propagate through nonlinear model
            self.X[:, i] = fx_func(self.X[:, i], dt, **kwargs)

            # Add process noise to each member
            noise = np.random.multivariate_normal(np.zeros(self.dim_x), q_curr)
            self.X[:, i] += noise

    def update(
        self,
        z: np.ndarray,
        hx_func: Callable,
        r_mat: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        r"""
        Update step (Ensemble transformation).

        Parameters
        ----------
        z : np.ndarray
            Measurement vector (dim_z,).
        hx_func : Callable
            Nonlinear measurement function $h(x, **kwargs) \to z_{pred}$.
        r_mat : np.ndarray, optional
            Measurement noise covariance (dim_z, dim_z). If None, uses `self.R`.
        **kwargs : Any
            Additional arguments passed to measurement function.
        """
        r_curr = r_mat if r_mat is not None else self.R

        # Transform ensemble to measurement space
        z_ensemble = np.zeros((self.dim_z, self.num_ensemble))
        for i in range(self.num_ensemble):
            z_ensemble[:, i] = hx_func(self.X[:, i], **kwargs)

        # Sample mean of measurement ensemble
        z_mean = np.mean(z_ensemble, axis=1, keepdims=True)

        # Calculate ensemble anomalies (perturbations)
        x_mean_vec = np.mean(self.X, axis=1, keepdims=True)
        anomalies_x = self.X - x_mean_vec  # State anomalies
        anomalies_z = z_ensemble - z_mean  # Measurement anomalies

        # Perturbed measurements (adding noise for each ensemble member)
        z_perturbed = np.zeros((self.dim_z, self.num_ensemble))
        for i in range(self.num_ensemble):
            noise = np.random.multivariate_normal(np.zeros(self.dim_z), r_curr)
            z_perturbed[:, i] = z + noise

        # Innovation
        innov_ensemble = z_perturbed - z_ensemble

        # Innovation covariance S = (1/(N-1)) * B * B.T + R
        s_mat = (1.0 / (self.num_ensemble - 1)) * (anomalies_z @ anomalies_z.T) + r_curr

        # Cross-covariance Pxz = (1/(N-1)) * A * B.T
        pxz = (1.0 / (self.num_ensemble - 1)) * (anomalies_x @ anomalies_z.T)

        # Kalman Gain K = Pxz * inv(S)
        k_gain = pxz @ np.linalg.inv(s_mat)

        # Correct each ensemble member
        self.X += k_gain @ innov_ensemble

    @property
    def x(self) -> np.ndarray:
        """Ensemble mean state vector."""
        return np.mean(self.X, axis=1)

    @property
    def P(self) -> np.ndarray:
        """Ensemble covariance matrix."""
        anomalies_x = self.X - np.mean(self.X, axis=1, keepdims=True)
        return (1.0 / (self.num_ensemble - 1)) * (anomalies_x @ anomalies_x.T)
