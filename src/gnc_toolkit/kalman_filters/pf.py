"""
Particle Filter (Sequential Importance Resampling) for non-Gaussian/non-linear systems.
"""

from collections.abc import Callable
from typing import Any

import numpy as np


class ParticleFilter:
    """
    Bootstrap Particle Filter (Sequential Importance Resampling).

    Estimated non-Gaussian distributions and handles highly non-linear dynamics
    by propagating a set of discrete particles (samples).

    Parameters
    ----------
    dim_x : int
        Dimension of the state vector.
    dim_z : int
        Dimension of the measurement vector.
    num_particles : int, optional
        Number of particles (N). Default is 1000.
    """

    def __init__(self, dim_x: int, dim_z: int, num_particles: int = 1000) -> None:
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.num_particles = num_particles

        # Particles: shape (num_particles, dim_x)
        self.particles = np.zeros((num_particles, dim_x))
        # Weights: shape (num_particles,)
        self.weights = np.ones(num_particles) / num_particles

        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

    def initialize_particles(self, x_mean: np.ndarray, p_cov: np.ndarray) -> None:
        """
        Initialize particles from a multivariate Gaussian distribution.

        Parameters
        ----------
        x_mean : np.ndarray
            Mean initial state (dim_x,).
        p_cov : np.ndarray
            Initial covariance (dim_x, dim_x).
        """
        self.particles = np.random.multivariate_normal(x_mean, p_cov, self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def predict(
        self, dt: float, fx_func: Callable, q_mat: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        r"""
        Predict step (Proposal distribution).

        Parameters
        ----------
        dt : float
            Time step (s).
        fx_func : Callable
            State transition function $f(x, dt, **kwargs) \to x_{new}$.
        q_mat : np.ndarray, optional
            Process noise covariance (dim_x, dim_x). If None, uses `self.Q`.
        **kwargs : Any
            Additional arguments passed to transition function.
        """
        q_curr = q_mat if q_mat is not None else self.Q

        # Propagate each particle through the model and add noise
        for i in range(self.num_particles):
            self.particles[i] = fx_func(self.particles[i], dt, **kwargs)
            noise = np.random.multivariate_normal(np.zeros(self.dim_x), q_curr)
            self.particles[i] += noise

    def update(
        self, z: np.ndarray, hx_func: Callable, r_mat: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        r"""
        Update step (Weighting and Resampling).

        Parameters
        ----------
        z : np.ndarray
            Measurement vector (dim_z,).
        hx_func : Callable
            Measurement function $h(x, **kwargs) \to z_{pred}$.
        r_mat : np.ndarray, optional
            Measurement noise covariance (dim_z, dim_z). If None, uses `self.R`.
        **kwargs : Any
            Additional arguments passed to measurement function.
        """
        r_curr = r_mat if r_mat is not None else self.R

        # Update weights based on measurement likelihood
        inv_r = np.linalg.inv(r_curr)
        det_r = np.linalg.det(r_curr)
        norm_factor = 1.0 / np.sqrt((2 * np.pi) ** self.dim_z * det_r)

        for i in range(self.num_particles):
            zp = hx_func(self.particles[i], **kwargs)
            diff = z - zp
            # Multivariate Gaussian likelihood
            prob = norm_factor * np.exp(-0.5 * (diff.T @ inv_r @ diff))
            self.weights[i] *= prob

        # Normalize weights
        self.weights += 1e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)

        # Resample if effective number of particles is too low
        if self.neff() < self.num_particles / 2:
            self.resample()

    def resample(self) -> None:
        """Resample particles using Systematic Resampling."""
        cum_sum = np.cumsum(self.weights)
        cum_sum[-1] = 1.0  # Ensure last element is exactly 1

        # Systematic Resampling
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles
        indices = np.zeros(self.num_particles, dtype=int)

        i, j = 0, 0
        while i < self.num_particles:
            if positions[i] < cum_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def neff(self) -> float:
        """
        Calculate effective number of particles.

        Returns
        -------
        float
            Effective sample size.
        """
        return 1.0 / np.sum(np.square(self.weights))

    @property
    def x(self) -> np.ndarray:
        """Weighted mean state vector."""
        return np.average(self.particles, weights=self.weights, axis=0)

    @property
    def P(self) -> np.ndarray:
        """Weighted error covariance matrix."""
        x_mean = self.x
        diff = self.particles - x_mean
        return (self.weights * diff.T) @ diff
