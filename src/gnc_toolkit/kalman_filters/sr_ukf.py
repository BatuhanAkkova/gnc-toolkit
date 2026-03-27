"""
Square-Root Unscented Kalman Filter (SR-UKF) algorithm.
"""

import numpy as np
from typing import Callable, Any, Optional
from scipy.linalg import cholesky, qr, solve_triangular


class SRUKF:
    """
    Square-Root Unscented Kalman Filter (SR-UKF).

    Provides better numerical stability and efficiency than standard UKF
    by propagating the Cholesky factor (S) of the covariance matrix, where $P = S S^T$.

    Parameters
    ----------
    dim_x : int
        Dimension of the state vector.
    dim_z : int
        Dimension of the measurement vector.
    alpha : float, optional
        Primary scaling parameter. Default is 1e-3.
    beta : float, optional
        Secondary scaling parameter (optimal for Gaussians = 2). Default is 2.0.
    kappa : float, optional
        Tertiary scaling parameter. Default is 0.0.
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.lambda_ = alpha**2 * (dim_x + kappa) - dim_x
        self.gamma = np.sqrt(dim_x + self.lambda_)

        self.num_sigmas = 2 * dim_x + 1
        self.Wm = np.zeros(self.num_sigmas)
        self.Wc = np.zeros(self.num_sigmas)

        self.Wm[0] = self.lambda_ / (dim_x + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

        weight = 1.0 / (2 * (dim_x + self.lambda_))
        for i in range(1, self.num_sigmas):
            self.Wm[i] = weight
            self.Wc[i] = weight

        self.x = np.zeros(dim_x)
        self.S = np.eye(dim_x)  # Cholesky factor of covariance P

        # Square-root noise covariances
        self.Qs = np.eye(dim_x)
        self.Rs = np.eye(dim_z)

    def predict(
        self, dt: float, fx_func: Callable, qs_mat: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        r"""
        Predict step.

        Parameters
        ----------
        dt : float
            Time step (s).
        fx_func : Callable
            State transition function $f(x, dt, **kwargs) \to x_{new}$.
        qs_mat : np.ndarray, optional
            Square-root process noise covariance (S_Q). If None, uses `self.Qs`.
        **kwargs : Any
            Additional arguments passed to transition function.
        """
        qs_curr = qs_mat if qs_mat is not None else self.Qs

        # Generate sigma points using S
        sigmas = self._generate_sigma_points()

        # Propagate sigma points
        sigmas_f = np.zeros((self.num_sigmas, self.dim_x))
        for i in range(self.num_sigmas):
            sigmas_f[i] = fx_func(sigmas[i], dt, **kwargs)

        # Predicted mean
        self.x = self.Wm @ sigmas_f

        # Update square-root covariance S
        # Compute S- using QR decomposition
        x_pts = np.sqrt(self.Wc[1]) * (sigmas_f[1:] - self.x)

        # QR decomposition
        _, s_transpose = qr(np.vstack((x_pts, qs_curr)), mode="economic")
        self.S = s_transpose[: self.dim_x, : self.dim_x].T

        # Cholesky rank-1 update for the first weight
        dx_vec = sigmas_f[0] - self.x
        self.S = self._cholesky_update(self.S, dx_vec, self.Wc[0])

    def update(
        self, z: np.ndarray, hx_func: Callable, rs_mat: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        r"""
        Update step.

        Parameters
        ----------
        z : np.ndarray
            Measurement vector (dim_z,).
        hx_func : Callable
            Measurement function $h(x, **kwargs) \to z_{pred}$.
        rs_mat : np.ndarray, optional
            Square-root measurement noise covariance (S_R). If None, uses `self.Rs`.
        **kwargs : Any
            Additional arguments passed to measurement function.
        """
        rs_curr = rs_mat if rs_mat is not None else self.Rs

        # Regenerate sigma points
        sigmas_f = self._generate_sigma_points()

        # Transform to measurement space
        sigmas_h = np.zeros((self.num_sigmas, self.dim_z))
        for i in range(self.num_sigmas):
            sigmas_h[i] = hx_func(sigmas_f[i], **kwargs)

        # Mean measurement
        zp = self.Wm @ sigmas_h

        # Square-root innovation covariance Sy
        h_pts = np.sqrt(self.Wc[1]) * (sigmas_h[1:] - zp)
        _, s_transpose_y = qr(np.vstack((h_pts, rs_curr)), mode="economic")
        sy_mat = s_transpose_y[: self.dim_z, : self.dim_z].T

        # Rank-1 update for first weight
        dz_0 = sigmas_h[0] - zp
        sy_mat = self._cholesky_update(sy_mat, dz_0, self.Wc[0])

        # Cross-covariance Pxz
        pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(self.num_sigmas):
            dx_vec = sigmas_f[i] - self.x
            dz_vec = sigmas_h[i] - zp
            pxz += self.Wc[i] * np.outer(dx_vec, dz_vec)

        # Kalman gain
        k_gain = solve_triangular(
            sy_mat, solve_triangular(sy_mat, pxz.T, lower=True), lower=True, trans="T"
        ).T

        # Correct mean and square-root covariance
        self.x += k_gain @ (z - zp)

        u_mat = k_gain @ sy_mat
        for i in range(self.dim_z):
            self.S = self._cholesky_update(self.S, u_mat[:, i], -1.0)

    def _generate_sigma_points(self) -> np.ndarray:
        """Generate 2n+1 sigma points using the current Cholesky factor."""
        sigmas = np.zeros((self.num_sigmas, self.dim_x))
        sigmas[0] = self.x
        for i in range(self.dim_x):
            sigmas[i + 1] = self.x + self.gamma * self.S[:, i]
            sigmas[i + 1 + self.dim_x] = self.x - self.gamma * self.S[:, i]
        return sigmas

    def _cholesky_update(self, s_mat: np.ndarray, vec: np.ndarray, weight: float) -> np.ndarray:
        """
        Performs rank-1 Cholesky update or downdate.

        Parameters
        ----------
        s_mat : np.ndarray
            Current Cholesky factor (lower triangular).
        vec : np.ndarray
            Vector to update with.
        weight : float
            Scalar weight (positive for update, negative for downdate).

        Returns
        -------
        np.ndarray
            Updated lower-triangular Cholesky factor.
        """
        if weight > 0:
            v_scaled = np.sqrt(weight) * vec
            _, s_transpose = qr(np.vstack((s_mat.T, v_scaled)), mode="economic")
            return s_transpose[: s_mat.shape[0], : s_mat.shape[1]].T
        else:
            p_cov = s_mat @ s_mat.T + weight * np.outer(vec, vec)
            try:
                return cholesky(p_cov, lower=True)
            except np.linalg.LinAlgError:
                return cholesky(p_cov + np.eye(s_mat.shape[0]) * 1e-12, lower=True)

    @property
    def P(self) -> np.ndarray:
        """Full error covariance matrix P = S S^T."""
        return self.S @ self.S.T
