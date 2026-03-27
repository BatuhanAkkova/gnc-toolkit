"""
Residual generation for fault detection using observers.
"""


import numpy as np


class ObserverResidualGenerator:
    r"""
    Fault Residual Generation via Luenberger Observer.

    Implements a linear observer to estimate system outputs and generate 
    innovation-based residuals.
    Observer Dynamics: $\hat{\mathbf{x}}_{k+1} = \mathbf{A} \hat{\mathbf{x}}_k + \mathbf{B} \mathbf{u}_k + \mathbf{L} (\mathbf{y}_k - \hat{\mathbf{y}}_k)$
    Residual: $\mathbf{r}_k = \mathbf{y}_k - \mathbf{C} \hat{\mathbf{x}}_k$.

    Parameters
    ----------
    A : np.ndarray
        State transition matrix $(n, n)$.
    B : np.ndarray
        Input matrix $(n, m)$.
    C : np.ndarray
        Output matrix $(p, n)$.
    D : np.ndarray, optional
        Feedthrough matrix $(p, m)$. Defaults to zero.
    L : np.ndarray
        Observer gain matrix $(n, p)$.
    x0 : np.ndarray, optional
        Initial state estimate.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray | None = None,
        L: np.ndarray = None,
        x0: np.ndarray | None = None,
    ) -> None:
        """Initialize observer state and matrices."""
        self.A, self.B, self.C = np.asarray(A), np.asarray(B), np.asarray(C)
        self.D = np.asarray(D) if D is not None else np.zeros((self.C.shape[0], self.B.shape[1]))
        self.L = np.asarray(L)

        self.n, self.m, self.p = self.A.shape[0], self.B.shape[1], self.C.shape[0]
        self.x_hat = np.asarray(x0) if x0 is not None else np.zeros(self.n)

    def step(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""
        Advance observer and return residual $\mathbf{r}_k$.

        Parameters
        ----------
        u : np.ndarray
            Input vector $\mathbf{u}_k$.
        y : np.ndarray
            Measurement vector $\mathbf{y}_k$.

        Returns
        -------
        np.ndarray
            Residual $\mathbf{r}_k \in \mathbb{R}^p$.
        """
        uv, yv = np.asarray(u), np.asarray(y)

        # Output estimate
        y_hat = self.C @ self.x_hat + self.D @ uv
        resid = yv - y_hat

        # Update estimate
        self.x_hat = self.A @ self.x_hat + self.B @ uv + self.L @ resid

        return resid


class AnalyticalRedundancy:
    """Utilities for Analytical Redundancy-based Fault Detection."""

    @staticmethod
    def check_threshold(r: np.ndarray, threshold: float) -> bool:
        r"""
        Threshold check for a residual vector.

        Parameters
        ----------
        r : np.ndarray
            Residual vector.
        threshold : float
            Fault limit.

        Returns
        -------
        bool
            True if $\|\mathbf{r}\| > \text{threshold}$.
        """
        return np.linalg.norm(r) > threshold

    @staticmethod
    def gyro_vs_quaternion_residual(
        q_dot_measured: np.ndarray,
        q_dot_calculated: np.ndarray
    ) -> np.ndarray:
        r"""
        Residual between attitude rate kinematics and sensor differentiator.

        Calculates $\mathbf{r} = \dot{\mathbf{q}}_m - \dot{\mathbf{q}}_c$, 
        where $\dot{\mathbf{q}}_c = \frac{1}{2} \mathbf{q} \otimes \mathbf{\omega}$.

        Parameters
        ----------
        q_dot_measured : np.ndarray
            Measured quaternion derivative.
        q_dot_calculated : np.ndarray
            Calculated derivative from gyros.

        Returns
        -------
        np.ndarray
            Attitude residual vector.
        """
        return np.asarray(q_dot_measured) - np.asarray(q_dot_calculated)




