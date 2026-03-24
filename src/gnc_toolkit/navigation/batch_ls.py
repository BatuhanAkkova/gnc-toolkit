"""
Differential Correction using Batch Least Squares for Orbit Determination.
"""

import numpy as np

from gnc_toolkit.disturbances.gravity import TwoBodyGravity


class BatchLeastSquaresOD:
    """
    Differential Correction using Batch Least Squares for Orbit Determination.
    """

    def __init__(self, x_guess, mu=398600.4415e9):
        self.x = np.array(x_guess, dtype=float)
        self.mu = mu
        self.gravity = TwoBodyGravity(mu=mu)

    def _propagate(self, x0, t_start, t_end):
        """Simple RK4 propagation for mapping state to observation times."""

        def f(state):
            r = state[:3]
            v = state[3:]
            a = self.gravity.get_acceleration(r)
            return np.concatenate([v, a])

        dt = t_end - t_start
        if abs(dt) < 1e-8:
            return x0

        h = dt
        k1 = f(x0)
        k2 = f(x0 + 0.5 * h * k1)
        k3 = f(x0 + 0.5 * h * k2)
        k4 = f(x0 + h * k3)
        return x0 + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def solve(self, observations, times, max_iter=15):
        """
        Solve OD using Batch Least Squares.
        """
        for i in range(max_iter):
            A = []
            b = []

            for z, t in zip(observations, times):
                x_t = self._propagate(self.x, 0, t)
                z_hat = x_t[:3]
                residual = z - z_hat

                r = x_t[:3]
                r_mag = np.linalg.norm(r)
                G = (self.mu / r_mag**3) * (3.0 * np.outer(r, r) / r_mag**2 - np.eye(3))

                Phi = np.eye(6)
                Phi[:3, 3:] = np.eye(3) * t
                Phi[3:, :3] = G * t
                Phi[3:, 3:] = np.eye(3) + G * (t**2 / 2.0)

                H = np.zeros((3, 6))
                H[:, :3] = np.eye(3)
                H_scaled = H @ Phi

                A.append(H_scaled)
                b.append(residual)

            A = np.vstack(A)
            b = np.concatenate(b)

            # Normal equations
            try:
                dx = np.linalg.solve(A.T @ A, A.T @ b)
            except np.linalg.LinAlgError:
                dx = np.linalg.pinv(A.T @ A) @ (A.T @ b)

            self.x += dx

            if np.linalg.norm(dx) < 1e-6:
                break

        return self.x
