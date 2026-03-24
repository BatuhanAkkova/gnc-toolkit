"""
H-Infinity Robust Controller state-feedback design.
"""

import numpy as np
from scipy.linalg import solve_continuous_are


class HInfinityController:
    """
    H-Infinity Robust Controller.

    Designs a state-feedback controller u = -K*x that minimizes the H-infinity norm
    of the closed-loop transfer function from disturbances to error signals.

    System model:
    x_dot = A*x + B1*w + B2*u
    z = C1*x + D11*w + D12*u (performance output)

    Simplified case (standard state-feedback):
    Minimizes J = integral(x.T*Q*x + u.T*R*u - gamma^2 * w.T*w) dt
    """

    def __init__(self, A, B1, B2, Q, R, gamma):
        """
        Initialize H-Infinity Controller.

        Args:
            A (np.ndarray): State matrix (nx x nx)
            B1 (np.ndarray): Disturbance input matrix (nx x nw)
            B2 (np.ndarray): Control input matrix (nx x nu)
            Q (np.ndarray): State cost matrix (nx x nx)
            R (np.ndarray): Control cost matrix (nu x nu)
            gamma (float): Target attenuation level
        """
        self.A = np.array(A)
        self.B1 = np.array(B1)
        self.B2 = np.array(B2)
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.gamma = gamma
        self.P = None
        self.K = None

    def solve(self):
        """
        Solve the H-infinity Riccati Equation:
        P*A + A.T*P + P*(B1*B1.T/gamma^2 - B2*inv(R)*B2.T)*P + Q = 0
        """
        # Modified B matrix for LQR solver logic:
        # B_eff * B_eff.T = B2*inv(R)*B2.T - B1*B1.T/gamma^2

        # Standard form for solve_continuous_are:
        # A.T*P + P*A - P * (B*R^-1*B.T - B1*B1.T/gamma^2) * P + Q = 0
        # Let S = B2*inv(R)*B2.T - (1/gamma^2)*B1*B1.T

        S = self.B2 @ np.linalg.inv(self.R) @ self.B2.T - (1.0 / self.gamma**2) * (
            self.B1 @ self.B1.T
        )

        # Eigen decomposition of S to handle potential non-definiteness (though it should be controllable)
        vals, vecs = np.linalg.eigh(S)
        # Note: If any vals are negative, it means control effort is "weaker" than disturbance capacity.
        # solve_continuous_are can handle this if the Hamiltonian matrix has no eigenvalues on the imaginary axis.

        try:
            R_inv_eff = S
            self.P = solve_continuous_are(
                self.A, np.eye(self.A.shape[0]), self.Q, np.linalg.inv(R_inv_eff)
            )
            return self.P
        except Exception as e:
            raise ValueError(
                f"H-infinity solution does not exist for gamma={self.gamma}. Try increasing gamma. Error: {e}"
            )

    def compute_gain(self):
        """
        Compute feedback gain K = inv(R) * B2.T * P
        """
        if self.P is None:
            self.solve()

        self.K = np.linalg.solve(self.R, self.B2.T @ self.P)
        return self.K

    def compute_control(self, x):
        """
        Compute control input u = -K * x
        """
        if self.K is None:
            self.compute_gain()
        return -self.K @ x
