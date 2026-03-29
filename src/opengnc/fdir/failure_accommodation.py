"""
Actuator accommodation and control allocation weight adjustment.
"""


import numpy as np


class ActuatorAccommodation:
    r"""
    Actuator Fault Accommodation and Weighted Control Allocation.

    Redistributes control efforts across healthy actuators in response to 
    partial or total failures.
    Allocation Logic: $\mathbf{u} = \mathbf{W}^{-1} \mathbf{B}^T (\mathbf{B} \mathbf{W}^{-1} \mathbf{B}^T)^{-1} \tau$.

    Parameters
    ----------
    B : np.ndarray
        Control allocation (geometry) matrix of shape $(k, m)$.
    initial_weights : np.ndarray, optional
        Weight matrix $\mathbf{W}$ (relative cost of using each actuator). 
        Can be $(m, m)$ or diagonal $(m,)$. Defaults to identity.
    """

    def __init__(self, B: np.ndarray, initial_weights: np.ndarray | None = None) -> None:
        """Initialize allocation matrix and baseline weights."""
        self.B = np.asarray(B)
        self.k, self.m = self.B.shape

        if initial_weights is None:
            self.W_diag: np.ndarray = np.ones(self.m)
        else:
            w_arr = np.asarray(initial_weights, dtype=float)
            self.W_diag = np.asarray(np.diag(w_arr) if w_arr.ndim == 2 else w_arr, dtype=float)

        self.health = np.ones(self.m)  # 1.0 = Healthy, 0.0 = Dead

    def set_health(self, index: int, status: float) -> None:
        """
        Update the health status of a specific actuator.

        Parameters
        ----------
        index : int
            Actuator index.
        status : float
            Health factor in range [0, 1].
        """
        if not (0 <= index < self.m):
            raise IndexError("Actuator index out of range")
        self.health[index] = np.clip(status, 0.0, 1.0)

    def update_allocation_matrix(self) -> np.ndarray:
        r"""
        Compute the optimal weighted pseudo-inverse based on health.

        Returns
        -------
        np.ndarray
            $m\times k$ allocation matrix.
        """
        # effective_cost = weight / health
        eff_weights = self.W_diag.copy()
        for i in range(self.m):
            if self.health[i] <= 1e-6:
                eff_weights[i] = 1e12  # Infinite cost
            else:
                eff_weights[i] /= self.health[i]

        w_inv_diag = 1.0 / eff_weights
        w_inv = np.diag(w_inv_diag)

        # B_pinv = W_inv * B^T * (B * W_inv * B^T)^-1
        try:
            gram_mat = self.B @ w_inv @ self.B.T
            return np.asarray(w_inv @ self.B.T @ np.linalg.inv(gram_mat))
        except np.linalg.LinAlgError:
            return np.asarray(np.linalg.pinv(self.B))

    def allocate(self, tau: np.ndarray) -> np.ndarray:
        """
        Map desired torque/effort to actuator commands.

        Parameters
        ----------
        tau : np.ndarray
            Desired control effort $(k,)$.

        Returns
        -------
        np.ndarray
            Actuator command vector $(m,)$.
        """
        tau_vec = np.asarray(tau).flatten()
        b_pinv = self.update_allocation_matrix()
        return np.asarray(b_pinv @ tau_vec)




