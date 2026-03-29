"""
Parity Space methods for Fault Detection and Isolation (FDI).
"""

from __future__ import annotations

import numpy as np


class ParitySpaceDetector:
    r"""
    Fault Detection and Isolation (FDI) via Parity Space.

    Measurement Model:
    $\mathbf{y} = \mathbf{M}\mathbf{x} + \mathbf{v} + \mathbf{f}$

    The Parity Matrix $\mathbf{P}$ satisfies $\mathbf{P}\mathbf{M} = \mathbf{0}$.
    The Parity Vector is $\mathbf{p} = \mathbf{P}\mathbf{y}$.

    Parameters
    ----------
    M : np.ndarray
        Geometry matrix $(p \times n)$, $p > n$.
    """

    def __init__(self, M: np.ndarray) -> None:
        """Initialize FDI detector using SVD to find the null space of M."""
        self.M = np.asarray(M)
        self.p_dim, self.n_dim = self.M.shape

        if self.p_dim <= self.n_dim:
            raise ValueError("Parity space requires redundant measurements (p > n)")

        # P is the left null space of M (from U matrix of SVD)
        u, _, _ = np.linalg.svd(self.M)
        self.P = u[:, self.n_dim:].T  # Shape: (p-n, p)

    def get_parity_vector(self, y: np.ndarray) -> np.ndarray:
        r"""
        Compute the parity vector $\mathbf{p} = \mathbf{P} \mathbf{y}$.

        Parameters
        ----------
        y : np.ndarray
            Measurement vector $(p, 1)$ or $(p,)$.

        Returns
        -------
        np.ndarray
            Parity vector of shape $(p-n,)$.
        """
        y_vec = np.asarray(y).flatten()
        return np.asarray(self.P @ y_vec)

    def detect_fault(self, y: np.ndarray, threshold: float) -> bool:
        r"""
        Detect fault by checking if $\|\mathbf{p}\| > \epsilon$.

        Parameters
        ----------
        y : np.ndarray
            Measurement vector.
        threshold : float
            Fault detection threshold.

        Returns
        -------
        bool
            True if a fault is detected.
        """
        p_vec = self.get_parity_vector(y)
        return bool(np.linalg.norm(p_vec) > threshold)

    def isolate_fault(self, y: np.ndarray) -> int:
        r"""
        Isolate the faulty sensor by identifying the column of $\mathbf{P}$ 
        most aligned with the parity vector $\mathbf{p}$.

        Parameters
        ----------
        y : np.ndarray
            Measurement vector.

        Returns
        -------
        int
            Index of the faulty sensor (0 to $p-1$), or -1 if no fault.
        """
        p_vec = self.get_parity_vector(y)
        p_mag = np.linalg.norm(p_vec)

        if p_mag < 1e-12:
            return -1

        # Normalize parity vector for directional comparison
        p_u = p_vec / p_mag

        # Column alignment: argmax(|p_u^T * P_col_u|)
        alignments = []
        for i in range(self.p_dim):
            p_col = self.P[:, i]
            col_mag = np.linalg.norm(p_col)
            if col_mag > 1e-12:
                alignments.append(np.abs(np.dot(p_u, p_col / col_mag)))
            else:
                alignments.append(0.0)

        return int(np.argmax(alignments))




