"""
Object Tracking and Orbit Correlation Utilities.
"""

import numpy as np
from scipy.linalg import inv


def compute_mahalanobis_distance(
    x1: np.ndarray, x2: np.ndarray, cov1: np.ndarray, cov2: np.ndarray
) -> float:
    """
    Computes the Mahalanobis distance between two state vectors.
    State vectors usually contain Position and Velocity (6 elements).

    Args:
        x1 (np.ndarray): State 1 (e.g., shape (6,))
        x2 (np.ndarray): State 2 (e.g., shape (6,))
        cov1 (np.ndarray): Covariance matrix of State 1 (6, 6)
        cov2 (np.ndarray): Covariance matrix of State 2 (6, 6)

    Returns
    -------
        float: Mahalanobis distance.
    """
    dx = x1 - x2
    cov_comb = cov1 + cov2

    try:
        dist = np.sqrt(dx.T @ inv(cov_comb) @ dx)
    except Exception:
        # Fallback if covariance is singular
        dist = np.linalg.norm(dx)

    return dist


def correlate_tracks(
    x1: np.ndarray, x2: np.ndarray, cov1: np.ndarray, cov2: np.ndarray, threshold: float = 3.0
) -> bool:
    """
    Correlates two state vectors to determine if they represent the same object.

    Args:
        x1 (np.ndarray): State 1 (e.g., shape (6,))
        x2 (np.ndarray): State 2 (e.g., shape (6,))
        cov1 (np.ndarray): Covariance 1 (6, 6)
        cov2 (np.ndarray): Covariance 2 (6, 6)
        threshold (float): Threshold for Mahalanobis distance (standard is 3.0 for 3-sigma).

    Returns
    -------
        bool: True if correlated, False otherwise.
    """
    dist = compute_mahalanobis_distance(x1, x2, cov1, cov2)
    return dist <= threshold
