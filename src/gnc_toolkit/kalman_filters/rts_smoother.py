"""
Rauch-Tung-Striebel (RTS) Smoother for linear systems.
"""

import numpy as np


def rts_smoother(Xs, Ps, Fs, Qs):
    """
    Rauch-Tung-Striebel (RTS) Smoother.
    Performs a backward pass over Kalman filter results to provide optimal
    estimates given all data in a fixed interval.

    Args:
        Xs (list of np.ndarray): List of predicted/updated states from forward pass.
                                 Length N. Xs[k] is x_{k|k}.
        Ps (list of np.ndarray): List of predicted/updated covariances from forward pass.
                                 Length N. Ps[k] is P_{k|k}.
        Fs (list of np.ndarray): List of state transition matrices.
                                 Length N-1. Fs[k] is F from k to k+1.
        Qs (list of np.ndarray): List of process noise covariances.
                                 Length N-1. Qs[k] is Q from k to k+1.

    Returns
    -------
        X_smooth (np.ndarray): Smoothed states, (N, dim_x).
        P_smooth (np.ndarray): Smoothed covariances, (N, dim_x, dim_x).
    """
    num_steps = len(Xs)
    dim_x = Xs[0].shape[0]

    X_smooth = np.zeros((num_steps, dim_x))
    P_smooth = np.zeros((num_steps, dim_x, dim_x))

    # Initialize with the last filtered state
    X_smooth[-1] = Xs[-1]
    P_smooth[-1] = Ps[-1]

    # Backward pass
    for k in range(num_steps - 2, -1, -1):
        # Forward prediction
        P_pred = np.dot(np.dot(Fs[k], Ps[k]), Fs[k].T) + Qs[k]

        # Smoother gain
        C = np.dot(np.dot(Ps[k], Fs[k].T), np.linalg.inv(P_pred))

        # Smoothed state
        X_pred = np.dot(Fs[k], Xs[k])
        X_smooth[k] = Xs[k] + np.dot(C, X_smooth[k + 1] - X_pred)

        # Smoothed covariance
        P_smooth[k] = Ps[k] + np.dot(np.dot(C, P_smooth[k + 1] - P_pred), C.T)

    return X_smooth, P_smooth
