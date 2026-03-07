import numpy as np

def fixed_interval_smoother(Xs_f, Ps_f, Fs, Qs, Zs, Hs, Rs):
    """
    Fixed-Interval Smoother (Two-Filter / Fraser-Potter approach).
    Combines a forward Kalman filter with a backward information filter.

    Args:
        Xs_f (list of np.ndarray): Forward filtered states x_{k|k}.
        Ps_f (list of np.ndarray): Forward filtered covariances P_{k|k}.
        Fs (list of np.ndarray): State transition matrices F_k (from k to k+1).
        Qs (list of np.ndarray): Process noise covariances Q_k.
        Zs (list of np.ndarray): Measurements z_k.
        Hs (list of np.ndarray): Measurement matrices H_k.
        Rs (list of np.ndarray): Measurement noise covariances R_k.

    Returns:
        X_smooth (np.ndarray): Smoothed states.
        P_smooth (np.ndarray): Smoothed covariances.
    """
    num_steps = len(Xs_f)
    dim_x = Xs_f[0].shape[0]

    # Backward Information Filter
    # y_b: information state (Y * x)
    # Y_b: information matrix (inv(P))
    y_b = np.zeros((num_steps, dim_x))
    Y_b = np.zeros((num_steps, dim_x, dim_x))
    
    # Backward pass
    for k in range(num_steps - 2, -1, -1):
        # Update with measurement at k+1
        H = Hs[k+1]
        R_inv = np.linalg.inv(Rs[k+1])
        
        # Information update
        Y_hat = Y_b[k+1] + H.T @ R_inv @ H
        y_hat = y_b[k+1] + H.T @ R_inv @ Zs[k+1]
        
        # Backward predict to k
        F = Fs[k]
        Q = Qs[k]
        
        # Backward prediction (Information form)
        # Y_k = F' * Y_hat * (I + Q * Y_hat)^-1 * F
        # y_k = F' * (I + Y_hat * Q)^-1 * y_hat
        
        I = np.eye(dim_x)
        tmp = np.linalg.inv(I + Y_hat @ Q)
        Y_b[k] = F.T @ Y_hat @ tmp @ F
        y_b[k] = F.T @ tmp @ y_hat

    # Combine Forward and Backward
    X_smooth = np.zeros((num_steps, dim_x))
    P_smooth = np.zeros((num_steps, dim_x, dim_x))

    for k in range(num_steps):
        P_f_inv = np.linalg.inv(Ps_f[k])
        P_smooth[k] = np.linalg.inv(P_f_inv + Y_b[k])
        X_smooth[k] = P_smooth[k] @ (P_f_inv @ Xs_f[k] + y_b[k])

    return X_smooth, P_smooth
