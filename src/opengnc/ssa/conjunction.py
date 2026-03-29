"""
Conjunction Analysis and Probability of Collision (Pc) computation.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import dblquad


def compute_pc_foster(
    r1: np.ndarray,
    v1: np.ndarray,
    cov1: np.ndarray,
    r2: np.ndarray,
    v2: np.ndarray,
    cov2: np.ndarray,
    hbr: float,
) -> float:
    r"""
    Compute Probability of Collision (Pc) via Foster's method.

    Projects the encounter into a 2D plane and integrates the PDF:
    $P_c = \iint_{HBR} \frac{1}{2\pi \sqrt{|\mathbf{C}|}} \exp\left(-\frac{1}{2} \mathbf{x}^T \mathbf{C}^{-1} \mathbf{x}\right) dA$

    Parameters
    ----------
    r1, v1 : np.ndarray
        ECI state of Object 1 at TCA (m, m/s).
    cov1 : np.ndarray
        $3 \times 3$ covariance of Object 1 ($m^2$).
    r2, v2 : np.ndarray
        ECI state of Object 2 at TCA (m, m/s).
    cov2 : np.ndarray
        $3 \times 3$ covariance of Object 2 ($m^2$).
    hbr : float
        Combined Hard Body Radius (m).

    Returns
    -------
    float
        Probability of collision $P_c \in [0, 1]$.
    """
    rv1, rv2 = np.asarray(r1), np.asarray(r2)
    vv1, vv2 = np.asarray(v1), np.asarray(v2)
    cv1, cv2 = np.asarray(cov1), np.asarray(cov2)

    r_rel = rv1 - rv2
    v_rel = vv1 - vv2
    v_mag = np.linalg.norm(v_rel)

    if v_mag < 1e-6:
        raise ValueError("Relative velocity is too small for encounter projection.")

    combined_cov = cv1 + cv2

    # Define Encounter Frame (Equinoctial/Collision Plane)
    z_hat = v_rel / v_mag

    # x-hat is along the projected relative position (impact parameter)
    if np.linalg.norm(r_rel) > 1e-4:
        x_hat = r_rel - np.dot(r_rel, z_hat) * z_hat
        x_mag = np.linalg.norm(x_hat)
        if x_mag > 1e-6:
            x_hat /= x_mag
        else:
            x_hat = np.array([1.0, 0.0, 0.0]) if abs(z_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_hat -= np.dot(x_hat, z_hat) * z_hat
            x_hat /= np.linalg.norm(x_hat)
    else:
        # Direct collision at TCA
        x_hat = np.array([1.0, 0.0, 0.0]) if abs(z_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_hat -= np.dot(x_hat, z_hat) * z_hat
        x_hat /= np.linalg.norm(x_hat)

    y_hat = np.cross(z_hat, x_hat)
    m_rot = np.vstack([x_hat, y_hat, z_hat])

    # Project to 2D
    r_enc = m_rot @ r_rel
    cov_enc = m_rot @ combined_cov @ m_rot.T
    cov_2d = cov_enc[:2, :2]
    det_c = np.linalg.det(cov_2d)

    if det_c < 1e-15:
        return 0.0

    inv_c = np.linalg.inv(cov_2d)
    x_c, y_c = r_enc[0], r_enc[1]

    def pdf_2d(x: float, y: float) -> float:
        d = np.array([x - x_c, y - y_c])
        arg = -0.5 * d.T @ inv_c @ d
        return float((1.0 / (2.0 * np.pi * np.sqrt(det_c))) * np.exp(arg))

    pc, _ = dblquad(
        pdf_2d,
        -hbr, hbr,
        lambda y: -np.sqrt(max(0, hbr**2 - y**2)),
        lambda y: np.sqrt(max(0, hbr**2 - y**2))
    )

    return float(pc)


def compute_pc_chan(
    r1: np.ndarray,
    v1: np.ndarray,
    cov1: np.ndarray,
    r2: np.ndarray,
    v2: np.ndarray,
    cov2: np.ndarray,
    hbr: float,
) -> float:
    r"""
    Probability of Collision (Pc) via Chan's Analytical Approximation.

    Provides a fast, series-based solution to the 2D Gaussian integral over 
    a circular region. Most accurate when the HBR is small compared to 
    the standard deviation of the covariance.

    Parameters
    ----------
    r1, r2 : np.ndarray
        ECI Position vectors at TCA (m).
    v1, v2 : np.ndarray
        ECI Velocity vectors at TCA (m/s).
    cov1, cov2 : np.ndarray
        $3\times 3$ error covariance matrices ($m^2$).
    hbr : float
        Combined Hard Body Radius (m).

    Returns
    -------
    float
        Computed probability of collision.
    """
    rv1, rv2 = np.asarray(r1), np.asarray(r2)
    vv1, vv2 = np.asarray(v1), np.asarray(v2)
    cv1, cv2 = np.asarray(cov1), np.asarray(cov2)

    r_rel = rv1 - rv2
    v_rel = vv1 - vv2
    v_mag = np.linalg.norm(v_rel)

    if v_mag < 1e-6:
        raise ValueError("Relative velocity is too small.")

    # Frame projection (same as Foster)
    z_hat = v_rel / v_mag
    x_hat = np.array([1.0, 0.0, 0.0]) if abs(z_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_hat -= np.dot(x_hat, z_hat) * z_hat
    x_hat /= np.linalg.norm(x_hat)
    y_hat = np.cross(z_hat, x_hat)
    m_rot = np.vstack([x_hat, y_hat, z_hat])

    r_enc = m_rot @ r_rel
    cov_2d = (m_rot @ (cv1 + cv2) @ m_rot.T)[:2, :2]

    # Diagonalize covariance
    vals, vecs = np.linalg.eigh(cov_2d)
    if np.any(vals <= 0):
        return 0.0

    # Principal components
    r_p = vecs.T @ r_enc[:2]
    sig_x, sig_y = np.sqrt(vals[0]), np.sqrt(vals[1])

    u = (r_p[0]**2 / vals[0]) + (r_p[1]**2 / vals[1])
    v = hbr**2 / (sig_x * sig_y)

    # Chan series: Pc = exp(-u/2) * sum(...)
    pc = 0.0
    term_u = np.exp(-u / 2.0)
    term_v = np.exp(-v / 2.0)
    sum_v = term_v

    for n in range(50):
        inc = term_u * (1.0 - sum_v)
        pc += inc
        if inc < 1e-15:
            break
        term_u *= (u / 2.0) / (n + 1)
        term_v *= (v / 2.0) / (n + 1)
        sum_v += term_v

    return float(pc)




