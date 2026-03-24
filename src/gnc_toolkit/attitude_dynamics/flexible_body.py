"""
Flexible body dynamics propagation and coupling models.
"""

import numpy as np


def flexible_body_dynamics(eta, eta_dot, omega_dot, natural_freqs, damping_ratios, modal_influence):
    """
    Computes the second derivative of modal coordinates for flexible body dynamics.

    The model assumes:
    eta_ddot + 2 * zeta * omega_n * eta_dot + omega_n^2 * eta = Phi^T * omega_dot

    Args:
        eta (np.ndarray): Modal displacements (n_modes,).
        eta_dot (np.ndarray): Modal velocities (n_modes,).
        omega_dot (np.ndarray): Rigid body angular acceleration (3,).
        natural_freqs (np.ndarray): Natural frequencies [rad/s] (n_modes,).
        damping_ratios (np.ndarray): Damping ratios (n_modes,).
        modal_influence (np.ndarray): Modal influence matrix (n_modes, 3).
                                     Maps rigid body acceleration to modal forces.

    Returns
    -------
        np.ndarray: Modal accelerations eta_ddot (n_modes,).
    """
    # Force/Torque coupling term
    coupling_term = modal_influence @ omega_dot

    # Damping term: 2 * zeta * omega_n * eta_dot
    damping_term = 2 * damping_ratios * natural_freqs * eta_dot

    # Stiffness term: omega_n^2 * eta
    stiffness_term = (natural_freqs**2) * eta

    # Solve for eta_ddot
    eta_ddot = coupling_term - damping_term - stiffness_term

    return eta_ddot


def coupled_flexible_rigid_dynamics(
    J_rigid, omega, torque, eta, eta_dot, natural_freqs, damping_ratios, modal_influence
):
    """
    Computes coupled rigid-flexible body dynamics.

    System Equations:
    [ J_rigid  -Phi^T ] [ omega_dot ] + [ omega x (J_rigid * omega) ] = [ torque ]
    [ -Phi      I     ] [ eta_ddot  ]   [ 2*zeta*omega_n*eta_dot + omega_n^2*eta ] = [ 0 ]

    Args:
        J_rigid (np.ndarray): Rigid body inertia tensor (3, 3).
        omega (np.ndarray): Angular velocity (3,).
        torque (np.ndarray): External torque (3,).
        eta (np.ndarray): Modal displacements (n_modes,).
        eta_dot (np.ndarray): Modal velocities (n_modes,).
        natural_freqs (np.ndarray): (n_modes,).
        damping_ratios (np.ndarray): (n_modes,).
        modal_influence (np.ndarray): (n_modes, 3).

    Returns
    -------
        omega_dot (np.ndarray): (3,).
        eta_ddot (np.ndarray): (n_modes,).
    """
    n_modes = len(natural_freqs)

    # Build augmented mass matrix
    # [ J_rigid         Phi^T ]
    # [  Phi             I    ]
    M = np.zeros((3 + n_modes, 3 + n_modes))
    M[:3, :3] = J_rigid
    M[:3, 3:] = modal_influence.T
    M[3:, :3] = modal_influence
    M[3:, 3:] = np.eye(n_modes)

    # Build RHS
    # [ torque - omega x (J_rigid * omega) ]
    # [ -(2 * zeta * omega_n * eta_dot + omega_n^2 * eta) ]
    rhs = np.zeros(3 + n_modes)
    rhs[:3] = torque - np.cross(omega, J_rigid @ omega)
    rhs[3:] = -(2 * damping_ratios * natural_freqs * eta_dot + (natural_freqs**2) * eta)

    # Solve for accelerations
    sol = np.linalg.solve(M, rhs)

    return sol[:3], sol[3:]
