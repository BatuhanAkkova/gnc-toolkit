"""
Fuel slosh dynamics using equivalent pendulum models.
"""

import numpy as np

def fuel_slosh_dynamics(theta, theta_dot, omega, omega_dot, L, r_base, g_equiv):
    """
    Computes the second derivative of the slosh pendulum angle.
    
    Model: Pendulum of length L, base at r_base from spacecraft CM.
    g_equiv: Equivalent acceleration (e.g., from thrust).
    
    Args:
        theta (float): Pendulum angle [rad].
        theta_dot (float): Pendulum angular velocity [rad/s].
        omega (np.ndarray): Spacecraft angular velocity (3,).
        omega_dot (np.ndarray): Spacecraft angular acceleration (3,).
        L (float): Pendulum length [m].
        r_base (np.ndarray): Location of pendulum base from CM (3,).
        g_equiv (np.ndarray): Equivalent gravity/acceleration vector (3,).
    
    Returns:
        float: Pendulum angular acceleration theta_ddot [rad/s^2].
    """
    # Simple pendulum in a non-inertial frame
    # Accel at base: a_base = g_equiv - omega_dot x r_base - omega x (omega x r_base)
    a_base = g_equiv - np.cross(omega_dot, r_base) - np.cross(omega, np.cross(omega, r_base))
    
    # Projection of a_base onto the pendulum's transverse direction
    # Assume pendulum swings in the x-z plane of its local frame
    # Local unit vectors: 
    # e_r (radial) = [sin(theta), 0, -cos(theta)]
    # e_theta (transverse) = [cos(theta), 0, sin(theta)]
    e_theta = np.array([np.cos(theta), 0, np.sin(theta)])
    
    # theta_ddot = (a_base . e_theta) / L
    theta_ddot = np.dot(a_base, e_theta) / L
    
    return theta_ddot

def fuel_slosh_torque(m_p, L, theta, theta_dot, theta_ddot, r_base):
    """
    Computes the reaction torque from fuel slosh on the spacecraft body.
    
    Args:
        m_p (float): Pendulum mass [kg].
        L (float): Pendulum length [m].
        theta (float): Pendulum angle [rad].
        theta_dot (float): Pendulum angular velocity [rad/s].
        theta_ddot (float): Pendulum angular acceleration [rad/s^2].
        r_base (np.ndarray): Base location (3,).
        
    Returns:
        np.ndarray: Slosh reaction torque (3,).
    """    
    # Relative acceleration of bob:
    # a_rel = L * theta_ddot * e_theta - L * theta_dot^2 * e_r
    e_r = np.array([np.sin(theta), 0, -np.cos(theta)])
    e_theta = np.array([np.cos(theta), 0, np.sin(theta)])
    a_rel = L * theta_ddot * e_theta - L * (theta_dot**2) * e_r
    
    # Reaction force on body: -m_p * a_rel
    force_slosh = -m_p * a_rel
    
    # Torque on body: r_bob x force_slosh
    r_bob = r_base + L * e_r
    torque_slosh = np.cross(r_bob, force_slosh)
    
    return torque_slosh
