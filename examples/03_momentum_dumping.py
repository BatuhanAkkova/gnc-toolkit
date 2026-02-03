"""
Momentum Dumping Example
========================

This script demonstrates Reaction Wheel (RW) momentum management using Magnetorquers (MTQ).

Scenario:
    - A spacecraft maintains a fixed inertial attitude (e.g., Zenith pointing at equator).
    - Constant environmental disturbance torque (e.g., Solar Pressure) causes RW momentum buildup.
    - When momentum exceeds a threshold, the MTQ system is activated to "dump" momentum.

Control Law:
    - Cross-Product Law: m = (k / |B|^2) * (h_error x B)
    - Generates torque T = m x B which opposes the perpendicular component of accumulated momentum.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from gnc_toolkit.environment.mag_field import tilted_dipole_field
from gnc_toolkit.classical_control.momentum_dumping import CrossProductLaw

def simulation():
    # -------------------------------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------------------------------
    
    # Simulation Settings
    dt = 10.0 # Step size [s]
    duration = 20000.0 # [s] ~3.5 Orbits
    
    # Environment
    # Disturbance Torque (Inertial Frame)
    # Constant secular torque causing buildup
    T_dist_inertial = np.array([0.0, 5.0e-5, 0.0]) # 50 uNm
    
    # Orbit (Polar, for variety of B-field)
    Re = 6371000.0
    h = 500000.0
    r_orbit = Re + h
    mu = 3.986e14
    n = np.sqrt(mu / r_orbit**3)
    
    # Reaction Wheel System
    h_rw = np.array([0.0, 0.0, 0.0]) # Initial Momentum [Nms]
    h_sat = 0.1 # Saturation limit [Nms]
    
    # Desaturation Controller (Cross-Product Law)
    # Law: m = (k / |B|^2) * (h_error x B)
    # This creates Torque T = m x B = -k * h_perp
    k_gain = 0.01 
    desat_ctrl = CrossProductLaw(gain=k_gain)
    
    dumping_active = False
    dump_threshold_on = 0.08 # Start dumping at 80% saturation
    dump_threshold_off = 0.01 # Stop at 10%
    
    # Magnetorquer
    max_dipole = 2.0 # Am^2 (Larger coil for dumping)
    
    # Data Recording
    history = {
        't': [],
        'h_rw_norm': [],
        'h_rw': [],
        'm_dipole': []
    }
    
    t = 0.0
    
    print(f"Starting Momentum Dumping Simulation. Disturbance: {T_dist_inertial} Nm")
    
    while t < duration:
        # A. Environment (Magnetic Field)
        # Position in Orbit
        theta = n * t
        r_eci = r_orbit * np.array([np.cos(theta), 0, np.sin(theta)])
        
        b_eci = tilted_dipole_field(r_eci)
        b_norm = np.linalg.norm(b_eci)
        
        # B. Momentum Management Logic
        h_norm = np.linalg.norm(h_rw)
        
        # Hysteresis Logic
        if h_norm > dump_threshold_on:
            dumping_active = True
        elif h_norm < dump_threshold_off:
            dumping_active = False
            
        m_cmd = np.zeros(3)
        T_mag = np.zeros(3)
        
        if dumping_active and b_norm > 1e-9:
            # Calculate desired dipole to oppose momentum using the CrossProductLaw class
            m_ideal = desat_ctrl.calculate_control(h_rw, b_eci)
            
            # Saturate Magnetorquer
            m_mag = np.linalg.norm(m_ideal)
            if m_mag > max_dipole:
                m_cmd = (m_ideal / m_mag) * max_dipole
            else:
                m_cmd = m_ideal
                
            # Resulting Magnetic Torque
            T_mag = np.cross(m_cmd, b_eci)
            
        # C. Dynamics (Momentum Budget)
        # h_dot = T_external + T_control_magnetics
        
        h_dot = T_dist_inertial + T_mag
        
        h_rw = h_rw + h_dot * dt
        
        # Recording
        history['t'].append(t)
        history['h_rw'].append(h_rw)
        history['h_rw_norm'].append(np.linalg.norm(h_rw))
        history['m_dipole'].append(m_cmd)
        
        t += dt
        
        if t % 2000 == 0:
            print(f"Time: {t:.0f}s | RW Momentum: {h_norm:.4f} Nms | Dumping: {dumping_active}")

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    t_arr = np.array(history['t'])
    h_arr = np.array(history['h_rw'])
    m_arr = np.array(history['m_dipole'])
    h_norm_arr = np.array(history['h_rw_norm'])
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # 1. Momentum Norm
    ax[0].plot(t_arr, h_norm_arr, 'k', linewidth=2, label='RW Momentum')
    ax[0].axhline(h_sat, color='r', linestyle='--', label='Saturation')
    ax[0].axhline(dump_threshold_on, color='orange', linestyle=':', label='Dump Start')
    ax[0].axhline(dump_threshold_off, color='g', linestyle=':', label='Dump Stop')
    ax[0].set_ylabel('Momentum [Nms]')
    ax[0].set_title('Reaction Wheel Momentum Management')
    ax[0].legend()
    ax[0].grid(True)
    
    # 2. Momentum Components
    ax[1].plot(t_arr, h_arr[:, 0], label='Hx')
    ax[1].plot(t_arr, h_arr[:, 1], label='Hy')
    ax[1].plot(t_arr, h_arr[:, 2], label='Hz')
    ax[1].set_ylabel('Components [Nms]')
    ax[1].grid(True)
    ax[1].legend()
    
    # 3. Magnetorquer Activity
    ax[2].plot(t_arr, m_arr[:, 0], label='Mx')
    ax[2].plot(t_arr, m_arr[:, 1], label='My')
    ax[2].plot(t_arr, m_arr[:, 2], label='Mz')
    ax[2].set_ylabel('Dipole [Am^2]')
    ax[2].set_xlabel('Time [s]')
    ax[2].grid(True)
    ax[2].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulation()
