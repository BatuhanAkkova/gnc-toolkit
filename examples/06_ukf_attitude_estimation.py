"""
Attitude Estimation with Unscented Kalman Filter (UKF)
======================================================

This example demonstrates how to use the UKF for satellite attitude estimation
by fusing a star tracker and a gyroscope.

Scenario:
    - 3-axis gyro (rate measurements with bias and noise).
    - Star tracker (quaternion measurements).
    - UKF handles the quaternion state on a manifold using tangent space errors.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from gnc_toolkit.kalman_filters.ukf import UKF_Attitude
from gnc_toolkit.sensors.gyroscope import Gyroscope
from gnc_toolkit.sensors.star_tracker import StarTracker
from gnc_toolkit.utils.quat_utils import quat_mult, quat_normalize, axis_angle_to_quat, quat_rot, quat_conj

def run_example():
    # 1. Configuration
    dt = 0.1
    t_max = 100.0
    time = np.arange(0, t_max, dt)
    
    # 2. Sensors
    gyro = Gyroscope(bias=np.array([0.01, -0.01, 0.005]), noise_std=0.001)
    st = StarTracker(noise_std=0.0001) # Very accurate
    
    # 3. Filter Initialization
    ukf = UKF_Attitude(alpha=1e-3)
    ukf.P *= 0.1
    ukf.Q = np.eye(6) * 1e-6
    ukf.R = np.eye(3) * 1e-8 # Star tracker is very accurate
    
    # Truth state
    q_true = np.array([0, 0, 0, 1.0])
    omega_true = np.array([0.05, 0.02, -0.01]) # Constant rate
    
    # 4. Simulation Loop
    results_t = []
    results_q_err = []
    results_bias_err = []
    
    # Custom process model for UKF
    def fx(x, dt, omega_meas):
        q = x[:4]
        bias = x[4:]
        omega = omega_meas - bias
        
        # Simple Euler integration for quaternion
        dq = axis_angle_to_quat(omega * dt)
        q_new = quat_normalize(quat_mult(q, dq))
        
        return np.concatenate([q_new, bias])
        
    def hx(x):
        # ST measures quaternion directly, but we only "see" it as vectors usually.
        # Here we'll simulate a vector measurement for the update.
        # Let's assume ST measures 3 orthogonal vectors in inertial frame.
        v_ref = np.eye(3)
        q_conj = quat_conj(x[:4])
        v_meas = []
        for v in v_ref:
            v_meas.append(quat_rot(q_conj, v))
        return np.array(v_meas).flatten()

    print("Running UKF Attitude Estimation...")
    for t in time:
        # Step Truth
        dq_true = axis_angle_to_quat(omega_true * dt)
        q_true = quat_normalize(quat_mult(q_true, dq_true))
        
        # Measurements
        omega_meas = gyro.measure(omega_true)
        
        # Update ST at 1Hz
        st_available = (int(t/dt) % int(1.0/dt) == 0)
        
        # Predict
        ukf.predict(dt, fx, omega_meas=omega_meas)
        
        # Update
        if st_available:
            q_conj_true = quat_conj(q_true)
            v_meas = []
            for v in np.eye(3):
                v_meas.append(st.measure(quat_rot(q_conj_true, v)))
            z = np.array(v_meas).flatten()
            ukf.update(z, hx)
            
        # Logging
        # Quaternion error (angle)
        q_err = quat_mult(quat_conj(ukf.x[:4]), q_true)
        angle_err = 2 * np.linalg.norm(q_err[:3]) * 180/np.pi
        
        bias_err = np.linalg.norm(ukf.x[4:] - gyro.bias)
        
        results_t.append(t)
        results_q_err.append(angle_err)
        results_bias_err.append(bias_err)

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(results_t, results_q_err)
    plt.ylabel('Attitude Error [deg]')
    plt.title('UKF Attitude Estimation Performance')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(results_t, results_bias_err)
    plt.ylabel('Bias Error [rad/s]')
    plt.xlabel('Time [s]')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_example()
