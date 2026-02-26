"""
Deterministic Attitude Determination with QUEST Algorithm
==========================================================

This example demonstrates how to determine the attitude of a spacecraft
using the QUEST (Quaternion Estimator) algorithm from two or more vector 
measurements.

Scenario:
    - Sensors: Magnetometer and Sun Sensor providing body-frame vectors.
    - Reference: IGRF Magnetic Field and Analytical Sun Position in ECI.
    - Algorithm: QUEST for optimal quaternion (Inertial -> Body).
"""

import numpy as np
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from gnc_toolkit.attitude_determination.quest import quest
from gnc_toolkit.utils.quat_utils import quat_rot, quat_conj, quat_to_euler
from gnc_toolkit.sensors.magnetometer import Magnetometer
from gnc_toolkit.sensors.sun_sensor import SunSensor

def run_example():
    # 1. Truth Attitude (Random)
    # Roll=10, Pitch=20, Yaw=30 deg
    euler_true = np.radians([10, 20, 30])
    # For simplicity, let's just define a rotation matrix or quaternion
    from gnc_toolkit.utils.state_conversion import euler_to_quat
    q_true = euler_to_quat(euler_true, seq="321")
    
    # 2. Reference Vectors (Inertial Frame)
    # e.g., Sun and Magnetic field
    v_sun_eci = np.array([1.0, 0.0, 0.0])
    v_mag_eci = np.array([0.0, 0.0, 1.0])
    
    ref_vectors = [v_sun_eci, v_mag_eci]
    
    # 3. Simulate Measurements (Body Frame)
    mag_sensor = Magnetometer(noise_std=0.001)
    sun_sensor = SunSensor(noise_std=0.001)
    
    q_inv = quat_conj(q_true) # ECI -> Body
    
    v_sun_body_true = quat_rot(q_inv, v_sun_eci)
    v_mag_body_true = quat_rot(q_inv, v_mag_eci)
    
    v_sun_meas = sun_sensor.measure(v_sun_body_true)
    v_mag_meas = mag_sensor.measure(v_mag_body_true)
    
    body_vectors = [v_sun_meas, v_mag_meas]
    
    # Weights (Star tracker / Sun sensor usually more weighted than Mag)
    weights = [0.8, 0.2]
    
    # 4. Run QUEST
    print("Running QUEST Algorithm...")
    q_est = quest(body_vectors, ref_vectors, weights=weights)
    
    # 5. Results
    euler_est = quat_to_euler(q_est, seq="321")
    
    print("-" * 30)
    print(f"True Euler [deg]:  {np.degrees(euler_true)}")
    print(f"Est. Euler [deg]:  {np.degrees(euler_est)}")
    print("-" * 30)
    
    error_angle = np.degrees(np.linalg.norm(euler_true - euler_est))
    print(f"Total Angular Error: {error_angle:.4f} degrees")

if __name__ == "__main__":
    run_example()
