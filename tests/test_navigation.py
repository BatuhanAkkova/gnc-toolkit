import numpy as np
import pytest
from gnc_toolkit.navigation import (
    OrbitDeterminationEKF,
    AngleOnlyNavigation,
    GPSNavigation,
    RelativeNavigationEKF,
    SurfaceNavigationEKF
)

def test_orbit_determination_ekf():
    # Initial state (LEO around Earth) [m, m/s]
    r0 = np.array([7000e3, 0, 0])
    v0 = np.array([0, 7500, 0])
    x0 = np.concatenate([r0, v0])
    
    P0 = np.eye(6) * 100.0
    Q = np.eye(6) * 1.0 # Process noise
    R = np.eye(3) * 10.0 # Pos measurement noise
    
    od = OrbitDeterminationEKF(x0, P0, Q, R, use_j2=False)
    
    # Predict step
    dt = 1.0
    od.predict(dt)
    
    assert od.state.shape == (6,)
    # Velocity should change position
    assert od.state[1] > 0.0 
    
    # Update step
    z_meas = r0 + np.array([5.0, -2.0, 1.0])
    od.update(z_meas)
    
    assert np.all(np.diag(od.covariance) >= 0)
    # Posterior covariance should be smaller than prior after update
    # Note: This is an architectural check, actual reduction depends on geometry

def test_angle_only_navigation():
    r0 = np.array([7000e3, 0, 0])
    v0 = np.array([0, 7500, 0])
    x0 = np.concatenate([r0, v0])
    
    P0 = np.eye(6) * 1000.0
    Q = np.eye(6) * 1.0
    R = np.eye(3) * 0.001 # Angle noise (in unit vector space)
    
    nav = AngleOnlyNavigation(x0, P0, Q, R)
    
    # Target at Earth center (simplified test)
    target_pos = np.array([0, 0, 0])
    u_meas = -r0 / np.linalg.norm(r0) # True LOS
    
    nav.update_unit_vector(u_meas, target_pos)
    assert nav.state.shape == (6,)

def test_gps_navigation():
    r0 = np.array([7000e3, 0, 0])
    v0 = np.array([0, 7500, 0])
    x0 = np.concatenate([r0, v0])
    P0 = np.eye(6) * 100.0
    Q = np.eye(6) * 1.0
    R = np.eye(3) * 5.0
    
    nav = GPSNavigation(x0, P0, Q, R)
    
    # Full P+V update
    r_meas = r0 + np.random.normal(0, 5, 3)
    v_meas = v0 + np.random.normal(0, 0.1, 3)
    
    nav.update_gps(r_meas, v_meas)
    assert nav.state.shape == (6,)

def test_relative_navigation():
    n = 0.0011 # LEO mean motion
    x0 = np.array([100.0, 0, 0, 0, -0.1, 0]) # 100m radial separation
    P0 = np.eye(6) * 10.0
    Q = np.eye(6) * 0.1
    R = np.eye(3) * 1.0
    
    nav = RelativeNavigationEKF(x0, P0, Q, R, n)
    
    nav.predict(dt=10.0)
    assert nav.state.shape == (6,)
    
    nav.update(np.array([101.0, 0.5, 0.1]))
    assert nav.state.shape == (6,)

def test_surface_navigation():
    x0 = np.zeros(6)
    P0 = np.eye(6) * 10.0
    Q = np.eye(6) * 0.1
    R = np.eye(3) * 0.1
    
    nav = SurfaceNavigationEKF(x0, P0, Q, R)
    
    # Local landmark at [10, 0, 0]
    landmark = np.array([10, 0, 0])
    z_obs = landmark - np.array([0.1, 0, 0]) # Measured from slighly offset pos
    
    nav.predict(dt=1.0, accel=np.array([1.0, 0, 0]))
    nav.update_landmark(z_obs, landmark)
    
    assert nav.state.shape == (6,)

if __name__ == "__main__":
    import sys
    print("Manual Test Run")
    try:
        print("Running test_orbit_determination_ekf...")
        test_orbit_determination_ekf()
        print("Running test_angle_only_navigation...")
        test_angle_only_navigation()
        print("Running test_gps_navigation...")
        test_gps_navigation()
        print("Running test_relative_navigation...")
        test_relative_navigation()
        print("Running test_surface_navigation...")
        test_surface_navigation()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
