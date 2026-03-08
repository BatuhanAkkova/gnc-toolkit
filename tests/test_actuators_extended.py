import pytest
import numpy as np
from gnc_toolkit.actuators.reaction_wheel import ReactionWheel
from gnc_toolkit.actuators.cmg import ControlMomentGyro
from gnc_toolkit.actuators.vscmg import VariableSpeedCMG
from gnc_toolkit.actuators.solar_sail import SolarSail
from gnc_toolkit.actuators.allocation import PseudoInverseAllocator, SingularRobustAllocator, NullMotionManager

def test_rw_friction():
    # RW with friction
    rw = ReactionWheel(max_torque=1.0, static_friction=0.1, viscous_friction=0.01)
    
    # Test static friction at zero speed
    # Command < static friction should result in 0
    assert rw.command(0.05, current_speed=0.0) == 0.0
    # Command > static friction
    assert rw.command(0.5, current_speed=0.0) == 0.4
    
    # Test viscous friction at speed
    # Command 0.5 at speed 10 -> Friction = 0.01 * 10 = 0.1. Result = 0.5 - 0.1 = 0.4
    assert rw.command(0.5, current_speed=10.0) == 0.4

def test_cmg_torque():
    # CMG: h along Z, gimbal along X -> Torque should be along Y
    cmg = ControlMomentGyro(wheel_momentum=10.0, gimbal_axis=[1, 0, 0], spin_axis_init=[0, 0, 1])
    
    # gimbal rate = 0.1 rad/s
    # T = g_rate * h * (g x s) = 0.1 * 10 * ([1,0,0] x [0,0,1]) = 1.0 * [0, -1, 0]
    torque = cmg.command(0.1)
    np.testing.assert_allclose(torque, [0, -1.0, 0], atol=1e-7)

def test_solar_sail_force():
    sail = SolarSail(area=100.0, reflectivity=1.0, specular_reflect_coeff=1.0) # Perfect specular reflection
    
    # Sun along X, Normal along X
    sun_vec = np.array([1, 0, 0])
    normal = np.array([1, 0, 0])
    
    # F = P * A * cos(theta) * ( (1-rho)*u + 2*rho*cos(theta)*n )
    # cos_theta = 1, rho = 1 -> F = P * A * (0*u + 2*1*n) = 2 * P * A * n
    # P = 4.56e-6
    force = sail.calculate_force(sun_vec, normal)
    expected_f = 2 * 4.56e-6 * 100.0 * np.array([1, 0, 0])
    np.testing.assert_allclose(force, expected_f, atol=1e-10)

def test_pseudo_inverse_allocation():
    # 4 RWs in pyramid configuration
    # Angle beta = 54.7 deg
    beta = np.deg2rad(54.744)
    c, s = np.cos(beta), np.sin(beta)
    
    # Rows: Tx, Ty, Tz
    A = np.array([
        [s, 0, -s, 0],
        [0, s, 0, -s],
        [c, c, c, c]
    ])
    
    allocator = PseudoInverseAllocator(A)
    
    desired_torque = np.array([0.1, 0.0, 0.0])
    u = allocator.allocate(desired_torque)
    
    # Verify A * u == desired
    np.testing.assert_allclose(A @ u, desired_torque, atol=1e-7)

def test_null_motion():
    A = np.array([[1, 1]]) # Two actuators for 1-DOF
    manager = NullMotionManager(A)
    
    u_base = np.array([0.5, 0.5]) # Producer 1.0 total
    z = np.array([1.0, -1.0]) # Desired shift
    
    u_net = manager.apply_null_command(u_base, z)
    
    # Total output should still be 1.0
    assert np.isclose(np.sum(u_net), 1.0)
    # The actuators should have shifted
    assert u_net[0] > 0.5
    assert u_net[1] < 0.5

if __name__ == "__main__":
    try:
        test_rw_friction()
        print("RW Friction: PASSED")
        test_cmg_torque()
        print("CMG Torque: PASSED")
        test_solar_sail_force()
        print("Solar Sail Force: PASSED")
        test_pseudo_inverse_allocation()
        print("Pseudo-Inverse Allocation: PASSED")
        test_null_motion()
        print("Null Motion: PASSED")
        print("\nAll tests passed successfully!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)
