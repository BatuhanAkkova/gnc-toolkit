import numpy as np
import pytest

from gnc_toolkit.guidance.formation_flying import (
    virtual_structure_control,
    leader_follower_control,
    fuel_balanced_formation_keeping,
    distributed_consensus_control
)

def test_virtual_structure_control():
    state_actual = np.array([[1.0, 2.0], [0.5, 1.5]])
    state_desired = np.array([[1.0, 2.0], [1.0, 2.0]])
    gains = np.array([2.0, 2.0])
    
    control = virtual_structure_control(state_actual, state_desired, gains)
    assert np.allclose(control[0], [0.0, 0.0])
    assert np.allclose(control[1], [1.0, 1.0])

def test_leader_follower_control():
    leader_state = np.array([10.0, 5.0])
    follower_state = np.array([8.0, 4.0])
    desired_relative_state = np.array([-1.0, -1.0])  # follower should be at [9.0, 4.0]
    gains = np.array([0.5, 0.5])
    
    control = leader_follower_control(leader_state, follower_state, desired_relative_state, gains)
    # Target = [9.0, 4.0]
    # Follower error = [8.0, 4.0] - [9.0, 4.0] = [-1.0, 0.0]
    # control = -0.5 * [-1.0, 0.0] = [0.5, 0.0]
    assert np.allclose(control, [0.5, 0.0])

def test_fuel_balanced_formation_keeping():
    states = np.array([[0.0, 0.0], [0.0, 0.0]])
    fuel_levels = np.array([75.0, 25.0]) # 75% and 25% of total 100
    weights = np.array([[10.0, 10.0], [10.0, 10.0]])
    
    control = fuel_balanced_formation_keeping(states, fuel_levels, weights)
    # Expected: 75% -> 0.75 * 10 = 7.5
    # Expected: 25% -> 0.25 * 10 = 2.5
    assert np.allclose(control[0], [7.5, 7.5])
    assert np.allclose(control[1], [2.5, 2.5])

def test_distributed_consensus_control():
    states = np.array([
        [1.0, 0.0],
        [3.0, 0.0],
        [2.0, 0.0]
    ])
    # A complete graph Laplacian for 3 nodes:
    laplacian = np.array([
        [2, -1, -1],
        [-1, 2, -1],
        [-1, -1, 2]
    ])
    gains = 1.0
    
    control = distributed_consensus_control(states, laplacian, gains)
    # Control = -1.0 * (L @ states)
    expected = np.array([
        [3.0, 0.0],
        [-3.0, 0.0],
        [0.0, 0.0]
    ])
    assert np.allclose(control, expected)
