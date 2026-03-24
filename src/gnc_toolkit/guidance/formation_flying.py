"""
Formation Flying Control routines (Virtual Structure, Leader-Follower,
Fuel-Balanced, and Distributed Consensus).
"""

import numpy as np


def virtual_structure_control(
    state_actual: np.ndarray, state_desired: np.ndarray, gains: np.ndarray
) -> np.ndarray:
    """
    Computes control inputs to track a virtual structure.

    Args:
        state_actual (np.ndarray): Current state of the spacecraft (N x state_dim).
        state_desired (np.ndarray): Desired state derived from the virtual structure (N x state_dim).
        gains (np.ndarray): Proportional/Derivative feedback gains.

    Returns
    -------
        np.ndarray: Computed control actions for each spacecraft (N x control_dim).
    """
    state_actual = np.asarray(state_actual)
    state_desired = np.asarray(state_desired)
    # Simple PD style tracking or proportional feedback to desired state
    # u = -K * (x - x_d)
    return -gains * (state_actual - state_desired)


def leader_follower_control(
    leader_state: np.ndarray,
    follower_state: np.ndarray,
    desired_relative_state: np.ndarray,
    gains: np.ndarray,
) -> np.ndarray:
    """
    Computes the control command for a follower spacecraft to maintain a desired
    relative state with respect to a leader spacecraft.

    Args:
        leader_state (np.ndarray): State of the leader spacecraft.
        follower_state (np.ndarray): State of the follower spacecraft.
        desired_relative_state (np.ndarray): Target offset from the leader (follower - leader).
        gains (np.ndarray): Control gains.

    Returns
    -------
        np.ndarray: Control command for the follower.
    """
    # Follower wants to be at: leader_state + desired_relative_state
    target_state = leader_state + desired_relative_state
    # u_follower = -K * (follower_state - target_state)
    return -gains * (follower_state - target_state)


def fuel_balanced_formation_keeping(
    states: np.ndarray, fuel_levels: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """
    Distributes control effort among the formation to balance fuel consumption.
    Spacecraft with more fuel provide more control effort.

    Args:
        states (np.ndarray): (N x M) array of states for N spacecraft.
        fuel_levels (np.ndarray): (N,) array of current fuel mass/levels.
        weights (np.ndarray): (N, M) base control actions or distribution weights.

    Returns
    -------
        np.ndarray: Fuel-balanced control actions.
    """
    states = np.asarray(states)
    fuel_levels = np.asarray(fuel_levels)
    weights = np.asarray(weights)

    # Normalize fuel levels
    total_fuel = np.sum(fuel_levels)
    if total_fuel < 1e-9:
        return np.zeros_like(weights)

    fuel_ratios = fuel_levels / total_fuel

    # Scale control weights by available fuel fraction
    # Expand fuel_ratios slightly to broadcast against (N, M) weights
    scaled_weights = weights * fuel_ratios[:, np.newaxis]
    return scaled_weights


def distributed_consensus_control(
    states: np.ndarray, laplacian_matrix: np.ndarray, gains: np.ndarray
) -> np.ndarray:
    """
    Computes consensus control for a distributed multi-agent system based on
    the network Laplacian matrix.

    u_i = -K * SUM(a_ij * (x_i - x_j))
    In matrix form: U = -K * (L @ X)

    Args:
        states (np.ndarray): N x M array of states for N agents.
        laplacian_matrix (np.ndarray): N x N graph Laplacian matrix.
        gains (np.ndarray): Gains to apply to the consensus error.

    Returns
    -------
        np.ndarray: N x M array of control inputs.
    """
    states = np.asarray(states)
    laplacian_matrix = np.asarray(laplacian_matrix)

    # Laplacian L = D - A, where D is degree matrix and A is adjacency
    # The term (L @ X) computes SUM(a_ij * (x_i - x_j)) for all i
    consensus_error = laplacian_matrix @ states
    return -gains * consensus_error
