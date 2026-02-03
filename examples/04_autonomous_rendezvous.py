"""
Autonomous Rendezvous Example
=============================

This script demonstrates relative guidance and control for a spacecraft rendezvous mission
using the Clohessy-Wiltshire (CW) / Hill's Equations.

Scenario:
    - Target Orbit: Geostationary (GEO).
    - Chaser Initial State: 10 km behind target (Along-Track), 100m radial offset.
    - Guidance:
        1. Multi-impulse approach to "Hold Point 1" (-1 km Along-Track).
        2. Station keeping at Hold Point 1.
        3. Final Approach to "Hold Point 2" (-100 m Along-Track).

Dynamics:
    - Linearized CW Equations (Relative Motion).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from gnc_toolkit.guidance.rendezvous import cw_equations, cw_targeting

def simulation():
    # -------------------------------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------------------------------
    
    # Orbit (GEO)
    mu = 3.986004418e5 # km^3/s^2
    Re = 6378.137 # km
    r_geo = 42164.0 # km
    n = np.sqrt(mu / r_geo**3) # Mean motion [rad/s]
    period = 2 * np.pi / n
    
    print(f"GEO Period: {period/3600:.2f} hours")
    
    # Initial Relative State [x, y, z] (Hill Frame)
    # x: Radial (out), y: Along-Track (velocity), z: Cross-Track (momentum)
    r0 = np.array([0.0, -10.0, 0.5]) # km. 500m cross track error too.
    v0 = np.array([0.0, 0.0, 0.0]) # km/s. Co-moving.
    
    current_r = r0.copy()
    current_v = v0.copy()
    current_t = 0.0
    
    # Mission Plan
    # List of Maneuvers: (Time relative to start, Target Position, Transfer Duration)
    # If Target is None, it means "Hold/Co-move" or just propagate.
    
    # Waypoints:
    # 1. T=0: Burn to reach [-1, 0, 0] km in 2 hours.
    # 2. T=2h (at -1km): Braking burn to match velocity (0).
    # 3. T=2.5h (Hold for 30m): Burn to reach [-0.1, 0, 0] km in 1 hour.
    # 4. T=3.5h (at -100m): Braking burn.
    
    trajectory_r = []
    trajectory_v = []
    trajectory_t = []
    
    maneuvers = [] # Record delta-vs
    
    # Simulation Step (for plotting resolution)
    dt_sim = 60.0 # 1 minute steps
    
    # --- PHASE 1: INITIAL APPROACH (-10km -> -1km) ---
    print("\n--- Phase 1: Initiation ---")
    t_transfer = 2.0 * 3600 # 2 hours
    r_target_1 = np.array([0.0, -1.0, 0.0]) # 1 km behind, 0 radial/cross
    
    # 1. Calculate Maneuver 1 (Depart)
    v_req_1 = cw_targeting(current_r, r_target_1, t_transfer, n)
    dv_1 = v_req_1 - current_v
    print(f"Burn 1 (Start): dV = {np.linalg.norm(dv_1)*1000:.3f} m/s")
    maneuvers.append((current_t, dv_1))
    
    # Execute Burn with small execution error (e.g., 2 cm/s)
    dv_error = np.array([1e-5, -2e-5, 5e-6]) 
    current_v = v_req_1 + dv_error
    
    # Propagate Phase 1
    steps = int(t_transfer / dt_sim)
    for _ in range(steps):
        trajectory_t.append(current_t)
        trajectory_r.append(current_r)
        trajectory_v.append(current_v)
        
        # Propagate 1 dt using analytic CW
        current_r, current_v = cw_equations(current_r, current_v, n, dt_sim)
        current_t += dt_sim
        
    print(f"End Phase 1 Position: {current_r}")
    
    # --- PHASE 2: POSITION CORRECTION & STATION KEEPING ---
    print("\n--- Phase 2: Position Correction & Hold ---")
    
    # 1. Check Position Error
    pos_err = current_r - r_target_1
    print(f"Arrival Position Error: {np.linalg.norm(pos_err)*1000:.3f} m")
    
    # 2. Correction Maneuver (Target r_target_1 in 10 minutes)
    t_corr = 600.0 # 10 mins
    v_req_corr = cw_targeting(current_r, r_target_1, t_corr, n)
    dv_corr_start = v_req_corr - current_v
    print(f"Correction Burn (Start): dV = {np.linalg.norm(dv_corr_start)*1000:.3f} m/s")
    maneuvers.append((current_t, dv_corr_start))
    current_v = v_req_corr
    
    # Propagate Correction (10 mins)
    steps_corr = int(t_corr / dt_sim)
    for _ in range(steps_corr):
        trajectory_t.append(current_t)
        trajectory_r.append(current_r)
        trajectory_v.append(current_v)
        current_r, current_v = cw_equations(current_r, current_v, n, dt_sim)
        current_t += dt_sim
        
    # Arrival at exact target -> Stop
    dv_corr_stop = np.array([0.0, 0.0, 0.0]) - current_v
    print(f"Correction Burn (Stop): dV = {np.linalg.norm(dv_corr_stop)*1000:.3f} m/s")
    maneuvers.append((current_t, dv_corr_stop))
    current_v = np.array([0.0, 0.0, 0.0])
    
    # 3. Station Keeping (Hold for 30 minutes)
    print(f"Holding at Point 1 for 30 mins...")
    t_hold = 1800.0 # 30 mins
    steps_hold = int(t_hold / dt_sim)
    for _ in range(steps_hold):
        trajectory_t.append(current_t)
        trajectory_r.append(current_r)
        trajectory_v.append(current_v)
        current_r, current_v = cw_equations(current_r, current_v, n, dt_sim)
        current_t += dt_sim
        
    # --- PHASE 3: FINAL APPROACH (-1km -> -100m) ---
    print("\n--- Phase 3: Final Approach to -100m ---")
    t_transfer_2 = 1.0 * 3600 # 1 hour
    r_target_2 = np.array([0.0, -0.1, 0.0]) # 100m behind
    
    # Burn 3
    v_req_3 = cw_targeting(current_r, r_target_2, t_transfer_2, n)
    dv_3 = v_req_3 - current_v
    print(f"Burn 3 (Approach): dV = {np.linalg.norm(dv_3)*1000:.3f} m/s")
    maneuvers.append((current_t, dv_3))
    
    current_v = v_req_3
    
    # Propagate Phase 3
    steps = int(t_transfer_2 / dt_sim)
    for _ in range(steps):
        trajectory_t.append(current_t)
        trajectory_r.append(current_r)
        trajectory_v.append(current_v)
        
        current_r, current_v = cw_equations(current_r, current_v, n, dt_sim)
        current_t += dt_sim
        
    # --- PHASE 4: FINAL STOP ---
    print("\n--- Phase 4: Arrival at -100m ---")
    dv_4 = np.array([0.0, 0.0, 0.0]) - current_v
    print(f"Burn 4 (Stop): dV = {np.linalg.norm(dv_4)*1000:.3f} m/s")
    maneuvers.append((current_t, dv_4))
    
    current_v = np.array([0.0, 0.0, 0.0])
    
    # Record Final Point
    trajectory_t.append(current_t)
    trajectory_r.append(current_r)
    trajectory_v.append(current_v)
    
    # -------------------------------------------------------------------------
    # 2. Visualization
    # -------------------------------------------------------------------------
    traj_r = np.array(trajectory_r)
    traj_t = np.array(trajectory_t) / 3600.0 # Hours
    
    # 2D In-Plane Plot (Radial vs Along-Track)
    # X_hill = Radial (Up/Down)
    # Y_hill = Along-Track (East/West)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Trajectory
    ax.plot(traj_r[:, 1], traj_r[:, 0], label='Chaser Trajectory')
    ax.plot(0, 0, 'ko', markersize=10, label='Target (GEO)')
    
    # Plot Waypoints
    ax.plot(r0[1], r0[0], 'rs', label='Start (-10km)')
    ax.plot(r_target_1[1], r_target_1[0], 'bs', label='Hold 1 (-1km)')
    ax.plot(r_target_2[1], r_target_2[0], 'gs', label='Hold 2 (-100m)')
    
    ax.set_xlabel('Along-Track [km] (Negative is Behind)')
    ax.set_ylabel('Radial [km] (Positive is Above)')
    ax.set_title('Autonomous Rendezvous (Hill Frame)')
    ax.grid(True)
    ax.legend()
    ax.axis('equal') # Important for geometry
    
    # Inset for Cross Track? Or seperate plot
    # Let's verify Z deviation zeroed out
    
    plt.figure()
    plt.plot(traj_t, traj_r[:, 2])
    plt.xlabel('Time [hr]')
    plt.ylabel('Cross-Track [km]')
    plt.title('Cross-Track Motion')
    plt.grid(True)
    
    print(f"Total Delta-V: {sum([np.linalg.norm(m[1]) for m in maneuvers])*1000:.2f} m/s")
    
    plt.show()

if __name__ == "__main__":
    simulation()
