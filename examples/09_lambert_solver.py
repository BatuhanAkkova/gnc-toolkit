"""
Lambert Solver for Orbit Targeting
==================================

This example demonstrates how to solve the Lambert problem to find the 
velocity required for a spacecraft to travel between two points in 
a given time of flight.

Scenario:
    - Target a GEO position from a LEO position.
    - Time of flight: 5 hours.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from opengnc.guidance.rendezvous import solve_lambert

def run_example():
    mu = 398600.4418 # Earth [km^3/s^2]
    
    # 1. Define Positions (LEO to GEO)
    r_leo = np.array([7000.0, 0, 0])
    # GEO is 42164 km, let's put it on the Y axis
    r_geo = np.array([0, 42164.0, 0])
    
    # Time of flight: 5 hours
    tof = 5 * 3600.0
    
    # 2. Solve Lambert Problem
    print(f"Solving Lambert Problem...")
    print(f"Initial Pos: {r_leo} km")
    print(f"Target Pos:  {r_geo} km")
    print(f"Time of Flight: {tof/3600:.2f} hours")
    
    # tm=1 (delta_nu < 180 deg)
    v1, v2 = solve_lambert(r_leo, r_geo, tof, mu, tm=1)
    
    # 3. Results
    print("-" * 30)
    print(f"Departure Velocity: {v1} km/s (Mag: {np.linalg.norm(v1):.4f})")
    print(f"Arrival Velocity:   {v2} km/s (Mag: {np.linalg.norm(v2):.4f})")
    print("-" * 30)
    
    # Check if delta-V is reasonable for LEO to GEO
    v_leo = np.sqrt(mu / np.linalg.norm(r_leo))
    dv = np.linalg.norm(v1 - np.array([0, v_leo, 0]))
    print(f"Departure Delta-V (from circular): {dv:.4f} km/s")

    # 4. Visualization
    # Approximate trajectory by propagating v1
    # Simple Keplerian propagation for visualization
    dt_plot = tof / 100
    r_traj = [r_leo]
    r_curr = r_leo.copy()
    v_curr = v1.copy()
    
    for _ in range(100):
        r_mag = np.linalg.norm(r_curr)
        a = -mu / r_mag**3 * r_curr
        v_curr += a * dt_plot
        r_curr += v_curr * dt_plot
        r_traj.append(r_curr.copy())
    
    r_traj = np.array(r_traj)
    
    plt.figure(figsize=(8, 8))
    plt.plot(r_traj[:, 0], r_traj[:, 1], 'r', label='Lambert Trajectory')
    plt.plot(r_leo[0], r_leo[1], 'ro', label='Departure (LEO)')
    plt.plot(r_geo[0], r_geo[1], 'go', label='Arrival (GEO)')
    
    # Earth for scale
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(6378 * np.cos(theta), 6378 * np.sin(theta), 'b', alpha=0.3, label='Earth')
    
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('Lambert Transfer: LEO to GEO')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Save the plot to assets/
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets', 'lambert_transfer.png'))
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    run_example()




