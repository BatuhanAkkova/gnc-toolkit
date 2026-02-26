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
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from gnc_toolkit.guidance.rendezvous import solve_lambert

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

if __name__ == "__main__":
    run_example()
