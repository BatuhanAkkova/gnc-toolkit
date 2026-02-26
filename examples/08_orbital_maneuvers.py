"""
Orbital Maneuvers: Hohmann Transfer and Plane Change
====================================================

This example demonstrates common orbital maneuvers used for mission planning.

Scenario:
    - Transfer from LEO (7000 km) to GEO (42164 km).
    - Perform an inclination change from a 28.5 deg launch to 0 deg equatorial.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from gnc_toolkit.guidance.maneuvers import hohmann_transfer, plane_change, combined_plane_change

def run_example():
    mu = 398600.4418 # Earth's Gravitational Parameter [km^3/s^2]
    
    # 1. Hohmann Transfer: LEO to GEO
    r_leo = 7000.0 # km
    r_geo = 42164.0 # km
    
    dv1, dv2, tof = hohmann_transfer(r_leo, r_geo, mu)
    
    print("--- Hohmann Transfer (LEO to GEO) ---")
    print(f"Initial Radius: {r_leo} km")
    print(f"Final Radius:   {r_geo} km")
    print(f"Burn 1 Delta-V: {dv1:.4f} km/s")
    print(f"Burn 2 Delta-V: {dv2:.4f} km/s")
    print(f"Total Delta-V:  {dv1 + dv2:.4f} km/s")
    print(f"Time of Flight: {tof/3600:.2f} hours")
    print()
    
    # 2. Simple Plane Change at GEO
    v_geo = np.sqrt(mu / r_geo) # Velocity at GEO
    inc_change = np.radians(28.5) # 28.5 deg to 0 deg
    
    dv_plane = plane_change(v_geo, inc_change)
    
    print("--- Simple Plane Change at GEO ---")
    print(f"Velocity at GEO: {v_geo:.4f} km/s")
    print(f"Inclination Change: 28.5 deg")
    print(f"Delta-V Required: {dv_plane:.4f} km/s")
    print()
    
    # 3. Combined Plane Change and Circularization
    # Often more efficient to do plane change at the second burn of Hohmann
    a_trans = (r_leo + r_geo) / 2.0
    v_trans_a = np.sqrt(mu * (2/r_geo - 1/a_trans)) # Velocity at apogee of transfer orbit
    v_final = np.sqrt(mu / r_geo) # Final circular velocity (at GEO)
    
    dv_combined = combined_plane_change(v_trans_a, v_final, inc_change)
    
    print("--- Combined Burn (Circularization + Plane Change) ---")
    print(f"Pure Circularization Burn: {dv2:.4f} km/s")
    print(f"Combined Burn Delta-V:     {dv_combined:.4f} km/s")
    print(f"Saving (vs Simple):        {(dv2 + dv_plane) - dv_combined:.4f} km/s")

if __name__ == "__main__":
    run_example()
