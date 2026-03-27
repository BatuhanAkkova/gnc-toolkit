"""
Orbital Maneuvers: Hohmann Transfer and Plane Change
====================================================

This example demonstrates common orbital maneuvers used for mission planning.

Scenario:
    - Transfer from LEO (7000 km) to GEO (42164 km).
    - Perform an inclination change from a 28.5 deg launch to 0 deg equatorial.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from opengnc.guidance.maneuvers import hohmann_transfer, plane_change, combined_plane_change

def run_example():
    mu = 398600.4418 # Earth's Gravitational Parameter [km^3/s^2]
    
    # Hohmann Transfer: LEO to GEO
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
    
    # Simple Plane Change at GEO
    v_geo = np.sqrt(mu / r_geo) # Velocity at GEO
    inc_change = np.radians(28.5) # 28.5 deg to 0 deg
    
    dv_plane = plane_change(v_geo, inc_change)
    
    print("--- Simple Plane Change at GEO ---")
    print(f"Velocity at GEO: {v_geo:.4f} km/s")
    print(f"Inclination Change: 28.5 deg")
    print(f"Delta-V Required: {dv_plane:.4f} km/s")
    print()
    
    # Combined Plane Change and Circularization
    a_trans = (r_leo + r_geo) / 2.0
    v_trans_a = np.sqrt(mu * (2/r_geo - 1/a_trans)) # Velocity at apogee of transfer orbit
    v_final = np.sqrt(mu / r_geo) # Final circular velocity (at GEO)
    
    dv_combined = combined_plane_change(v_trans_a, v_final, inc_change)
    
    print("--- Combined Burn (Circularization + Plane Change) ---")
    print(f"Pure Circularization Burn: {dv2:.4f} km/s")
    print(f"Combined Burn Delta-V:     {dv_combined:.4f} km/s")
    print(f"Saving (vs Simple):        {(dv2 + dv_plane) - dv_combined:.4f} km/s")

    # Visualization
    theta = np.linspace(0, 2*np.pi, 200)
    
    # Orbits
    x_leo = r_leo * np.cos(theta)
    y_leo = r_leo * np.sin(theta)
    x_geo = r_geo * np.cos(theta)
    y_geo = r_geo * np.sin(theta)
    
    # Transfer Ellipse (Apogee=r_geo, Perigee=r_leo)
    a_trans = (r_leo + r_geo) / 2.0
    e_trans = (r_geo - r_leo) / (r_geo + r_leo)
    r_trans = a_trans * (1 - e_trans**2) / (1 + e_trans * np.cos(theta))
    x_trans = r_trans * np.cos(theta)
    y_trans = r_trans * np.sin(theta)
    
    plt.figure(figsize=(8, 8))
    plt.plot(x_leo, y_leo, 'b--', label='Initial LEO')
    plt.plot(x_geo, y_geo, 'g--', label='Target GEO')
    # Filter transfer orbit to show only half (perigee at 0 to apogee at pi)
    plt.plot(x_trans[theta <= np.pi], y_trans[theta <= np.pi], 'r', linewidth=2, label='Hohmann Transfer')
    
    plt.plot(r_leo, 0, 'ro', label='Burn 1')
    plt.plot(-r_geo, 0, 'go', label='Burn 2')
    plt.plot(0, 0, 'yo', markersize=10, label='Earth')
    
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.title('Hohmann Transfer: LEO to GEO')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Save the plot to assets/
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets', 'hohmann_transfer.png'))
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    run_example()




