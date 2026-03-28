import numpy as np
import sys
import os

# Add src to path to ensure we can import opengnc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from opengnc.visualization import (
    plot_orbit_3d, 
    plot_attitude_sphere, 
    plot_ground_track, 
    plot_coverage_heatmap
)

def main():
    print("Generating demo visualization data...")

    # -------------------------------------------------------------------------
    # 1. 3D Orbit Data (Circular LEO)
    # -------------------------------------------------------------------------
    R_Earth = 6378137.0 # m
    h = 500000.0        # 500 km altitude
    r_mag = R_Earth + h
    theta = np.linspace(0, 2 * np.pi, 200)
    
    # 45 deg inclined orbit
    inc = np.deg2rad(45.0)
    x = r_mag * np.cos(theta)
    y = r_mag * np.sin(theta) * np.cos(inc)
    z = r_mag * np.sin(theta) * np.sin(inc)
    
    r_eci = np.vstack([x, y, z]).T
    fig_orbit = plot_orbit_3d(r_eci, title="3D Orbit (500 km, 45° Inclination)")

    # -------------------------------------------------------------------------
    # 2. Attitude Data (Cone Trace on Unit Sphere)
    # -------------------------------------------------------------------------
    angle = np.linspace(0, 4 * np.pi, 200) # 2 rotations
    # spiral pattern
    vx = 0.5 * np.cos(angle)
    vy = 0.5 * np.sin(angle)
    vz = np.sqrt(1 - vx**2 - vy**2) 
    
    vectors = np.vstack([vx, vy, vz]).T
    fig_attitude = plot_attitude_sphere(vectors, title="Attitude Vector Cone Trace")

    # -------------------------------------------------------------------------
    # 3. Ground Track Data (Dummy values wrapping across Earth)
    # -------------------------------------------------------------------------
    # Map LEO coords roughly back to lat/lon for simplicity
    lats = 45.0 * np.sin(theta) # matches orbital inclination amplitude
    lons = np.linspace(-180, 180, 200) # simple sweep
    times = np.linspace(0, 6000, 200)  # ~1.5 hours in seconds
    
    fig_gt = plot_ground_track(lats, lons, times=times, title="Simplified Ground Track")

    # -------------------------------------------------------------------------
    # 4. Coverage Heat Map Data (Full Grid with continuous values)
    # -------------------------------------------------------------------------
    lat_g = np.linspace(-90, 90, 20)
    lon_g = np.linspace(-180, 180, 40)
    LA, LO = np.meshgrid(lat_g, lon_g)
    
    # Coverage value based on some trigonometric signal (e.g. access availability)
    val = (np.sin(LA * np.pi / 180.0) * np.cos(LO * np.pi / 180.0) + 1.0) * 10.0
    
    fig_cov = plot_coverage_heatmap(LA.flatten(), LO.flatten(), val.flatten(), title="Global Coverage Grid")

    # Save PNG files for results
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
        
    print(f"Saving plots to {assets_dir}...")
    fig_orbit.write_image(os.path.join(assets_dir, "orbit_viz.png"))
    fig_attitude.write_image(os.path.join(assets_dir, "attitude_viz.png"))
    fig_gt.write_image(os.path.join(assets_dir, "ground_track_viz.png"))
    fig_cov.write_image(os.path.join(assets_dir, "coverage_viz.png"))

    print("Generation complete.")

if __name__ == "__main__":
    main()




