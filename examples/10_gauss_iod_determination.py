import numpy as np
from opengnc.navigation.iod import gauss_iod
from opengnc.utils.state_to_elements import kepler2eci, eci2kepler
import datetime

def main():
    print("--- Gauss Initial Orbit Determination (IOD) Example ---")
    
    # 1. Define a ground truth orbit (e.g., ISS-like LEO)
    mu = 398600.4415e9
    a = 6738e3       # Semi-major axis (m)
    ecc = 0.001      # Eccentricity
    incl = np.radians(51.64)  # Inclination (rad)
    raan = np.radians(45.0)   # RAAN (rad)
    argp = np.radians(30.0)   # Argument of perigee (rad)
    nu0 = np.radians(0.0)     # Initial true anomaly (rad)

    # 2. Generate observations at three time steps
    dt = 300.0  # 5 minutes between observations
    t1, t2, t3 = 0.0, dt, 2.0 * dt
    
    # Calculate true states at each time step
    # (Using simple keplerian propagation for the example)
    mean_motion = np.sqrt(mu / a**3)
    nu1 = nu0
    nu2 = nu0 + mean_motion * t2
    nu3 = nu0 + mean_motion * t3
    
    r1_true, v1_true = kepler2eci(a, ecc, incl, raan, argp, nu1)
    r2_true, v2_true = kepler2eci(a, ecc, incl, raan, argp, nu2)
    r3_true, v3_true = kepler2eci(a, ecc, incl, raan, argp, nu3)

    # 3. Simulate observer sitting at Equator (R1=R2=R3 for simplicity)
    # In a real scenario, this would include diurnal or orbital propagation
    Re = 6378e3
    R1 = np.array([Re, 0.0, 0.0])
    R2 = np.array([Re, 0.0, 0.0])
    R3 = np.array([Re, 0.0, 0.0])

    # 4. Calculate unit Line-of-Sight (LOS) vectors (rho_hat)
    rho1_vec = r1_true - R1
    rho2_vec = r2_true - R2
    rho3_vec = r3_true - R3
    
    rho_hat1 = rho1_vec / np.linalg.norm(rho1_vec)
    rho_hat2 = rho2_vec / np.linalg.norm(rho2_vec)
    rho_hat3 = rho3_vec / np.linalg.norm(rho3_vec)

    # 5. Perform Gauss IOD
    print(f"Propagating {dt/60:.1f} minute arcs...")
    state_est = gauss_iod(rho_hat1, rho_hat2, rho_hat3, t1, t2, t3, R1, R2, R3, mu=mu)
    
    r2_est = state_est[:3]
    v2_est = state_est[3:]

    # 6. Compare with ground truth at t2
    pos_error = np.linalg.norm(r2_est - r2_true)
    vel_error = np.linalg.norm(v2_est - v2_true)

    print("\nResults at Epoch t2:")
    print(f"True Position:  {r2_true / 1e3} km")
    print(f"Est Position:   {r2_est / 1e3} km")
    print(f"Position Error: {pos_error:.2f} m")
    
    print(f"\nTrue Velocity:  {v2_true / 1e3} km/s")
    print(f"Est Velocity:   {v2_est / 1e3} km/s")
    print(f"Velocity Error: {vel_error:.4f} m/s")

    # 7. Convert estimated state back to Keplerian elements
    a_est, ecc_est, incl_est, raan_est, argp_est, nu_est = eci2kepler(r2_est, v2_est)[:6]
    
    print("\nEstimated Orbital Elements:")
    print(f"Semi-major axis: {a_est/1e3:.2f} km")
    print(f"Eccentricity:    {ecc_est:.6f}")
    print(f"Inclination:     {np.degrees(incl_est):.2f} deg")

    # 8. Plotting Results
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Earth
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x_e = Re * np.cos(u) * np.sin(v)
        y_e = Re * np.sin(u) * np.sin(v)
        z_e = Re * np.cos(v)
        ax.plot_wireframe(x_e, y_e, z_e, color="blue", alpha=0.1)

        # Generate orbit trajectory for visualization (one period)
        period = 2 * np.pi * np.sqrt(a**3 / mu)
        time_vec = np.linspace(0, period, 200)
        orbit_pts = []
        for t in time_vec:
            nu_t = nu0 + mean_motion * t
            r_t, _ = kepler2eci(a, ecc, incl, raan, argp, nu_t)
            orbit_pts.append(r_t)
        orbit_pts = np.array(orbit_pts)

        ax.plot(orbit_pts[:, 0], orbit_pts[:, 1], orbit_pts[:, 2], 'k--', label="True Orbit", alpha=0.5)
        
        # Generate estimated orbit trajectory
        mean_motion_est = np.sqrt(mu / a_est**3)
        est_pts = []
        for t in time_vec:
            nu_t = nu_est + mean_motion_est * (t - t2)
            r_t, _ = kepler2eci(a_est, ecc_est, incl_est, raan_est, argp_est, nu_t)
            est_pts.append(r_t)
        est_pts = np.array(est_pts)
        ax.plot(est_pts[:, 0], est_pts[:, 1], est_pts[:, 2], 'b-', label="Estimated Orbit", linewidth=2)

        # Plot observations
        obs_pts = np.array([r1_true, r2_true, r3_true])
        ax.scatter(obs_pts[:, 0], obs_pts[:, 1], obs_pts[:, 2], color='red', s=50, label='Observations')
        
        # Plot LOS vectors from observers
        for R, r_true in zip([R1, R2, R3], [r1_true, r2_true, r3_true]):
            ax.plot([R[0], r_true[0]], [R[1], r_true[1]], [R[2], r_true[2]], 'g:', alpha=0.6)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Gauss IOD Results\nPos Error: {pos_error:.2f} m, Vel Error: {vel_error:.4f} m/s')
        ax.legend()
        
        # Adjust view
        # ax.view_init(elev=20, azim=45)
        
        output_path = "assets/gauss_iod_results.png"
        plt.savefig(output_path, dpi=150)
        print(f"\nFigure saved to {output_path}")
        
    except ImportError:
        print("\nMatplotlib not found. Skipping plot generation.")

if __name__ == "__main__":
    main()




