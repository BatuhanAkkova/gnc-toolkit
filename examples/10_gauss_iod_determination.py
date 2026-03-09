import numpy as np
from gnc_toolkit.navigation.iod import gauss_iod
from gnc_toolkit.utils.state_to_elements import kepler2eci, eci2kepler
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

    # 3. Simulate observer at Geocenter (for simplicity)
    # In a real scenario, R1, R2, R3 would be the observer's ECI position vectors
    R1 = np.zeros(3)
    R2 = np.zeros(3)
    R3 = np.zeros(3)

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

if __name__ == "__main__":
    main()
