import numpy as np
from gnc_toolkit.navigation.iod import gauss_iod
from gnc_toolkit.utils.state_to_elements import kepler2eci

def verify_accuracy():
    print("--- Gauss IOD Accuracy Verification ---")
    
    # Orbit: LEO-like
    a = 7000e3
    ecc = 0.01
    incl = np.radians(45.0)
    raan = np.radians(10.0)
    argp = np.radians(20.0)
    mu = 398600.4415e9
    
    # Observer: Fixed ground station for simplicity
    R_earth = 6378137.0
    lat = np.radians(35.0)
    lon = np.radians(135.0)
    R_obs = np.array([
        R_earth * np.cos(lat) * np.cos(lon),
        R_earth * np.cos(lat) * np.sin(lon),
        R_earth * np.sin(lat)
    ])
    
    # Times (10 minute arc, 3 observations)
    t2 = 1800.0
    dt = 300.0
    times = [t2 - dt, t2, t2 + dt]
    
    # Generate perfect observations (No Light-Time first to check baseline)
    rho_hats = []
    Rs = []
    true_states = []
    
    n = np.sqrt(mu / a**3)
    
    for t in times:
        nu = n * t
        r, v = kepler2eci(a, ecc, incl, raan, argp, nu)
        true_states.append(np.concatenate([r, v]))
        
        rho_vec = r - R_obs
        rho_hats.append(rho_vec / np.linalg.norm(rho_vec))
        Rs.append(R_obs)

    expected_r2 = true_states[1][:3]
    expected_v2 = true_states[1][3:]

    # Run Enhanced Gauss IOD
    est_state = gauss_iod(
        rho_hats[0], rho_hats[1], rho_hats[2],
        times[0], times[1], times[2],
        Rs[0], Rs[1], Rs[2], mu
    )
    
    est_r2 = est_state[:3]
    est_v2 = est_state[3:]
    
    pos_err = np.linalg.norm(est_r2 - expected_r2)
    vel_err = np.linalg.norm(est_v2 - expected_v2)
    
    print(f"Position Error: {pos_err:.4f} m")
    print(f"Velocity Error: {vel_err:.4f} m/s")
    
    # Now test WITH Light-Time effect simulated
    print("\n--- Testing with Simulated Light-Time Effect ---")
    c = 2.99792458e8
    rho_hats_lt = []
    times_obs = []
    
    for i, t in enumerate(times):
        r_true = true_states[i][:3]
        rho_vec = r_true - R_obs
        rho_mag = np.linalg.norm(rho_vec)
        
        # The light we see at t was emitted at t - rho/c
        # But usually we measure at t_obs. Let's say times[i] are t_obs.
        # So the satellite was at r(t_obs - rho/c) when it emitted the light.
        t_emit = times[i] - (rho_mag / c)
        nu_emit = n * t_emit
        r_emit, _ = kepler2eci(a, ecc, incl, raan, argp, nu_emit)
        
        rho_vec_lt = r_emit - R_obs
        rho_hats_lt.append(rho_vec_lt / np.linalg.norm(rho_vec_lt))
        times_obs.append(times[i])

    est_state_lt = gauss_iod(
        rho_hats_lt[0], rho_hats_lt[1], rho_hats_lt[2],
        times_obs[0], times_obs[1], times_obs[2],
        Rs[0], Rs[1], Rs[2], mu
    )
    
    pos_err_lt = np.linalg.norm(est_state_lt[:3] - expected_r2)
    vel_err_lt = np.linalg.norm(est_state_lt[3:] - expected_v2)
    
    print(f"Position Error (with LT): {pos_err_lt:.4f} m")
    print(f"Velocity Error (with LT): {vel_err_lt:.4f} m/s")

if __name__ == "__main__":
    verify_accuracy()
