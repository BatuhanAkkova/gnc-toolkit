# GNC Toolkit Examples

This directory contains a suite of high-fidelity simulations designed to showcase the **Guidance, Navigation, and Control (GNC)** capabilities of the toolkit. Each example focuses on a specific aspect of satellite operations, from low-level stabilization to high-level mission planning.

---

## 1. CubeSat Detumbling (Control)
**Script:** `01_cubesat_detumbling.py`
**Scenario:** A 3U CubeSat is deployed from a launch vehicle with significant tip-off rates.
**GNC Focus:**
*   **Sensors:** Noisy Magnetometer.
*   **Actuators:** Magnetic Torquer rods.
*   **Control:** B-Dot magnetic controller for kinetic energy dissipation.
*   **Dynamics:** 6-DOF Rigid body Euler equations.

## 2. VLEO Orbit Maintenance (Guidance & Control)
**Script:** `02_vleo_orbit_maintenance.py`
**Scenario:** A satellite in a 250 km Very Low Earth Orbit (VLEO) experiences high atmospheric drag.
**GNC Focus:**
*   **Environment:** `Harris-Priester` diurnal bulge density model.
*   **Dynamics:** Cowell propagator with J2 perturbations and complex drag ($C_d, Area$).
*   **Control:** Hysteresis-based thruster logic to maintain altitude within a +/- 100m deadband.
*   **Actuators:** `ElectricThruster` model with fuel and power tracking.

## 3. Momentum Dumping (Actuator Management)
**Script:** `03_momentum_dumping.py`
**Scenario:** A spacecraft maintains Earth-pointing attitude while being subject to a constant disturbance torque (e.g., solar pressure). 
**GNC Focus:**
*   **Actuators:** `ReactionWheels` (primary) and `Magnetorquers` (secondary).
*   **Control Allocation:** Distributing torque demands to prevent wheel saturation.
*   **Desaturation:** Cross-product law to use magnetorquers for dumping accumulated angular momentum.

## 4. Autonomous Rendezvous (Relative GNC)
**Script:** `04_autonomous_rendezvous.py`
**Scenario:** A chaser satellite performs a multi-burn approach to a target in Geostationary Orbit (GEO).
**GNC Focus:**
*   **Guidance:** `cw_targeting` for precise Delta-V calculation in the Hill's frame.
*   **Navigation:** Relative state propagation using Clohessy-Wiltshire equations.
*   **Control:** Execution of translational maneuvers to achieve proximity operations.

## 5. Attitude Estimation with MEKF (Navigation)
**Script:** `05_attitude_estimation_mekf.py`
**Scenario:** Fusing sensors with different characteristics to provide a stable attitude estimate.
**GNC Focus:**
*   **Filter:** `MEKF` (Multiplicative Extended Kalman Filter) handling quaternion states and rate-gyro biases.
*   **Sensors:** `StarTracker` (accurate, absolute, low frequency) + `Gyroscope` (noisy, relative, high frequency).
*   **Analysis:** Verification of estimation consistency (Error vs. 3-sigma bounds).
