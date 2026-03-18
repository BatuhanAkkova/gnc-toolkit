# GNC Toolkit — Open-Source Roadmap

> **Mission**: Become the definitive open-source Guidance, Navigation & Control toolkit for spacecraft engineers, scientists, and researchers — covering the full mission lifecycle from concept to operations.

## 1. Environment & Physical Models

Foundation models that all other modules depend on.

### 1.1 Atmosphere
| Feature | Status | Notes |
|---|---|---|
| Exponential atmosphere | ✅ | `density.py` |
| Harris-Priester (diurnal bulge) | ✅ | `density.py` |
| NRLMSISE-00 (via PyMSIS) | ✅ | `density.py` |
| Jacchia-Bowman 2008 (JB2008) | ✅ | `density.py` (Structural) |
| COSPAR International Reference Atmosphere (CIRA) | ✅ | `density.py` (CIRA-72) |
| Atmosphere co-rotation (wind model) | ✅ | `wind.py` |

### 1.2 Gravity Field
| Feature | Status | Notes |
|---|---|---|
| Two-body (Keplerian) | ✅ | |
| J2 perturbation | ✅ | `gravity.py` |
| EGM2008 Spherical Harmonics (recursive) | ✅ | `gravity.py` (`egm2008.csv`) |
| Luni-solar third-body gravity | ✅ | `gravity.py` (Sun/Moon) |
| Ocean tides (EGM96 corrections) | ✅ | `gravity.py` (Simplified) |
| Relativistic corrections (Schwarzschild, Lense-Thirring) | ✅ | `gravity.py` |

### 1.3 Geomagnetic & Solar
| Feature | Status | Notes |
|---|---|---|
| Tilted dipole model | ✅ | `mag_field.py` |
| IGRF-13 (via PPIGRF) | ✅ | `mag_field.py` |
| WMM (World Magnetic Model) | ✅ | `mag_field.py` (Harmonic proxy) |
| Analytical solar position | ✅ | `solar.py` |
| Solar irradiance model | ✅ | `srp.py` |
| Umbra / penumbra shadow cones | ✅ | `srp.py` |
| Solar flux (F10.7) indexing | ✅ | `space_weather.py` |

### 1.4 Radiation & Thermal
| Feature | Status | Notes |
|---|---|---|
| Total Ionising Dose (TID) model | ✅ | `radiation.py` |
| Single-Event Upset (SEU) rate | ✅ | `radiation.py` |
| Basic thermal environment (albedo, IR) | ✅ | `thermal.py` |

---

## 2. Orbital Mechanics & Propagation

### 2.1 Propagators
| Feature | Status | Notes |
|---|---|---|
| Keplerian (analytical) | ✅ | `kepler.py` |
| Cowell numerical (RK4/45/853) | ✅ | `cowell.py` |
| SGP4 / SDP4 (TLE-based) | ✅ | `sgp4_propagator.py` |
| Gauss-Jackson multi-step integrator | ✅ | Implemented `ab_moulton.py` 8th order |
| Symplectic integrators (Störmer-Verlet) | ✅ | `symplectic.py` (Yoshida 4th order) |
| Encke's method | ✅ | `encke.py` |

### 2.2 Orbit Representation & Conversion
| Feature | Status | Notes |
|---|---|---|
| Keplerian elements ↔ Cartesian | ✅ | `state_to_elements.py` |
| ECI ↔ ECEF | ✅ | `frame_conversion.py` |
| ECI → LVLH DCM | ✅ | `frame_conversion.py` |
| ECI → LLH / Geodetic | ✅ | `frame_conversion.py` |
| Perifocal (PQW) ↔ ECI | ✅ | `frame_conversion.py` |
| ECI ↔ ICRF (full IAU 2006/2000A) | ✅ | `frame_conversion.py` (Precession) |
| ECI ↔ EME2000 | ✅ | `frame_conversion.py` |
| Mean ↔ Osculating elements | ✅ | `mean_elements.py` (J2 Secular) |
| Equinoctial orbital elements | ✅ | `equinoctial_utils.py` |
| Modified equinoctial elements (MEE) | ✅ | `mee_utils.py` |

### 2.3 Orbit Determination
| Feature | Status | Notes |
|---|---|---|
| Gauss IOD (3 observations) | ✅ | `iod.py` (Robust iterative refinement) |
| Laplace IOD | ✅ | `iod.py` |
| Herrick-Gibbs IOD | ✅ | `iod.py` (Short arc) |
| Gibbs IOD | ✅ | `iod.py` (Long arc) |
| Differential Correction (batch least squares) | ✅ | `batch_ls.py` |

---

## 3. Attitude Dynamics & Kinematics

### 3.1 Dynamics Models
| Feature | Status | Notes |
|---|---|---|
| Euler equations (rigid body) | ✅ | `rigid_body.py` |
| Flexible body coupling (modal) | ✅ | `flexible_body.py` |
| Fuel slosh dynamics (pendulum / mass-spring) | ✅ | `fuel_slosh.py` |
| Variable inertia tensor (fuel depletion) | ✅ | `variable_inertia.py` |

### 3.2 Attitude Kinematics & Representations
| Feature | Status | Notes |
|---|---|---|
| Quaternion kinematics | ✅ | `quat_utils.py` |
| Direction Cosine Matrices (DCM) | ✅ | via quat_utils / state_conversion |
| Euler angles (12 sequences) | ✅ | `euler_utils.py` |
| Modified Rodrigues Parameters (MRP) | ✅ | `mrp_utils.py` |
| Gibbs / Classical Rodrigues Parameters | ✅ | `crp_utils.py` |
| Cayley-Klein parameters | ✅ | `cayley_klein_utils.py` |

---

## 4. Sensors & Measurement Models

### 4.1 Existing Sensors
| Feature | Status | Notes |
|---|---|---|
| Gyroscope (bias + noise) | ✅ | `gyroscope.py` |
| Magnetometer | ✅ | `magnetometer.py` |
| Star tracker | ✅ | `star_tracker.py` |
| Sun sensor | ✅ | `sun_sensor.py` |
| Base sensor class | ✅ | `sensor.py` |

### 4.2 Missing Sensors
| Feature | Status | Notes |
|---|---|---|
| GPS / GNSS receiver (pseudoranges, L1/L2) | ✅ | `gnss_receiver.py` |
| Star catalog integration (Hipparcos / Tycho-2) | ✅ | `star_catalog.py` |
| Horizon / Earth sensor model | ✅ | `horizon_sensor.py` |
| Coarse Sun Sensor (CSS) array | ✅ | `sun_sensor_array.py` |
| Accelerometer / IMU model | ✅ | `imu.py` (includes `Accelerometer` & `IMU`) |
| Altimeter / Radar altimeter model | ✅ | `altimeter.py` |
| LIDAR model (for proximity ops) | ✅ | `lidar.py` |
| Camera / Optical sensor model | ✅ | `camera.py` |

### 4.3 Measurement Fidelity
| Feature | Status | Notes |
|---|---|---|
| Allan variance / FOGM noise model | ✅ | `sensor.py` (via `apply_fogm_noise`) |
| Sensor failure / fault injection | ✅ | `sensor.py` (via `apply_faults`) |
| Sensor calibration residuals | ✅ | `sensor.py` (via `apply_calibration`) |

---

## 5. State Estimation & Navigation

### 5.1 Attitude Determination
| Feature | Status | Notes |
|---|---|---|
| TRIAD | ✅ | `triad.py` |
| QUEST | ✅ | `quest.py` |
| FOAM (Fast Optimal Attitude Matrix) | ✅ | `foam.py` |
| REQUEST (recursive QUEST) | ✅ | `request.py` |
| Davenport q-method | ✅ | `davenport_q.py` |

### 5.2 Kalman-Family Filters
| Feature | Status | Notes |
|---|---|---|
| Linear Kalman Filter (KF) | ✅ | `kf.py` |
| Extended Kalman Filter (EKF) | ✅ | `ekf.py` |
| Multiplicative EKF (MEKF) | ✅ | `mekf.py` (attitude-specific) |
| Unscented Kalman Filter (UKF) | ✅ | `ukf.py` |
| Square-Root UKF | ✅ | `sr_ukf.py` |
| Ensemble Kalman Filter (EnKF) | ✅ | `enkf.py` |
| Cubature Kalman Filter (CKF) | ✅ | `ckf.py` |
| Particle Filter / Sequential Monte Carlo | ✅ | `pf.py` |
| Adaptive KF (noise covariance estimation) | ✅ | `akf.py` |
| Interacting Multiple Model (IMM) filter | ✅ | `imm.py` |

### 5.3 Orbit Determination & Navigation
| Feature | Status | Notes |
|---|---|---|
| EKF for orbit determination (OD-EKF) | ✅ | `orbit_determination.py` |
| Angle-only navigation | ✅ | `angle_only_nav.py` |
| GPS-based position estimation | ✅ | `gps_nav.py` |
| Relative navigation (CW + EKF) | ✅ | `relative_nav.py` |
| SLAM-like surface navigation | ✅ | `surface_nav.py` |

### 5.4 Smoother Algorithms
| Feature | Status | Notes |
|---|---|---|
| Rauch-Tung-Striebel (RTS) smoother | ✅ | `rts_smoother.py` |
| Fixed-interval smoother | ✅ | `fixed_interval_smoother.py` |

---

## 6. Guidance & Maneuver Planning

### 6.1 Impulsive Maneuvers
| Feature | Status | Notes |
|---|---|---|
| Hohmann transfer | ✅ | `maneuvers.py` |
| Bi-elliptic transfer | ✅ | `maneuvers.py` |
| Phasing maneuver | ✅ | `maneuvers.py` |
| Plane change (simple + combined) | ✅ | `maneuvers.py` |
| General Δv budget calculator | ✅ | `maneuvers.py` (Propellant mass) |
| Pork-chop plot generator | ✅ | `porkchop.py` |
| RAAN correction maneuver | ✅ | `maneuvers.py` (Optimal at poles) |
| Combined maneuver optimization | ✅ | `maneuvers.py` (Optimal split) |

### 6.2 Rendezvous & Proximity Operations (RPO)
| Feature | Status | Notes |
|---|---|---|
| Lambert solver (universal variables) | ✅ | `rendezvous.py` |
| Clohessy-Wiltshire (CW) propagation | ✅ | `rendezvous.py` |
| CW targeting (two-impulse) | ✅ | `rendezvous.py` |
| Tschauner-Hempel (elliptic CW) | ✅ | `rendezvous.py` |
| Gauss-Lobatto collocation trajectory | ✅ | `rendezvous.py` |
| Safe-approach corridors | ✅ | `rendezvous.py` |
| Fuel-optimal rendezvous (primer vector) | ✅ | `rendezvous.py` |
| Multi-revolution Lambert | ✅ | `rendezvous.py` |

### 6.3 Continuous-Thrust Guidance
| Feature | Status | Notes |
|---|---|---|
| Q-law Lyapunov guidance | ✅ | `continuous_thrust.py` |
| Indirect optimal (Pontryagin, primer vector) | ✅ | `continuous_thrust.py` |
| Direct collocation (GPOPS-style) | ✅ | `continuous_thrust.py` |
| ZEM/ZEV terminal guidance | ✅ | `continuous_thrust.py` |
| E-guidance / Apollo DPS | ✅ | `continuous_thrust.py` |
| Gravity-turn guidance | ✅ | `continuous_thrust.py` |

### 6.4 Entry, Descent & Landing (EDL)
| Feature | Status | Notes |
|---|---|---|
| Ballistic entry trajectory | ✅ | `edl.py` |
| Aerocapture guidance | ✅ | `edl.py` (NPC Placeholder) |
| Powered descent guidance | ✅ | `edl.py` (via `continuous_thrust.py`) |
| Terrain-relative navigation | ✅ | `terrain_nav.py` |
| Hazard avoidance | ✅ | `edl.py` |

### 6.5 Attitude Guidance (Reference Generation)
| Feature | Status | Notes |
|---|---|---|
| Nadir pointing reference | ✅ | `attitude_guidance.py` |
| Sun-pointing reference | ✅ | `attitude_guidance.py` |
| Target tracking (ground station / celestial) | ✅ | `attitude_guidance.py` |
| Slew path planning (eigenaxis vs. min-time) | ✅ | eigenaxis via `attitude_guidance.py` |
| Attitude blending / blending profile | ✅ | `attitude_guidance.py` |

---

## 7. Control Systems

### 7.1 Classical Control
| Feature | Status | Notes |
|---|---|---|
| PID controller | ✅ | `pid.py` |
| B-Dot magnetic detumbling | ✅ | `bdot.py` |
| Momentum wheel desaturation | ✅ | `momentum_dumping.py` (CrossProductLaw) |
| Cross-product detumbling | ✅ | `momentum_dumping.py` (CrossProductLaw) |
| Rate damping control | ✅ | `rate_damping.py` |

### 7.2 Optimal Control
| Feature | Status | Notes |
|---|---|---|
| LQR (Algebraic Riccati) | ✅ | `lqr.py` |
| LQE / Kalman regulator | ✅ | `lqe.py` |
| LQG (combined LQR + LQE) | ✅ | `lqg.py` |
| Finite-horizon LQR (time-varying) | ✅ | `finite_horizon_lqr.py` |
| H∞ robust control | ✅ | `h_infinity.py` |
| H2 optimal control | ✅ | `h2_control.py` |
| Linear MPC (constrained) | ✅ | `mpc.py` — SLSQP-based |
| Nonlinear MPC (single-shooting) | ✅ | `mpc.py` — SLSQP-based |
| MPC with CasADi / ACADOS backend | ✅ | `mpc_casadi.py` — Production-grade real-time NMPC |

### 7.3 Nonlinear & Geometric Control
| Feature | Status | Notes |
|---|---|---|
| Sliding Mode Control (SMC) | ✅ | `sliding_mode.py` |
| Feedback Linearization | ✅ | `feedback_linearization.py` |
| Geometric SO(3) attitude control (Lee et al.) | ✅ | `geometric_control.py` |
| Passivity-based control (PBC) | ✅ | `passivity_control.py` |
| Backstepping control | ✅ | `backstepping_control.py` |
| Adaptive control (MRAC, L1) | ✅ | `adaptive_control.py` |
| Incremental Nonlinear Dynamic Inversion (INDI) | ✅ | `indi_control.py` |

### 7.4 Formation Flying Control
| Feature | Status | Notes |
|---|---|---|
| Virtual structure formation control | ⬜ | |
| Leader-follower formation control | ⬜ | |
| Fuel-balanced formation keeping | ⬜ | |
| Distributed consensus algorithms | ⬜ | |

### 7.5 Fault Detection, Isolation & Recovery (FDIR)
| Feature | Status | Notes |
|---|---|---|
| Analytical redundancy / residual generation | ⬜ | |
| Parity-space FDIR | ⬜ | |
| Safe mode logic | ⬜ | |
| Actuator failure accommodation | ⬜ | |

---

## 8. Actuators

### 8.1 Existing Actuators
| Feature | Status | Notes |
|---|---|---|
| Reaction wheel (torque + momentum model) | ✅ | `reaction_wheel.py` |
| Magnetorquer | ✅ | `magnetorquer.py` |
| Chemical thruster | ✅ | `thruster.py` |
| Electric thruster (specific impulse) | ✅ | `thruster.py` |
| Actuator base class | ✅ | `actuator.py` |

### 8.2 Missing Actuators
| Feature | Status | Notes |
|---|---|---|
| Control moment gyroscope (CMG) | ✅ | `cmg.py` — Singularity avoidance support |
| Variable speed CMG (VSCMG) | ✅ | `vscmg.py` |
| Thruster cluster / torque allocation | ✅ | `thruster.py` (`ThrusterCluster`) |
| Solar sail model | ✅ | `solar_sail.py` |
| Tethered system dynamics | ⬜ | |
| Magnetically levitated reaction wheel | ⬜ | |

### 8.3 Actuator Dynamics & Allocation
| Feature | Status | Notes |
|---|---|---|
| Reaction wheel friction / saturation model | ✅ | `reaction_wheel.py` — added Static/Viscous/Coulomb |
| Control allocation / pseudo-inverse | ✅ | `allocation.py` — PseudoInverse, SR-Inverse, Null-motion |
| Null-motion management (for CMG) | ✅ | `allocation.py` (`NullMotionManager`) |

---

## 9. Mission Analysis & Design Tools

### 9.1 Coverage & Access
| Feature | Status | Notes |
|---|---|---|
| Ground station access windows | ⬜ | Link budget and pass planning |
| Constellation coverage analysis | ⬜ | Coverage gap and revisit time |
| Ground track visualization | ⬜ | |
| Lighting conditions analysis | ⬜ | Eclipse fraction, beta angle |

### 9.2 Launch & Deployment
| Feature | Status | Notes |
|---|---|---|
| Launch window calculator | ⬜ | Pork-chop plots |
| Trajectory-to-orbit injection | ⬜ | |
| Constellation deployment sequencing | ⬜ | |

### 9.3 Propellant & ΔV Budgeting
| Feature | Status | Notes |
|---|---|---|
| Tsiolkovsky rocket equation | ⬜ | Mass budget for maneuvers |
| Maneuver sequence optimizer | ⬜ | Minimize total ΔV for mission profile |
| Lifetime / reentry prediction (drag-based) | ⬜ | |

### 9.4 Link Budget & Communications
| Feature | Status | Notes |
|---|---|---|
| Friis link budget calculator | ⬜ | |
| Doppler shift calculator | ⬜ | |
| Atmospheric attenuation model | ⬜ | |

### 9.5 Space Situational Awareness
| Feature | Status | Notes |
|---|---|---|
| Conjunction analysis (Pc computation) | ⬜ | Probability of collision |
| TLE catalog interface | ⬜ | |
| Object tracking (orbit correlation) | ⬜ | |
| Debris avoidance maneuver planning | ⬜ | |

---

## 10. Simulation Framework

### 10.1 Existing Infrastructure
| Feature | Status | Notes |
|---|---|---|
| Modular integrators (RK4/45/853) | ✅ | `integrators/` |
| Perturbation injection via callback | ✅ | `cowell.py` `perturbation_acc_fn` |
| Examples / scenario scripts | ✅ | `examples/` directory |

### 10.2 Simulation Architecture
| Feature | Status | Notes |
|---|---|---|
| End-to-end mission simulator class | ⬜ | Unified sim loop: propagate → sense → estimate → control |
| Discrete-event simulation support | ⬜ | Maneuver scheduling, mode transitions |
| Monte Carlo simulation harness | ⬜ | Uncertainty quantification, WC analysis |
| Scenario configuration (YAML/JSON) | ⬜ | Reproducible mission setups |
| Simulation replay & logging | ⬜ | HDF5 / NetCDF output |
| Real-time simulation (wall-clock sync) | ⬜ | HIL / SIL testing |
| Multi-body / constellation simulation | ⬜ | Simulate N spacecraft simultaneously |

### 10.3 Visualization
| Feature | Status | Notes |
|---|---|---|
| 2D plots (existing examples) | ✅ | matplotlib plots in `assets/` |
| 3D orbit visualization | ⬜ | Plotly / Matplotlib 3D |
| Attitude sphere / Bloch sphere | ⬜ | |
| Ground track map | ⬜ | Cartopy or Basemap overlay |
| Coverage heat map | ⬜ | |
| Interactive dashboard | ⬜ | Panel or Plotly Dash |

---

## 11. Infrastructure & Developer Experience

### 11.1 Testing
| Feature | Status | Notes |
|---|---|---|
| Unit tests (pytest) | 🔄 | `tests/` directory exists; coverage needs expansion |
| Integration tests | ⬜ | End-to-end scenario validation |
| Regression tests vs. reference data | ⬜ | e.g., GMAT, STK, Orekit comparisons |
| Continuous integration (GitHub Actions) | 🔄 | `.github/` exists; needs full test pipeline |
| Code coverage reporting | ⬜ | Codecov / coveralls integration |

### 11.2 Documentation
| Feature | Status | Notes |
|---|---|---|
| GitHub Pages (Sphinx) | 🔄 | `docs/` skeleton exists |
| API reference (auto-generated) | ⬜ | sphinx-autodoc |
| Tutorial notebooks (Jupyter) | 🔄 | `tutorials/` exists; expand coverage |
| Theory background (equations) | ⬜ | Math-heavy docs explaining algorithms |
| Contributing guide | ⬜ | CONTRIBUTING.md |

### 11.3 Packaging & Distribution
| Feature | Status | Notes |
|---|---|---|
| pip installable (`pyproject.toml`) | ✅ | `pyproject.toml` |
| PyPI release workflow | ⬜ | |
| Conda-forge recipe | ⬜ | |
| Docker image for reproducibility | ⬜ | |
| MATLAB / Octave compatibility layer | ⬜ | Wider community reach |
| C extension / Cython acceleration | ⬜ | HPC use-cases |

### 11.4 Code Quality
| Feature | Status | Notes |
|---|---|---|
| Type hints | 🔄 | Partial; expand to all public APIs |
| Docstrings (NumPy style) | 🔄 | Partial; enforce style guide |
| Linting (ruff / flake8) | ⬜ | |
| Pre-commit hooks | ⬜ | |
| Benchmarks / performance profiling | ⬜ | `timeit`-based suite |

---

## Implementation Priority

| Phase | Focus Area | Key Deliverables |
|---|---|---|
| **Phase 1** (Foundation) | Frame transforms, attitude representations, test coverage | Full IAU 2006 ECI/ECEF, MRP, equinoctial elements, unit test parity with GMAT |
| **Phase 2** (Navigation) | Orbit determination, GPS, RTS smoother | IOD (Gauss/Laplace), OD-EKF, GPS receiver model, ✅ RTS smoother |
| **Phase 3** (Guidance) | Attitude guidance, low-thrust, rendezvous | Q-law, ZEM/ZEV, Tschauner-Hempel, primer vector, attitude slew profiles |
| **Phase 4** (Control) | Geometric control, FDIR, CMG | SO(3) attitude controller, CMG with singularity avoidance, IMM filter |
| **Phase 5** (Mission Design) | Coverage, link budgets, ΔV budgeting, conjunction | Ground station access, Tsiolkovsky budget, Pc computation |
| **Phase 6** (Simulation) | End-to-end sim, Monte Carlo, visualization | Unified sim loop, MC harness, 3D orbit viewer, scenario YAML |
| **Phase 7** (Distribution) | PyPI, conda-forge, docs, CI/CD | Full API docs, PyPI release, GitHub Actions pipeline, Jupyter book |

---

## Reference Implementations & Benchmarks

For validation, all new modules should be cross-verified against:
- **GMAT** (NASA open-source) — orbit propagation & maneuver planning
- **Orekit** (CS GROUP) — high-fidelity orbit determination
- **SPICE / NAIF** — ephemeris and frame transforms  
- **STK/AGI** — coverage and access analysis
- **Poliastro / AstroPy** — astrodynamics cross-check
- **ODTBX** (NASA) — estimation algorithm benchmarks

---

*Last updated: March 2026 | Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon)*
