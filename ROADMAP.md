# GNC Toolkit — Open-Source Roadmap

> **Mission**: Become the definitive open-source Guidance, Navigation & Control toolkit for spacecraft engineers, scientists, and researchers — covering the full mission lifecycle from concept to operations.

Legend: ✅ Implemented · 🔄 Partial/Needs Enhancement · ⬜ Not Yet Implemented

---

## Table of Contents

1. [Environment & Physical Models](#1-environment--physical-models)
2. [Orbital Mechanics & Propagation](#2-orbital-mechanics--propagation)
3. [Attitude Dynamics & Kinematics](#3-attitude-dynamics--kinematics)
4. [Sensors & Measurement Models](#4-sensors--measurement-models)
5. [State Estimation & Navigation](#5-state-estimation--navigation)
6. [Guidance & Maneuver Planning](#6-guidance--maneuver-planning)
7. [Control Systems](#7-control-systems)
8. [Actuators](#8-actuators)
9. [Mission Analysis & Design Tools](#9-mission-analysis--design-tools)
10. [Simulation Framework](#10-simulation-framework)
11. [Infrastructure & Developer Experience](#11-infrastructure--developer-experience)

---

## 1. Environment & Physical Models

Foundation models that all other modules depend on.

### 1.1 Atmosphere
| Feature | Status | Notes |
|---|---|---|
| Exponential atmosphere | ✅ | `density.py` |
| Harris-Priester (diurnal bulge) | ✅ | `density.py` |
| NRLMSISE-00 (via PyMSIS) | ✅ | `density.py` |
| Jacchia-Bowman 2008 (JB2008) | ⬜ | Higher accuracy for solar-max |
| COSPAR International Reference Atmosphere (CIRA) | ⬜ | |
| Atmosphere co-rotation (wind model) | 🔄 | Basic support; add HWM14 horizontal wind |

### 1.2 Gravity Field
| Feature | Status | Notes |
|---|---|---|
| Two-body (Keplerian) | ✅ | |
| J2 perturbation | ✅ | `gravity.py` |
| EGM2008 Spherical Harmonics (recursive) | ✅ | `gravity.py` (`egm2008.csv`) |
| Luni-solar third-body gravity | ⬜ | Sun & Moon point-mass |
| Ocean tides (EGM96 corrections) | ⬜ | |
| Relativistic corrections (Schwarzschild, Lense-Thirring) | ⬜ | Required for high-precision LEO/MEO |

### 1.3 Geomagnetic & Solar
| Feature | Status | Notes |
|---|---|---|
| Tilted dipole model | ✅ | `mag_field.py` |
| IGRF-13 (via PPIGRF) | ✅ | `mag_field.py` |
| WMM (World Magnetic Model) | ⬜ | |
| Analytical solar position | ✅ | `solar.py` |
| Solar irradiance model | ✅ | `srp.py` |
| Umbra / penumbra shadow cones | ✅ | `srp.py` |
| Solar flux (F10.7) indexing | ⬜ | Real-time space weather inputs |

### 1.4 Radiation & Thermal
| Feature | Status | Notes |
|---|---|---|
| Total Ionising Dose (TID) model | ⬜ | |
| Single-Event Upset (SEU) rate | ⬜ | |
| Basic thermal environment (albedo, IR) | ⬜ | External heat fluxes for thermal control design |

---

## 2. Orbital Mechanics & Propagation

### 2.1 Propagators
| Feature | Status | Notes |
|---|---|---|
| Keplerian (analytical) | ✅ | `kepler.py` |
| Cowell numerical (RK4/45/853) | ✅ | `cowell.py` |
| SGP4 / SDP4 (TLE-based) | ⬜ | Critical for real mission ops and SSA |
| Gauss-Jackson multi-step integrator | ⬜ | Higher efficiency for long propagations |
| Symplectic integrators (Störmer-Verlet) | ⬜ | Energy-conserving long-duration sims |
| Encke's method | ⬜ | Perturbation from reference trajectory |

### 2.2 Orbit Representation & Conversion
| Feature | Status | Notes |
|---|---|---|
| Keplerian elements ↔ Cartesian | ✅ | `state_to_elements.py` |
| ECI ↔ ECEF | ✅ | `frame_conversion.py` |
| ECI → LVLH DCM | ✅ | `frame_conversion.py` |
| ECI → LLH / Geodetic | ✅ | `frame_conversion.py` |
| Perifocal (PQW) ↔ ECI | ✅ | `frame_conversion.py` |
| ECI ↔ ICRF (full IAU 2006/2000A) | ⬜ | Precession, nutation, polar wander |
| ECI ↔ EME2000 | ⬜ | |
| Mean ↔ Osculating elements | ⬜ | Brouwer theory conversion |
| Equinoctial orbital elements | ⬜ | Non-singular for circular/equatorial orbits |
| Modified equidistant cylindrical (MEE) | ⬜ | Optimal control-friendly representation |

### 2.3 Orbit Determination
| Feature | Status | Notes |
|---|---|---|
| Gauss IOD (3 observations) | ⬜ | Initial orbit determination |
| Laplace IOD | ⬜ | |
| Herrick-Gibbs IOD | ⬜ | |
| Differential Correction (batch least squares) | ⬜ | Refine OD from tracking data |
| Admissible region / attributables | ⬜ | Uncorrelated track processing |

---

## 3. Attitude Dynamics & Kinematics

### 3.1 Dynamics Models
| Feature | Status | Notes |
|---|---|---|
| Euler equations (rigid body) | ✅ | `rigid_body.py` |
| Flexible body coupling (modal) | ⬜ | Solar panels, antennas — essential for large sats |
| Fuel slosh dynamics (pendulum / mass-spring) | ⬜ | |
| Variable inertia tensor (fuel depletion) | ⬜ | |

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
| EKF for orbit determination (OD-EKF) | ⬜ | Orbit-level state estimation |
| Angle-only navigation | ⬜ | Passive ranging from visual sensors |
| GPS-based position estimation | ⬜ | Autonomous on-board navigation |
| Relative navigation (CW + EKF) | 🔄 | CW equations exist; EKF integration needed |
| SLAM-like surface navigation | ⬜ | Planetary landers, rovers |

### 5.4 Smoother Algorithms
| Feature | Status | Notes |
|---|---|---|
| Rauch-Tung-Striebel (RTS) smoother | ✅ | `rts_smoother.py` |
| Fixed-interval smoother | ⬜ | |

---

## 6. Guidance & Maneuver Planning

### 6.1 Impulsive Maneuvers
| Feature | Status | Notes |
|---|---|---|
| Hohmann transfer | ✅ | `maneuvers.py` |
| Bi-elliptic transfer | ✅ | `maneuvers.py` |
| Phasing maneuver | ✅ | `maneuvers.py` |
| Plane change (simple + combined) | ✅ | `maneuvers.py` |
| General Δv budget calculator | 🔄 | Extend with propellant mass (Tsiolkovsky) |
| Pork-chop plot generator | ⬜ | Launch window analysis |
| RAAN correction maneuver | ⬜ | |
| Combined maneuver optimization | ⬜ | Optimal split between inclination + altitude change |

### 6.2 Rendezvous & Proximity Operations (RPO)
| Feature | Status | Notes |
|---|---|---|
| Lambert solver (universal variables) | ✅ | `rendezvous.py` |
| Clohessy-Wiltshire (CW) propagation | ✅ | `rendezvous.py` |
| CW targeting (two-impulse) | ✅ | `rendezvous.py` |
| Tschauner-Hempel (elliptic CW) | ⬜ | For non-circular target orbits |
| Gauss-Lobatto collocation trajectory | ⬜ | |
| Safe-approach corridors | ⬜ | Collision-avoidance zones for proximity ops |
| Fuel-optimal rendezvous (primer vector) | ⬜ | |
| Multi-revolution Lambert | ⬜ | |

### 6.3 Continuous-Thrust Guidance
| Feature | Status | Notes |
|---|---|---|
| Q-law Lyapunov guidance | ⬜ | Robust low-thrust orbit raising |
| Indirect optimal (Pontryagin, primer vector) | ⬜ | |
| Direct collocation (GPOPS-style) | ⬜ | |
| ZEM/ZEV terminal guidance | ⬜ | Powered descent / landing |
| E-guidance / Apollo DPS | ⬜ | Historic + still applicable |
| Gravity-turn guidance | ⬜ | Launch/ascent |

### 6.4 Entry, Descent & Landing (EDL)
| Feature | Status | Notes |
|---|---|---|
| Ballistic entry trajectory | ⬜ | |
| Aerocapture guidance | ⬜ | |
| Powered descent guidance | ⬜ | |
| Terrain-relative navigation | ⬜ | |
| Hazard avoidance | ⬜ | |

### 6.5 Attitude Guidance (Reference Generation)
| Feature | Status | Notes |
|---|---|---|
| Nadir pointing reference | ⬜ | |
| Sun-pointing reference | ⬜ | |
| Target tracking (ground station / celestial) | ⬜ | |
| Slew path planning (eigenaxis vs. min-time) | ⬜ | |
| Attitude blending / blending profile | ⬜ | |

---

## 7. Control Systems

### 7.1 Classical Control
| Feature | Status | Notes |
|---|---|---|
| PID controller | ✅ | `pid.py` |
| B-Dot magnetic detumbling | ✅ | `bdot.py` |
| Momentum wheel desaturation | ✅ | `momentum_dumping.py` |
| Cross-product detumbling | ⬜ | Alternative magnetic detumbling |
| Rate damping control | ⬜ | |

### 7.2 Optimal Control
| Feature | Status | Notes |
|---|---|---|
| LQR (Algebraic Riccati) | ✅ | `lqr.py` |
| LQE / Kalman regulator | ✅ | `lqe.py` |
| LQG (combined LQR + LQE) | 🔄 | Components exist; add unified LQG class |
| Finite-horizon LQR (time-varying) | ⬜ | |
| H∞ robust control | ⬜ | Robust to uncertainty |
| H2 optimal control | ⬜ | |
| Linear MPC (constrained) | ✅ | `mpc.py` — SLSQP-based |
| Nonlinear MPC (single-shooting) | ✅ | `mpc.py` — SLSQP-based |
| MPC with CasADi / ACADOS backend | ⬜ | Production-grade real-time NMPC |

### 7.3 Nonlinear & Geometric Control
| Feature | Status | Notes |
|---|---|---|
| Sliding Mode Control (SMC) | ✅ | `sliding_mode.py` |
| Feedback Linearization | ✅ | `feedback_linearization.py` |
| Geometric SO(3) attitude control (Lee et al.) | ⬜ | Global stability, avoids singularities |
| Passivity-based control (PBC) | ⬜ | |
| Backstepping control | ⬜ | |
| Adaptive control (MRAC, L1) | ⬜ | Handles plant uncertainty online |
| Incremental Nonlinear Dynamic Inversion (INDI) | ⬜ | Popular in aerospace applications |

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
| Control moment gyroscope (CMG) | ⬜ | Singularity avoidance (null motion, SR-inv) |
| Variable speed CMG (VSCMG) | ⬜ | |
| Thruster cluster / torque allocation | ⬜ | Over-actuated thruster distribution |
| Solar sail model | ⬜ | Attitude and orbit control via SRP |
| Tethered system dynamics | ⬜ | |
| Magnetically levitated reaction wheel | ⬜ | |

### 8.3 Actuator Dynamics & Allocation
| Feature | Status | Notes |
|---|---|---|
| Reaction wheel friction / saturation model | 🔄 | Add motor dynamics and speed limits |
| Control allocation / pseudo-inverse | ⬜ | Distribute commands across redundant actuators |
| Null-motion management (for CMG) | ⬜ | |

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
