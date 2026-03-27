# OpenGNC — Open-Source Roadmap

> **Mission**: Become the definitive open-source Guidance, Navigation & Control toolkit for spacecraft engineers, scientists, and researchers — covering the full mission lifecycle from concept to operations.

This roadmap outlines the strategic development of **OpenGNC** to bridge the gap between academic simulation and flight-ready software.

---

## Strategic Objectives

1. **High-Fidelity Transparency**: Every algorithm and environment model must be fully documented and traceable to published aerospace standards.
2. **Interoperability**: Seamless integration with tools like GMAT, Orekit, and STK.
3. **Flight-Readiness**: Focus on C/C++ acceleration and deterministic performance for embedded targets.
4. **Autonomous Ops**: Implement modern AI/ML methods for onboard decision-making and fault isolation.

---

## Development Phases

### Phase 1: High-Fidelity Foundation & Orbit Mastery
*Solidify the core orbital mechanics and environment tools.*
- [ ] **Recursive Gravity Models**: Optimize EGM2008 up to degree/order 360+ via Cython/Numba.
- [ ] **Multi-Body Dynamics**: Implement Circular Restricted Three-Body Problem (CR3BP) and Solar System JPL Ephemeris integration.
- [ ] **Advanced Propagators**: Add Gauss-Variational Equations (GVE) and Taylor Series integrators.
- [ ] **Covariance Analysis**: Formalize Covariance Transform Tool (ECI/RIC/TLE) and reach-ability analysis.

### Phase 2: Autonomy, AI & Next-Gen GNC
*Integrate modern control and estimation techniques.*
- [ ] **Relative Navigation**: Implement Filter-based relative state estimation for swarming and docking.
- [ ] **AI/ML Integration**: Reinforcement Learning (RL) hooks for autonomous rendezvous and optimal control tuning.
- [ ] **FDIR v2**: Implement PCA and LSTM-based anomaly detection for sensor/actuator failures.
- [ ] **MPC Optimization**: Leverage CasADi for highly constrained multi-burn optimization (Debris removal, refueling).

### Phase 3: Deep Space & Lunar Exploration
*Expand capabilities beyond Earth orbit.*
- [ ] **Lunar Environment**: High-fidelity Mascons, Moon-centered frames, and Lunar albedo models.
- [ ] **Deep Space Navigation**: DSN (Deep Space Network) scheduling helpers and Optical Navigation (OpNav) from limb/crater detection.
- [ ] **Radiation Shielding Analysis**: Integration of radiation environment models (AP8/AE8) for mission design.

### Phase 4: Flight Integration & HIL Framework
*Bridging simulation to real hardware.*
- [ ] **Embedded Target Support**: Move core Kalman Filters and Control Laws to optimized C99-compatible code.
- [ ] **HIL/PIL Framework**: Generic interface for simulated sensors to talk to hardware over Serial/UDP/ROS2.
- [ ] **Telemetry Ops**: Real-time CCSDS packet parsing and de-commutation framework for flight operations.
- [ ] **Verification**: Automated Monte Carlo analysis suite for robust stability and performance margins.

---

## Infrastructure & Maintenance

### Packaging & Distribution
| Feature | Status | Notes |
|---|---|---|
| pip installable (`pyproject.toml`) | ✅ | `pyproject.toml` |
| PyPI / Conda-forge Release | ✅ | Automate via GitHub Actions |
| Docker image for reproducibility | ✅ | For HPC and CI environments |
| C extension / Cython acceleration | 🔄 | Currently in progress for propagators |

### Code Quality & Validation
| Feature | Status | Notes |
|---|---|---|
| Type hints (Complete Coverage) | 🔄 | Expand to all public APIs |
| Docstrings (NumPy style) | ✅ | Enforced via Ruff |
| Linting / Pre-commit | ✅ | |
| Benchmarks / Profiling | 🔄 | Expand `timeit` suite to core integrators |
| Benchmarking vs Industry Tools | 🔄 | (GMAT, Orekit, SPICE) |

---

*Last updated: March 2026 | Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md)*




