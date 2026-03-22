# GNC Toolkit — Open-Source Roadmap

> **Mission**: Become the definitive open-source Guidance, Navigation & Control toolkit for spacecraft engineers, scientists, and researchers — covering the full mission lifecycle from concept to operations.

### 7.4 Formation Flying Control
| Feature | Status | Notes |
|---|---|---|
| Virtual structure formation control | ⬜ | |
| Leader-follower formation control | ⬜ | |
| Fuel-balanced formation keeping | ⬜ | |
| Distributed consensus algorithms | ⬜ | |

### 9.1 Coverage & Access
| Feature | Status | Notes |
|---|---|---|
| Ground station access windows | ✅ | `coverage.py` |
| Constellation coverage analysis | ⬜ | Coverage gap and revisit time |
| Ground track visualization | ✅ | `coverage.py` |
| Lighting conditions analysis | ✅ | `coverage.py` |

### 9.2 Launch & Deployment
| Feature | Status | Notes |
|---|---|---|
| Launch window calculator | ✅ | `launch.py` |
| Trajectory-to-orbit injection | ✅ | `launch.py` |
| Constellation deployment sequencing | ⬜ | |

### 9.5 Space Situational Awareness
| Feature | Status | Notes |
|---|---|---|
| Conjunction analysis (Pc computation) | ⬜ | Probability of collision |
| TLE catalog interface | ⬜ | |
| Object tracking (orbit correlation) | ⬜ | |
| Debris avoidance maneuver planning | ⬜ | |

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
