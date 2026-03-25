# GNC Toolkit — Open-Source Roadmap

> **Mission**: Become the definitive open-source Guidance, Navigation & Control toolkit for spacecraft engineers, scientists, and researchers — covering the full mission lifecycle from concept to operations.

### 11.3 Packaging & Distribution
| Feature | Status | Notes |
|---|---|---|
| pip installable (`pyproject.toml`) | ✅ | `pyproject.toml` |
| PyPI release workflow | ✅ | |
| Conda-forge recipe | ✅ | |
| Docker image for reproducibility | ✅ | |
| C extension / Cython acceleration | ⬜ | |

### 11.4 Code Quality
| Feature | Status | Notes |
|---|---|---|
| Type hints | 🔄 | Partial; expand to all public APIs |
| Docstrings (NumPy style) | 🔄 | Partial; enforce style guide |
| Linting (ruff / flake8) | ✅ | |
| Pre-commit hooks | ✅ | |
| Benchmarks / performance profiling | ✅ | `timeit`-based suite |

## Reference Implementations & Benchmarks

For validation, all new modules should be cross-verified against:
- **GMAT** (NASA open-source) — orbit propagation & maneuver planning
- **Orekit** (CS GROUP) — high-fidelity orbit determination
- **SPICE / NAIF** — ephemeris and frame transforms  
- **STK/AGI** — coverage and access analysis
- **Poliastro / AstroPy** — astrodynamics cross-check
- **ODTBX** (NASA) — estimation algorithm benchmarks

---

*Last updated: March 2026 | Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md)*
