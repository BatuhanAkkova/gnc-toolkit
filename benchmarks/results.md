# OpenGNC Benchmarks

Performance comparison between pure Python (NumPy) and C++ (Eigen) implementations.

## C++ Accelerated Kalman Filters (Current Results)

| Operation | Iterations | Avg Time (Python) | Avg Time (C++) | Speedup |
|-----------|------------|-------------------|----------------|---------|
| quat_normalize | 50000 | 3.5 us | 1.5 us | **2.1x** |
| MEKF Predict | 20000 | 6.0 us | 1.8 us | **3.3x** |
| MEKF Update | 20000 | 30.0 us | 3.9 us | **7.7x** |
| UKF Predict | 5000 | 120.0 us | 25.4 us | **4.7x** |
| UKF Update | 5000 | 220.0 us | 30.0 us | **7.3x** |

> [!NOTE]
> C++ implementations use Eigen with SIMD optimizations enabled. Speedups are most significant in matrix-heavy Update steps (up to 7.7x).

---

## Baseline Benchmarks (Pure Python)

### Propagators
| Operation                           | Iterations | Total Time (s) | Avg Time (us) |
|-------------------------------------|------------|----------------|---------------|
| Kepler Propagator (Analytical)      | 5000       | 0.270          | 54.0        |
| Cowell Propagator (RK4)             | 500        | 0.200          | 400.0       |

### Gravity Models
| Operation                           | Iterations | Total Time (s) | Avg Time (us) |
|-------------------------------------|------------|----------------|---------------|
| J2 Gravity Acceleration             | 10000      | 0.108          | 10.8        |
| Harmonics Gravity (EGM2008 20x20)   | 1000       | 0.074          | 74.3        |

### Atmospheric Density
| Operation                           | Iterations | Total Time (s) | Avg Time (us) |
|-------------------------------------|------------|----------------|---------------|
| Exponential Density                 | 10000      | 0.098          | 9.8         |
| Harris-Priester Density             | 5000       | 0.602          | 120.5       |
| NRLMSISE-00 (pymsis)                | 100        | 0.097          | 974.1       |

### Mission Design & Guidance
| Operation                           | Iterations | Total Time (s) | Avg Time (us) |
|-------------------------------------|------------|----------------|---------------|
| Hohmann Transfer                    | 10000      | 0.170          | 17.0        |
| Optimal Combined Maneuver           | 100        | 0.019          | 190.0       |

### Classical Control & Math
| Operation                           | Iterations | Total Time (s) | Avg Time (us) |
|-------------------------------------|------------|----------------|---------------|
| PID Update                          | 50000      | 0.018          | 0.36        |
| quat_mult                           | 10000      | 0.033          | 3.3         |
| RK4 Step (3D Linear)                | 50000      | 0.643          | 12.9        |
