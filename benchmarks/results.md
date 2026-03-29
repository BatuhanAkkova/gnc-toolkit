# OpenGNC Benchmarks

## Propagators

| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |
|-------------------------------------|------------|----------------|---------------|
| Kepler Propagator (Analytical)      | 5000       | 0.27974    | 55.95        |
| Cowell Propagator (RK4, dt=100, step=10) | 500        | 0.20386    | 407.71       |

## Gravity Models

| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |
|-------------------------------------|------------|----------------|---------------|
| J2 Gravity Acceleration             | 10000      | 0.12048    | 12.05        |
| Harmonics Gravity (EGM2008 20x20)   | 1000       | 3.07750    | 3077.50      |

## Atmospheric Density

| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |
|-------------------------------------|------------|----------------|---------------|
| Exponential Density                 | 10000      | 0.09810    | 9.81         |
| Harris-Priester Density             | 5000       | 0.64859    | 129.72       |
| NRLMSISE-00 Density (via pymsis)    | 100        | 0.13771    | 1377.13      |

## Kalman Filters

| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |
|-------------------------------------|------------|----------------|---------------|
| KF Predict                          | 10000      | 0.06102    | 6.10         |
| KF Update                           | 10000      | 0.32221    | 32.22        |
| EKF Predict                         | 5000       | 0.03717    | 7.43         |
| EKF Update                          | 5000       | 0.19468    | 38.94        |
| UKF Predict (dim=6)                 | 1000       | 0.13138    | 131.38       |
| UKF Update (dim=6)                  | 1000       | 0.24662    | 246.62       |

## Mission Design & Guidance

| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |
|-------------------------------------|------------|----------------|---------------|
| Hohmann Transfer                    | 10000      | 0.12477    | 12.48        |
| Optimal Combined Maneuver           | 100        | 0.02150    | 215.01       |

## Coordinate Frames & Time

| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |
|-------------------------------------|------------|----------------|---------------|
| ECI to ECEF Conversion              | 5000       | 0.18926    | 37.85        |
| ECI to LLH (Iterative)              | 2000       | 0.12445    | 62.22        |
| GMST Calculation                    | 10000      | 0.01263    | 1.26         |

## Classical Control & Math

| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |
|-------------------------------------|------------|----------------|---------------|
| PID Update                          | 50000      | 0.01741    | 0.35         |
| quat_mult                           | 10000      | 0.03611    | 3.61         |
| quat_conj                           | 10000      | 0.01265    | 1.27         |
| quat_normalize                      | 10000      | 0.03842    | 3.84         |

## Attitude Determination & Integrators

| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |
|-------------------------------------|------------|----------------|---------------|
| TRIAD Determination                 | 20000      | 2.69424    | 134.71       |
| RK4 Step (3D Linear)                | 50000      | 0.64110    | 12.82        |
# OpenGNC Benchmarks

## Propagators

| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |
|-------------------------------------|------------|----------------|---------------|
| Kepler Propagator (Analytical)      | 5000       | 0.28189    | 56.38        |
| Cowell Propagator (RK4, dt=100, step=10) | 500        | 0.20519    | 410.37       |

## Gravity Models

| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |
|-------------------------------------|------------|----------------|---------------|
| J2 Gravity Acceleration             | 10000      | 0.13552    | 13.55        |
| Harmonics Gravity (EGM2008 20x20)   | 1000       | 0.08578    | 85.78        |

## Atmospheric Density

| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |
|-------------------------------------|------------|----------------|---------------|
| Exponential Density                 | 10000      | 0.10602    | 10.60        |
| Harris-Priester Density             | 5000       | 0.65042    | 130.08       |
| NRLMSISE-00 Density (via pymsis)    | 100        | 0.01372    | 137.24       |

## Kalman Filters

| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |
|-------------------------------------|------------|----------------|---------------|
| KF Predict                          | 10000      | 0.05797    | 5.80         |
| KF Update                           | 10000      | 0.35966    | 35.97        |
| EKF Predict                         | 5000       | 0.03427    | 6.85         |
| EKF Update                          | 5000       | 0.18098    | 36.20        |
| UKF Predict (dim=6)                 | 1000       | 0.13751    | 137.51       |
| UKF Update (dim=6)                  | 1000       | 0.25044    | 250.44       |

## Mission Design & Guidance

| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |
|-------------------------------------|------------|----------------|---------------|
| Hohmann Transfer                    | 10000      | 0.17460    | 17.46        |
| Optimal Combined Maneuver           | 100        | 0.01862    | 186.20       |

## Coordinate Frames & Time

| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |
|-------------------------------------|------------|----------------|---------------|
| ECI to ECEF Conversion              | 5000       | 0.19734    | 39.47        |
| ECI to LLH (Iterative)              | 2000       | 0.11494    | 57.47        |
| GMST Calculation                    | 10000      | 0.01249    | 1.25         |

## Classical Control & Math

| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |
|-------------------------------------|------------|----------------|---------------|
| PID Update                          | 50000      | 0.01839    | 0.37         |
| quat_mult                           | 10000      | 0.03629    | 3.63         |
| quat_conj                           | 10000      | 0.01218    | 1.22         |
| quat_normalize                      | 10000      | 0.03813    | 3.81         |

## Attitude Determination & Integrators

| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |
|-------------------------------------|------------|----------------|---------------|
| TRIAD Determination                 | 20000      | 2.73421    | 136.71       |
| RK4 Step (3D Linear)                | 50000      | 0.62770    | 12.55        |
