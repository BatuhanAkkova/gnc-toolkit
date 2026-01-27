# GNC Toolkit

A comprehensive Guidance, Navigation, and Control (GNC) toolkit for satellite simulation and estimation. This project provides high-fidelity environment models, disturbance calculations, and advanced estimation algorithms in both Python and C++.

## Project Overview

The **GNC Toolkit** is designed to support mission analysis, simulation, and flight software development for small satellites, with a particular emphasis on Very Low Earth Orbit (VLEO) environments. It includes modular components for:

- **Environment Modeling**: Accurate calculations for atmospheric density, magnetic fields, and solar positions.
- **Disturbance Analysis**: High-fidelity models for gravity (including EGM2008 spherical harmonics), atmospheric drag, and solar radiation pressure.
- **State Estimation**: Robust filtering algorithms including EKF, MEKF, and UKF for orbit and attitude determination.
- **Cross-Platform Implementation**: Core logic implemented in Python for rapid prototyping and C++ for performance-critical applications.

## Features

### Environment & Disturbances
- **Gravity Models**: Two-body, J2, and high-fidelity EGM2008 Spherical Harmonics.
- **Atmospheric Drag**: Lumped drag models with Harris-Priester density estimation.
- **Solar Radiation Pressure**: Cannonball model with shadow effects and solar position tracking.
- **Magnetic Field**: Geomagnetic field modeling for sensor simulation.

### Estimation (Kalman Filters)
- **KF**: Standard Kalman Filter for linear systems.
- **EKF**: Extended Kalman Filter for non-linear state estimation.
- **MEKF**: Multiplicative Extended Kalman Filter for attitude estimation using quaternions.
- **UKF**: Unscented Kalman Filter for handling high degrees of non-linearity.

### Numerical Integrators
- **RK4**: Fixed step Runge-Kutta 4.
- **RK45**: Adaptive step Runge-Kutta-Fehlberg.
- **RK853**: High-order adaptive Dormand-Prince.

### Attitude Determination
- **TRIAD**: Deterministic method using two vectors.
- **QUEST**: Deterministic method using multiple vectors.

### C++ Library
- High-performance implementations of estimation filters (KF, EKF, MEKF) utilizing the Eigen library.

## TODO

### Propagators
- [ ] Cowell method (Integrate eqs of motion with disturbances)
- [ ] Fix kepler propagator

### Attitude Dynamics
- [ ] Rigid body euler equations

### Classical Control Algorithms
- [ ] PID control (add anti-windup)
- [ ] B-dot controller (magnetic detumbling)

### Optimal Control Algorithms
- [ ] LQR controller (infinite horizon, via solving ARE)
- [ ] LQE controller (duality of LQR for estimation)
- [ ] Sliding mode controller
- [ ] MPC
- [ ] Feedback Linearization (cancel out non-linear terms)

### C++ Library
- [ ] Develop cpp library further

### Guidance & Mission Analysis Tools
- [ ] Orbital Manouvers
    - [ ] Hohmann Transfer
    - [ ] Bi-Elliptic Transfer
    - [ ] Phasing Manouvers
    - [ ] Plane change manouvers
- [ ] Intercept & Rendezvous
    - [ ] Lambert Problem
    - [ ] CW eqs (Hill's equations, linearized relative motion)

### Actuator & Sensor Models
- [ ] Sensors
    - [ ] Star Tracker
    - [ ] Sun Sensor
    - [ ] Magnetometer
    - [ ] Gyroscope
- [ ] Actuators
    - [ ] RWs
    - [ ] MTQs
    - [ ] Thrusters (PWM logic for on/off thrusters)
