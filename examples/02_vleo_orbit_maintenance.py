"""
VLEO Orbit Maintenance Example
==============================

This script demonstrates an orbit maintenance scenario for a satellite in Very Low Earth Orbit (VLEO).

Scenario:
    - Satellite in 250 km circular orbit.
    - Perturbations: Atmospheric Drag (Harris-Priester Model) + J2 Gravity.
    - Guidance: Maintain altitude within +/- 100m of target.
    - Control: Continuous low-thrust propulsion (or pulse-modulated) to counteract drag.

Dynamics:
    - Cowell Propagator (Numerical Integration).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from gnc_toolkit.propagators.cowell import CowellPropagator
from gnc_toolkit.environment.density import HarrisPriester

from gnc_toolkit.disturbances.gravity import J2Gravity
from gnc_toolkit.disturbances.drag import LumpedDrag
from gnc_toolkit.actuators.thruster import ElectricThruster
from gnc_toolkit.utils.time_utils import calc_jd

# Helper for SMA calculation if utility missing
def get_sma(r, v, mu):
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    energy = (v_mag**2)/2 - mu/r_mag
    sma = -mu / (2*energy)
    return sma

def simulation():
    # -------------------------------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------------------------------
    mu = 3.986004418e14 # m^3/s^2
    Re = 6378137.0 # m
    
    # Target Orbit
    h_target = 250000.0 # 250 km
    r_target = Re + h_target
    
    # J2 corrected circular velocity (Equatorial)
    j2_val = 1.082635855e-3
    v_circ = np.sqrt(mu / r_target * (1 + 1.5 * j2_val * (Re/r_target)**2))

    # Satellite Properties
    mass = 100.0 # kg
    area = 1.0 # m^2
    Cd = 2.2 # Drag Coeff
    
    # Actuator
    thruster = ElectricThruster(max_thrust=0.02, isp=3000.0) # 20mN, 3000s Isp
    
    # Simulation Settings
    dt = 1.0 # Step size [s]
    duration = 5 * 5400 # ~5 Orbits
    
    # Initial State
    r0 = np.array([r_target, 0.0, 0.0])
    v0 = np.array([0.0, v_circ, 0.0]) 
    
    # Environment
    density_model = HarrisPriester()
    j2_model = J2Gravity(mu=mu, re=Re)
    jd_epoch = 2460000.0 
    drag_model = LumpedDrag(density_model)

    # Controller Settings
    # We maintain Mean SMA within a specific energy band
    deadband_low = 10.0 # m (Start thrusting if SMA drops 10m below target)
    deadband_high = 0.0 # m (Stop thrusting when SMA returns to target)
    
    # Values to record
    history = {
        't': [],
        'alt': [],
        'thrust': [],
        'rho': []
    }
    
    # State for Controller
    class SimState:
        fuel_used = 0.0
        thruster_on = False
    
    sim_state = SimState()
    
    # Calculate Reference SMA from Initial Condition (J2 Corrected)
    sma_ref = get_sma(r0, v0, mu)

    def perturbation_acc(t, r, v):
        r_mag = np.linalg.norm(r)
        
        # 1. Atmospheric Drag
        jd_curr = jd_epoch + t / 86400.0
        
        # Acceleration
        a_drag = drag_model.get_acceleration(r, v, jd_curr, mass, area, Cd)
        
        # 2. J2 Perturbation
        # CowellPropagator adds TwoBody. So we need ONLY J2 perturbation.
        a_j2_full = j2_model.get_acceleration(r)
        a_kep = -mu / (r_mag**3) * r
        a_j2_pert = a_j2_full - a_kep
        
        # 3. Control Logic (SMA Hysteresis)
        sma_curr = get_sma(r, v, mu)
        sma_err = sma_ref - sma_curr # Positive if SMA is lower than target
        
        # Hysteresis
        if sma_err > deadband_low:
            sim_state.thruster_on = True
        elif sma_err <= deadband_high:
            sim_state.thruster_on = False
            
        a_thrust = np.zeros(3)
        if sim_state.thruster_on:
            # Deliver thrust through actuator model
            thrust_val = thruster.command(0.02) # Command full thrust
            a_thrust = (thrust_val / mass) * (v / np.linalg.norm(v))
            
            # Fuel consumption
            sim_state.fuel_used += thruster.get_mass_flow(thrust_val) * dt
            
        return a_drag + a_j2_pert + a_thrust

    # -------------------------------------------------------------------------
    # 3. Propagation Loop
    # -------------------------------------------------------------------------
    propagator = CowellPropagator(mu=mu)
    
    t = 0.0
    r, v = r0, v0
    
    print(f"Starting VLEO Simulation. Target Alt: {h_target/1000} km. Drag Area: {area} m^2")
    
    while t < duration:
        # Propagate one step
        r_next, v_next = propagator.propagate(r, v, dt, perturbation_acc_fn=perturbation_acc)
        
        # Record Data
        jd_curr = jd_epoch + t / 86400.0
        rho = density_model.get_density(r, jd_curr)
        h_curr = np.linalg.norm(r) - Re
        
        history['t'].append(t)
        history['alt'].append(h_curr)
        history['rho'].append(rho)
        
        # Power calculation for recording
        p_cons = thruster.get_power_consumption(0.02) if sim_state.thruster_on else 0.0
        history['thrust'].append(p_cons)
        
        # Update
        r, v = r_next, v_next
        t += dt
        
        if t % 1000 == 0:
            print(f"Time: {t:.0f}s | Alt: {h_curr/1000:.3f} km | Thrust: {sim_state.thruster_on}")

    # -------------------------------------------------------------------------
    # 4. Visualization
    # -------------------------------------------------------------------------
    t_arr = np.array(history['t'])
    alt_arr = np.array(history['alt']) / 1000.0
    pwr_arr = np.array(history['thrust'])
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Altitude
    ax[0].plot(t_arr, alt_arr, 'b')
    ax[0].axhline(h_target/1000.0, color='k', linestyle='--', alpha=0.5, label='Target')
    ax[0].set_ylabel('Altitude [km]')
    ax[0].set_title(f'VLEO Orbit Maintenance ({h_target/1000} km)')
    ax[0].legend()
    ax[0].grid(True)
    
    # Density
    ax[1].plot(t_arr, np.array(history['rho']), 'r')
    ax[1].set_ylabel('Density [kg/m^3]')
    ax[1].grid(True)
    
    # Power Consumption
    ax[2].plot(t_arr, pwr_arr, 'g')
    ax[2].set_ylabel('Electric Power [W]')
    ax[2].set_xlabel('Time [s]')
    ax[2].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulation()
