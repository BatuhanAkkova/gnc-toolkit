"""
Porkchop plot grid generation for interplanetary transfers.
"""

import numpy as np
from ..guidance.rendezvous import solve_lambert

def generate_porkchop_grid(departure_dates, arrival_dates, r_dep_func, v_dep_func, r_arr_func, v_arr_func, mu=398600.4418):
    """
    Generates a grid of C3 and V_infinity values for a range of departure and arrival dates.

    Args:
        departure_dates (array-like): Array of departure times (e.g., seconds from epoch).
        arrival_dates (array-like): Array of arrival times (e.g., seconds from epoch).
        r_dep_func (callable): Function that returns position vector at departure time t.
        v_dep_func (callable): Function that returns velocity vector at departure time t.
        r_arr_func (callable): Function that returns position vector at arrival time t.
        v_arr_func (callable): Function that returns velocity vector at arrival time t.
        mu (float): Gravitational parameter.

    Returns:
        dict: Containing grids for 'c3', 'v_inf_arr', 'tof', and the input axes.
    """
    n_dep = len(departure_dates)
    n_arr = len(arrival_dates)

    c3_grid = np.full((n_arr, n_dep), np.nan)
    v_inf_arr_grid = np.full((n_arr, n_dep), np.nan)
    tof_grid = np.full((n_arr, n_dep), np.nan)

    for i, t_dep in enumerate(departure_dates):
        r_dep = r_dep_func(t_dep)
        v_dep_planet = v_dep_func(t_dep)
        
        for j, t_arr in enumerate(arrival_dates):
            dt = t_arr - t_dep
            if dt <= 0:
                continue
            
            r_arr = r_arr_func(t_arr)
            v_arr_planet = v_arr_func(t_arr)

            try:
                # Solve Lambert problem
                v_dep_sc, v_arr_sc = solve_lambert(r_dep, r_arr, dt, mu=mu)
                
                # V_infinity at departure
                v_inf_dep = v_dep_sc - v_dep_planet
                c3 = np.dot(v_inf_dep, v_inf_dep)
                
                # V_infinity at arrival
                v_inf_arr = np.linalg.norm(v_arr_sc - v_arr_planet)
                
                c3_grid[j, i] = c3
                v_inf_arr_grid[j, i] = v_inf_arr
                tof_grid[j, i] = dt
                
            except Exception:
                continue

    return {
        'departure_dates': departure_dates,
        'arrival_dates': arrival_dates,
        'c3': c3_grid,
        'v_inf_arr': v_inf_arr_grid,
        'tof': tof_grid
    }
