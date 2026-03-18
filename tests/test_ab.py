import numpy as np
import pytest

# Predictor Coefficients (Denominator 120960)
p_coeffs = np.array([434241, -1152169, 2183877, -2664477, 2102243, -1041723, 295767, -36799]) / 120960

# Corrector Coefficients (Denominator 120960)
c_coeffs = np.array([36799, 139849, -121797, 123133, -88547, 41499, -11351, 1375]) / 120960

def test_ab_moulton():
    mu = 398600.4418e9
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(mu / 7000e3), 0.0])
    y0 = np.concatenate([r0, v0])
    
    def f(t, y):
        r = y[:3]
        v = y[3:]
        r_mag = np.linalg.norm(r)
        a = -mu / (r_mag**3) * r
        return np.concatenate([v, a])

    t_span = (0, 1000.0)
    h = 10.0
    
    # Fill history using analytical solution for circular orbit
    omega = np.sqrt(mu / 7000e3**3)
    history_dy = [] # dy/dt
    t_values = []
    y_values = []
    
    for k in range(8):
        t_k = k * h
        rk = np.array([7000e3 * np.cos(omega * t_k), 7000e3 * np.sin(omega * t_k), 0.0])
        vk = np.array([-7000e3 * omega * np.sin(omega * t_k), 7000e3 * omega * np.cos(omega * t_k), 0.0])
        history_dy.append(np.concatenate([vk, -mu / (7000e3**3) * rk]))
        if k == 7:
            curr_y = np.concatenate([rk, vk])
            curr_t = t_k

    # Loop
    while curr_t < t_span[1]:
        # Predictor
        sum_p = np.zeros(6)
        for j in range(8):
            sum_p += p_coeffs[j] * history_dy[-(j+1)]
        
        y_p = curr_y + h * sum_p
        next_t = curr_t + h
        dy_p = f(next_t, y_p)
        
        # Corrector
        temp_history_dy = history_dy + [dy_p]
        sum_c = np.zeros(6)
        for j in range(8):
            sum_c += c_coeffs[j] * temp_history_dy[-(j+1)]
            
        y_c = curr_y + h * sum_c
        dy_c = f(next_t, y_c)
        
        # Update
        curr_y = y_c
        curr_t = next_t
        history_dy.append(dy_c)
        if len(history_dy) > 8:
            history_dy.pop(0)

    # Compare with analytical
    r_exact = np.array([7000e3 * np.cos(omega * curr_t), 7000e3 * np.sin(omega * curr_t), 0.0])
    error = np.linalg.norm(curr_y[:3] - r_exact)
    print(f"AB Position Error: {error} m")
    assert error < 1e-3

if __name__ == "__main__":
    test_ab_moulton()
