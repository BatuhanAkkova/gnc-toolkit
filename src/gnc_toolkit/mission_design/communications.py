"""
Link budget calculations and RF analysis tools.
"""

import numpy as np

# Speed of light in m/s
C = 299792458.0


def calculate_friis_link_budget(
    p_tx_w, g_tx_db, g_rx_db, frequency_hz, distance_m, losses_misc_db=0.0, l_atm_db=0.0
):
    """
    Calculates the link budget using Friis transmission equation.

    Formula:
        Pr (dBW) = Pt (dBW) + Gt (dB) + Gr (dB) - Lfs (dB) - Latm (dB) - Lmisc (dB)
        Lfs = 20 * log10(d) + 20 * log10(f) + 20 * log10(4*pi/c)

    Args:
        p_tx_w (float): Transmitter power [W].
        g_tx_db (float): Transmitter antenna gain [dB].
        g_rx_db (float): Receiver antenna gain [dB].
        frequency_hz (float): Carrier frequency [Hz].
        distance_m (float): Distance between Tx and Rx [m].
        losses_misc_db (float, optional): Miscellaneous losses (pointing, line, etc.) [dB]. Default 0.
        l_atm_db (float, optional): Atmospheric attenuation [dB]. Default 0.

    Returns
    -------
        dict: Containing:
            'p_rx_dbw' (float): Received power [dBW].
            'p_rx_w' (float): Received power [W].
            'l_fs_db' (float): Free space path loss [dB].
    """
    if p_tx_w <= 0:
        raise ValueError("Transmitter power must be strictly positive.")
    if distance_m <= 0:
        raise ValueError("Distance must be strictly positive.")
    if frequency_hz <= 0:
        raise ValueError("Frequency must be strictly positive.")

    p_tx_dbw = 10 * np.log10(p_tx_w)

    # Lfs = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c),  20*log10(4*pi/c) ≈ -147.554
    l_fs_db = 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) - 147.554
    p_rx_dbw = p_tx_dbw + g_tx_db + g_rx_db - l_fs_db - l_atm_db - losses_misc_db
    p_rx_w = 10 ** (p_rx_dbw / 10.0)

    return {"p_rx_dbw": p_rx_dbw, "p_rx_w": p_rx_w, "l_fs_db": l_fs_db}


def calculate_doppler_shift(f_tx_hz, r_ecef_rx, v_ecef_rx, r_ecef_tx, v_ecef_tx):
    """
    Calculates the Doppler shift for a signal sent from TX to RX.

    Formula:
        delta_f = - f_tx * (v_rel / c)
        where v_rel = dot(r_rel, v_rel_vec) / |r_rel|
        r_rel = r_rx - r_tx
        v_rel_vec = v_rx - v_tx

    Args:
        f_tx_hz (float): Transmitter frequency [Hz].
        r_ecef_rx (np.ndarray): Receiver position in ECEF [m], shape (3,).
        v_ecef_rx (np.ndarray): Receiver velocity in ECEF [m/s], shape (3,).
        r_ecef_tx (np.ndarray): Transmitter position in ECEF [m], shape (3,).
        v_ecef_tx (np.ndarray): Transmitter velocity in ECEF [m/s], shape (3,).

    Returns
    -------
        dict: Containing:
            'f_rx_hz' (float): Received frequency [Hz].
            'doppler_shift_hz' (float): Doppler shift [Hz].
    """
    r_rx = np.array(r_ecef_rx)
    v_rx = np.array(v_ecef_rx)
    r_tx = np.array(r_ecef_tx)
    v_tx = np.array(v_ecef_tx)

    if r_rx.shape != (3,) or v_rx.shape != (3,) or r_tx.shape != (3,) or v_tx.shape != (3,):
        raise ValueError("Positions and velocities must be vectors of length 3.")

    r_rel = r_rx - r_tx
    v_rel_vec = v_rx - v_tx
    distance = np.linalg.norm(r_rel)
    if distance == 0:
        return {"f_rx_hz": f_tx_hz, "doppler_shift_hz": 0.0}

    v_rel = np.dot(r_rel, v_rel_vec) / distance  # range rate
    doppler_shift_hz = -f_tx_hz * (v_rel / C)
    f_rx_hz = f_tx_hz + doppler_shift_hz

    return {"f_rx_hz": f_rx_hz, "doppler_shift_hz": doppler_shift_hz}


def calculate_atmospheric_attenuation(elevation_deg, frequency_hz):
    """
    Calculates atmospheric attenuation using a simplified cosecant model.

    Formula:
        L_atm = A_zenith / sin(elevation)
        where A_zenith is a base attenuation dependent on frequency band.

    Args:
        elevation_deg (float): Elevation angle [deg].
        frequency_hz (float): Frequency [Hz].

    Returns
    -------
        float: Atmospheric attenuation [dB].
    """
    # Zenith attenuation by frequency band (ballpark values)
    if frequency_hz < 3e9:  # S-band
        a_zenith = 0.03
    elif frequency_hz < 10e9:  # C/X-band
        a_zenith = 0.05
    elif frequency_hz < 18e9:  # Ku-band
        a_zenith = 0.15
    else:  # Ka-band and above
        a_zenith = 0.35

    # Cosecant model valid for elevation > 5 deg; clamp to avoid division singularity
    elevation_rad = np.radians(max(elevation_deg, 5.0))
    l_atm_db = a_zenith / np.sin(elevation_rad)
    return l_atm_db
