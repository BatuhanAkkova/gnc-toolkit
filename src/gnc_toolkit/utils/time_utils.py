import numpy as np
from datetime import datetime, timedelta
from typing import Tuple

def calc_jd(year, month, day, hour=0, minute=0, sec=0, leap_sec=False) -> tuple[float,float]:
    """Calculates Julian Date from date and universal time."""
    if month <= 2:
        year -= 1
        month += 12

    a = int(year / 100)
    b = 2 - a + int(a / 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
    jdfrac = (hour * 3600 + minute * 60 + sec) / 86400
    if jdfrac >= 1:
        jd += int(jdfrac)
        jdfrac %= 1
    return jd, jdfrac

def jd_to_datetime(jd, jdfrac) -> tuple[int, int, int, int, int, float]:
    """Calculates date and time from Julian Date."""
    jd = jd + jdfrac
    
    jd_i = int(jd + 0.5)
    df = jd + 0.5 - jd_i
    if df < 0:
        df += 1.0
    
    l = jd_i + 68569
    n = 4 * l // 146097
    l = l - (146097 * n + 3) // 4
    i = 4000 * (l + 1) // 1461001
    l = l - 1461 * i // 4 + 31
    j = 80 * l // 2447
    day = l - 2447 * j // 80
    l = j // 11
    month = j + 2 - 12 * l
    year = 100 * (n - 49) + i + l
    
    df *= 86400
    hour = int(df / 3600)
    df -= hour * 3600
    minute = int(df / 60)
    df -= minute * 60
    second = df
    
    return year, month, day, hour, minute, second

def day_to_mdtime(year, days) -> tuple[int, int, int, int, float]:
    """Calculates month, day, hour, minute, second from day and year."""
    len_month = [0] * 13
    for i in range(1,13):
        if i == 2: len_month[i] = 28
        elif i == 4 or i == 6 or i == 9 or i == 11:
            len_month[i] = 30
        else: len_month[i] = 31
    
    doy = np.floor(days) # Day of year

    if np.remainder(year - 1900, 4) == 0: # Feb in every 4 years
        len_month[2] = 29
    
    i = 1
    i_temp = 0
    while (doy > i_temp + len_month[i]) and i < 12:
        i_temp += len_month[i]
        i += 1
    month = i
    day = doy - i_temp

    hour = np.fix((days - doy)*24.0)
    minute = np.fix(((days - doy)*24.0 - hour) * 60.0)
    second = (((days - doy)*24.0 - hour) * 60.0 - minute) * 60.0

    return month, day, hour, minute, second

def calc_gmst(jd, dut1=0):
    """Calculates Greenwich Mean Sidereal Time from Julian Date and DUT1."""
    jd_ut1 = jd + dut1/86400.0
    ut1 = (jd_ut1 - 2451545.0) / 36525.0
    gmst = (67310.54841 + (876600*3600.0 + 8640184.812866)*ut1 + 0.093104*ut1**2 - 6.2E-6*ut1**3) % 86400.0
    gmst /= 240.0
    if gmst < 0: gmst+=360.0
    return np.radians(gmst)

def calc_last(jd, lon, dut1=0):
    """Calculates Local Apparent Sidereal Time from Julian Date and Longitude."""
    ut1 = (jd - 2451545.0) / 36525.0
    gmst = calc_gmst(jd + dut1/86400.0)
    
    eps = np.radians(23.439291 - 0.0130042*ut1) # Obliquity of ecliptic
    omega = np.radians(125.04452 - 1934.136261*ut1) # Nutation in Longitude of the ascending node of the Moon's orbit
    l_sun = np.radians(280.4665 + 36000.7698*ut1 + 0.0003032*ut1**2) # Mean longitude of the Sun
    d_psi = np.radians(-17.20*np.sin(omega) - 1.32*np.sin(2*l_sun) + 0.216*np.sin(2*eps)) / 3600.0 # Nutation in Longitude

    equinox = d_psi * np.cos(eps)
    # GAST = GMST + equinox
    last = gmst + equinox + lon
    last = last % (2*np.pi)
    return last
    
def calc_lst(gmst, lon, dut1=0):
    """Calculates Local Sidereal Time from GMST and Longitude."""
    return gmst + lon + dut1/86400.0

def calc_doy(year, month, day):
    """Calculates the day of the year."""
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if (is_leap_year(year)): months[1] = 29
    idx = month - 1
    doy = np.sum(months[:idx]) + day
    return doy

def is_leap_year(year):
    """Checks if a year is a leap year."""
    if (np.remainder(year,4) != 0):
        return False
    else:
        if (np.remainder(year, 100) == 0):
            if (np.remainder(year, 400) == 0):
                return True
            else:
                return False
        else:
            return True

def convert_time(year, month, day, hour, minute, sec, timezone, output, dut1, dat
) -> tuple[float,float,float,float,float,float,float,float,float,float,float,float,float,float]:
    """Converts UTC to any time system."""
    local_hour = timezone + hour
    utc = hour*3600 + minute*60 + sec

    ut1 = utc + dut1
    hour_temp = int(ut1/3600)
    minute_temp = int((ut1/3600 - hour_temp)*60)
    second_temp = ((ut1/3600 - hour_temp) - minute_temp/60) * 3600
    second_temp = round(second_temp) if abs(second_temp-round(second_temp)) < 1e-10 else second_temp
    jdut1, jdut1frac = calc_jd(year, month, day, hour_temp, minute_temp, second_temp)
    tut1 = (jdut1 + jdut1frac - 2451545) / 36525

    tai = utc + dat

    gps = tai - 19

    tt = tai + 32.184
    hour_temp = int(tt/3600)
    minute_temp = int((tt/3600 - hour_temp)*60)
    second_temp = ((tt/3600 - hour_temp) - minute_temp/60) * 3600
    second_temp = round(second_temp) if abs(second_temp-round(second_temp)) < 1e-10 else second_temp
    jdtt, jdttfrac = calc_jd(year, month, day, hour_temp, minute_temp, second_temp)
    tutt = (jdtt + jdttfrac - 2451545) / 36525
    ttt = tutt # define ttt for use in TDB calc

    tdb = (
        tt
        + 0.001657 * np.sin(628.3076 * ttt + 6.2401)
        + 0.000022 * np.sin(575.3385 * ttt + 4.297)
        + 0.000014 * np.sin(1256.6152 * ttt + 6.1969)
        + 0.000005 * np.sin(606.9777 * ttt + 4.0212)
        + 0.000005 * np.sin(52.9691 * ttt + 0.4444)
        + 0.000002 * np.sin(21.3299 * ttt + 5.5431)
        + 0.00001 * ttt * np.sin(628.3076 * ttt + 4.249)
    )
    hour_temp = int(tdb/3600)
    minute_temp = int((tdb/3600 - hour_temp)*60)
    second_temp = ((tdb/3600 - hour_temp) - minute_temp/60) * 3600
    second_temp = round(second_temp) if abs(second_temp-round(second_temp)) < 1e-10 else second_temp
    jdtdb, jdtdbfrac = calc_jd(year, month, day, hour_temp, minute_temp, second_temp)
    ttdb = (jdtdb + jdtdbfrac - 2451545) / 36525

    return (
        ut1,
        tut1,
        jdut1,
        jdut1frac,
        utc,
        tai,
        gps,
        tt,
        ttt,
        jdtt,
        jdttfrac,
        tdb,
        ttdb,
        jdtdb,
        jdtdbfrac,
    )

