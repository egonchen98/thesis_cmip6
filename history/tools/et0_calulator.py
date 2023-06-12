import math
import sympy as sym
import time
import numpy as np


def cal_et0_pm(params: dict):
    """Calculate et0 in PM formula with parameters
    Rn: MJ/m2/day
    G: MJ/m2/day
    T: oC
    u2: m/s
    es: kPa
    ea: kPa
    Delta: kPa/oC
    gamma: Kpa/oC

    :return: ET0: mm/day
    """
    Delta, Rn, G, gamma, T, u2, es, ea = params['Delta'], params['Rn'], params['G'], params['gamma'], \
        params['T'], params['u2'], params['es'], params['ea']
    frac1 = 0.408 * Delta * (Rn - G) + gamma * 900 / (T + 273) * u2 * (es - ea)
    frac2 = Delta + gamma * (1 + 0.34 * u2)
    return frac1/frac2


def get_et0_station(params: dict):
    """
    latitude: (degree)
    altitude: m
    Tmax: oC
    Tmin: oC
    Tmean: oC
    delta_Tmean: oC
    Wind: m/s
    sunhour: h
    RH: %
    cdays: day
    reuturn: Delta, Rn, G, gamma, T, u2, es, ea
    """
    pi = 3.14159

    # T, u2
    T = params['Tmean']
    u2 = params ['u2']

    # es, ea
    es_0= 0.6108 * math.exp(17.27 * T / (T+ 237.3))  # 饱合水汽压
    es_max = 0.6108 * math.exp(17.27 * params['Tmax'] / (params['Tmax'] + 237.3))  # 饱合水汽压
    es_min = 0.6108 * math.exp(17.27 * params['Tmin'] / (params['Tmin'] + 237.3))  # 饱合水汽压
    RHmin = es_min/es_max * 100
    es = (es_min + es_max) / 2
    ea = es * params['RH'] / 100

    # Delta
    Delta = 4098 * es_0 / (T+237.3)**2

    # G
    G = 0.21 * params['delta_Tmean']

    # Rn = Rns - Rnl
    # Rns
    dr = 1 + 0.033*math.cos(2*pi/356*params['cdays'])
    delta = 0.409*math.sin(2*pi/365*params['cdays'] - 1.39)
    phi = params['latitude'] * pi / 180
    omegas = math.acos( -math.tan(phi) * math.tan(delta))
    Ra = 24*60/pi * 0.0820 * dr * (omegas*math.sin(phi)*math.sin(delta) + math.cos(phi)*math.cos(delta)*math.sin(omegas))
    N = 24/pi*omegas  # max possible duration or daylight hours
    Rs = (0.25 + 0.5*params['sunhour']/N) * Ra
    Rso = (0.75 + 2*1e-5*params['altitude']) * Ra  # TODO 这么计算的原因？
    Rns = (1-0.23) * Rs
    # Rnl
    Rnl = 4.903e-9 * ((params['Tmax']+273.16)**4 + (params['Tmin']+273.16)**4)/2 * (0.34-0.14*ea**0.5) * (1.35*Rs/Rso-0.35)

    Rn = Rns - Rnl

    # psychrometric constant (gamma)
    if 'pressure' not in params:
        pressure = 101.3 * ((293 - 6.5e-3 * params['altitude']) / 293) ** 5.26
    else:
        pressure = params['pressure']
    gamma = pressure * 0.665e-3

    pm_params = {'Delta': Delta, 'gamma': gamma, 'Rn': Rn, 'G': G, 'u2': u2, 'T': T, 'es': es, 'ea': ea, 'Rs' :Rs, 'Rso': Rso, 'Rnl': Rnl}
    et0 = cal_et0_pm(pm_params)
    # res = pm_params | {'et0': et0}
    # res = [i for i in res.values()]
    return [RHmin, et0]

