"""
Precompiled Numexpr expressions for faster computation
"""
import numpy as np
import numexpr_mod as ne


def cache_numexpr_expressions():
    """
    Caches numexpr expressions for faster computation
    :return:
    """
    # Precompiled expressions for numexpr_mod.reevaluate function
    # Creating dummy variables of necessary types
    k1, k2, k3, k4, F, n, n0, tau, sigma, dt = np.arange(10, dtype=np.float64)
    se_flux = np.arange(1, dtype=np.int64)
    ne.cache_expression("(k1+k4)/6 +(k2+k3)/3", 'rk4')
    ne.cache_expression("(F * (1 - n / n0) - n / tau - n * sigma * se_flux) * dt", 'precursor')
    ne.cache_expression("F * dt * (1 - n / n0) - n * dt / tau - n * sigma * se_flux * dt", 'precursor_temp')