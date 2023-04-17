import numpy as np

from scipy.special import hyp2f1
from scipy.special import erf, erfinv
from scipy.interpolate import interp1d

def cdf_param_q(mass_ratio):

    return -5. * mass_ratio**(-1./5) * hyp2f1(-2./5, -1./5, 4./5, -mass_ratio)
        
def cdfinv_q(mass_ratio_min, mass_ratio_max, value):

    q_array = np.linspace(mass_ratio_min, mass_ratio_max, num=1000, endpoint=True)
    q_invcdf_interp = interp1d(cdf_param_q(q_array),
                               q_array, kind='cubic',
                               bounds_error=True)
    
    return q_invcdf_interp((cdf_param_q(mass_ratio_max) - cdf_param_q(mass_ratio_min)) * value + cdf_param_q(mass_ratio_min))

def normalcdf(mu, var, d_L):
    """The CDF of the normal distribution, without bounds."""
    return 0.5*(1. + erf((d_L - mu)/(2*var)**0.5))

def cdf(mu, var, d_L_min, d_L_max, d_L):
    """Returns the CDF of the given parameter value."""
    a, b = d_L_min, d_L_max
    if a != -numpy.inf:
        phi_a = normalcdf(mu, var, a)
    else:
        phi_a = 0.
    if b != numpy.inf:
        phi_b = normalcdf(mu, var, b)
    else:
        phi_b = 1.
    phi_x = normalcdf(mu, var, d_L)
    
    return (phi_x - phi_a)/(phi_b - phi_a)

def normalcdfinv(mu, var, val):
    """The inverse CDF of the normal distribution, without bounds."""
    return mu + (2*var)**0.5 * erfinv(2*val - 1.)

def cdfinv_dL(mu, var, d_L_min, d_L_max, val):
    """Return inverse of the CDF.
    """
    a, b = d_L_min, d_L_max
    if a != -np.inf:
        phi_a = normalcdf(mu, var, d_L_min)
    else:
        phi_a = 0.
    if b != np.inf:
        phi_b = normalcdf(mu, var, d_L_max)
    else:
        phi_b = 1.
    adjusted_p = phi_a + val * (phi_b - phi_a)
    
    return normalcdfinv(mu, var, adjusted_p)