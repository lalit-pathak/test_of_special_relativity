"""
Some utility functions to peform the PE analysis
Author: Lalit Pathak (lalit.pathak@iitgn.ac.in)

"""
import numpy as np
from scipy.special import hyp2f1
from scipy.special import erf, erfinv
from scipy.interpolate import interp1d

def cdf_param(mass_ratio):
    
    """
    Function to calculate cdf for a given value of mass-ratio
    
    Parameters
    ----------
    mass_ratio: float
    
    Returns
    --------
    cdf evaluated at mass_ratio
    
    """
    cdf = -5. * mass_ratio**(-1./5) * hyp2f1(-2./5, -1./5, 4./5, -mass_ratio)
    
    return cdf
        
def cdfinv_q(mass_ratio_min, mass_ratio_max, value):
    
    """
    Function to generate random mass_ratio sample from a distribtion which is uniform in component masses and constrainted
    by mass_ratio
    
    Parameter
    ---------
    mass_ratio_min: minimum value of the mass_ratio
    mass_ratio_max: maximum value of the mass_ratio
    value: a number between 0 and 1
    
    Returns
    --------
    mass_ratio: sampled mass_ratio
    
    """

    q_array = np.linspace(mass_ratio_min, mass_ratio_max, num=1000, endpoint=True)
    q_invcdf_interp = interp1d(cdf_param(q_array),
                               q_array, kind='cubic',
                               bounds_error=True)
    
    mass_ratio = q_invcdf_interp((cdf_param(mass_ratio_max) - cdf_param(mass_ratio_min)) * value + cdf_param(mass_ratio_min))
    
    return mass_ratio

def normalcdf(mu, var, k):
    """The CDF of the normal distribution, without bounds."""    
    return 0.5*(1. + erf((k - mu)/(np.sqrt(2*var))))

def normalcdfinv(mu, var, val):
    """The inverse CDF of the normal distribution, without bounds."""
    return mu + np.sqrt(2*var) * erfinv(2*val - 1.)

def cdfinv_k(mu, var, k_min, k_max, val):
    """Return inverse of the CDF.
    """
    a, b = k_min, k_max
    
    if a != -np.inf:
        
        phi_a = normalcdf(mu, var, a)
        
    else:
        
        phi_a = 0.
        
    if b != np.inf:
        
        phi_b = normalcdf(mu, var, b)
        
    else:
        
        phi_b = 1.
        
    adjusted_p = phi_a + val * (phi_b - phi_a)
    
    return normalcdfinv(mu, var, adjusted_p)