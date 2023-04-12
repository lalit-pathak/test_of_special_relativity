"""
Some utility functions to peform the PE analysis
Author: Lalit Pathak (lalit.pathak@iitgn.ac.in)

"""
import numpy as np
from scipy.special import hyp2f1
from scipy.interpolate import interp1d
from scipy.stats import cauchy, norm
from scipy.special import erf, erfinv

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

def cauchy_cdf(k, k0, gamma):
    
    return 0.5 + (1/np.pi) * np.arctan((k-k0)/gamma)

def cdfinv_k(k0, gamma, k_min, k_max, value):

    k_array = np.linspace(k_min, k_max, num=1000, endpoint=True)
    k_invcdf_interp = interp1d(cauchy_cdf(k_array, k0, gamma),
                               k_array, kind='cubic',
                               bounds_error=True)
    
    k = k_invcdf_interp((cauchy_cdf(k_max, k0, gamma) - cauchy_cdf(k_min, k0, gamma)) * value + cauchy_cdf(k_min, k0, gamma))
    
    return k
