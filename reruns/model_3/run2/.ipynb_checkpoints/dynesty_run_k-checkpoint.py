"""Script to run PE run using dynesty sampler for modified TaylorF2 model with a gaussian prior over distance (from EM counterpart)
Author: Lalit Pathak(lalit.pathak@iitgn.ac.in)"""

import dynesty
import numpy as np

import h5py
import time
import multiprocessing as mp

import pycbc
from numpy.random import Generator, PCG64
from pycbc.catalog import Merger
from pycbc.frame import read_frame
from pycbc.detector import Detector
from pycbc.pnutils import f_SchwarzISCO
from pycbc.psd import interpolate, welch
from pycbc.filter import highpass, matched_filter
from pycbc.types import FrequencySeries,TimeSeries
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta
from pycbc.inference.models import MarginalizedPhaseGaussianNoise, GaussianNoise
from pycbc.waveform.generator import (FDomainDetFrameGenerator, FDomainCBCGenerator)
from utils import cdfinv_q, cdfinv_dL

merger = Merger("GW170817")
ifos = ['L1', 'H1', 'V1'] # defining a list of interferometers
fLow = 20 # seismic cutoff frequency
whichfrac = float(input('fraction of fISCO: ')) # fraction of fISCO corresponding to M_map value (in Hz)
M_map = 2.76 # MAP total-mass value taken from bilby samples 
fHigh = whichfrac*f_SchwarzISCO(M_map) # high cutoff frequency

strain, stilde = {}, {}
low_frequency_cutoff = {}
high_frequency_cutoff = {}

for ifo in ifos:
    
    low_frequency_cutoff[ifo] = fLow
    high_frequency_cutoff[ifo] = fHigh
    
#-- reading GW170817 data ---
#-- We use 360 seconds open archival GW170817 data(containing the trigger)from GWOSC ---
#-- Using PyCBC utilities to perform some cleaning jobs on the raw data ---

for ifo in ifos:
    
    ts = read_frame("../../{}-{}_LOSC_CLN_4_V1-1187007040-2048.gwf".format(ifo[0], ifo),
                    '{}:LOSC-STRAIN'.format(ifo),
                   start_time=merger.time - 342,   
                   end_time=merger.time + 30,     
                   check_integrity=False)
    
    # Read the detector data and remove low frequency content
    strain[ifo] = highpass(ts, 18, filter_order=4)
    
    # Remove time corrupted by the high pass filter
    strain[ifo] = strain[ifo].crop(6,6)

    # Also create a frequency domain version of the data
    stilde[ifo] = strain[ifo].to_frequencyseries()

#-- calculating psds ---
psds = {}

for ifo in ifos:
    # Calculate a psd from the data. We'll use 2s segments in a median - welch style estimate
    # We then interpolate the PSD to the desired frequency step. 
    psds[ifo] = interpolate(strain[ifo].psd(2), stilde[ifo].delta_f)

    # We explicitly control how much data will be corrupted by overwhitening the data later on
    # In this case we choose 2 seconds.
    psds[ifo] = inverse_spectrum_truncation(psds[ifo], int(2 * strain[ifo].sample_rate),
                                    low_frequency_cutoff=low_frequency_cutoff[ifo], trunc_method='hann')

#-- link: https://dcc.ligo.org/LIGO-P1800370/public ---
#-- paper link: https://arxiv.org/pdf/1805.11579.pdf ---
#-- median values taken from the GW170817_GWTC-1.hdf5 files (low-spin) ---
approximant = 'TaylorF2_full'
ra = 3.44616 #radian
dec = -0.408084 #radian 

#-- setting fixed parameters and factors ---
static_params = {'approximant': approximant, 'f_lower': fLow, 'f_higher': fHigh, 'ra': ra, 'dec': dec}

variable_params = ['mass1', 'mass2', 'spin1z', 'spin2z', 'k', 'inclination', 'distance', 'tc', 'polarization']

model = MarginalizedPhaseGaussianNoise(variable_params, stilde, low_frequency_cutoff, \
                                              psds=psds, high_frequency_cutoff=high_frequency_cutoff, static_params=static_params)

#-- defining loglikelihood function ---
def pycbc_log_likelihood(query):
    
    mchirp, mass_ratio, s1z, s2z, k, iota, distance, pol, tc = query
    m1 = mass1_from_mchirp_q(mchirp, mass_ratio)
    m2 = mass2_from_mchirp_q(mchirp, mass_ratio)
    
    model.update(mass1=m1, mass2=m2, spin1z=s1z, spin2z=s2z, k=k, inclination=iota, distance=distance, polarization=pol, tc=tc)

    return model.loglr

#-- lambda for calling the PyCBC loglikelihood function ---
#PyCBC_logL = lambda q : pycbc_log_likelihood(q, psd, data)

#-- defining prior tranform ---
mchirp_min, mchirp_max = 1.197, 1.198
mass_ratio_min, mass_ratio_max = 1, 1.7
s1z_min, s1z_max = 0, 0.05
s2z_min, s2z_max = 0, 0.05
k_min, k_max = -6.6e-7, 6.6e-7
distance_mean, distance_var = 40.7, 3.3**2  # fixing the galaxy location
distance_min, distance_max = 12, 53
tc_min, tc_max = merger.time - 0.15, merger.time + 0.15

def prior_transform(cube):
     
    """
    chirpmass and q: distribution which is uniform in m1 and m2 and constrained by chirpmass and q
    spin1z/2z: uniform distribtion
    k: uniform distribution
    inclination: uniform in cos(iota)
    distance: uniform volume
    tc: uniform distribution
    """
        
    cube[0] = np.power((mchirp_max**2-mchirp_min**2)*cube[0]+mchirp_min**2,1./2)      # chirpmass: power law mc**1
    cube[1] = cdfinv_q(mass_ratio_min, mass_ratio_max, cube[1])                       # mass-ratio: uniform prior
    cube[2] = s1z_min + (s1z_max - s1z_min) * cube[2]                # s1z: uniform prior
    cube[3] = s2z_min + (s2z_max - s2z_min) * cube[3]                # s2z: uniform prior
    cube[4] = k_min + (k_max - k_min) * cube[4]                # s2z: uniform prior
    cube[5] = np.arccos(2*cube[5] - 1) 
    cube[6] = cdfinv_dL(distance_mean, distance_var, distance_min, distance_max, cube[6])  # distance: unifrom prior in dL**3
    cube[7] = 2*np.pi*cube[7] # pol: uniform angle
    cube[8] = tc_min + (tc_max - tc_min) * cube[8] # uniform in tc

    return cube

print('********** Sampling starts *********\n')

#-- sampling parameters ---
# nProcs = 64 
# nDims = 9
# sample = 'rwalk'
# dlogz_init = 1e-4
# seed_PE = 0

nLive = 1000
nWalks = 250
nDims = 9
dlogz = 0.1
sample = 'rwalk'
seed_PE = 0
nProcs = 64
        
st = time.time()

with mp.Pool(nProcs) as pool:

        sampler = dynesty.NestedSampler(pycbc_log_likelihood, prior_transform, nDims, sample=sample, pool=pool, nlive=nLive, \
                                                   walks=nWalks, queue_size=nProcs, rstate=Generator(PCG64(seed=seed_PE)))
        sampler.run_nested(dlogz=dlogz)

#-- definig dynesty sampler ---
# with mp.Pool(nProcs) as pool:
    
#     sampler = dynesty.DynamicNestedSampler(pycbc_log_likelihood, prior_transform, nDims, sample=sample, \
#                                             pool=pool, queue_size=nProcs, rstate=Generator(PCG64(seed=seed_PE)))
#     sampler.run_nested(dlogz_init=dlogz_init)

#-- saving raw pe samples ---
res = sampler.results
print('Evidence:{}'.format(res['logz'][-1]))

file = h5py.File('samples_data_k_{}_fISCO_seedPE_{}.hdf5'.format(whichfrac, seed_PE), 'w')
file.create_dataset('mchirp', data=res['samples'][:,0])
file.create_dataset('mass_ratio', data=res['samples'][:,1])
file.create_dataset('s1z', data=res['samples'][:,2])
file.create_dataset('s2z', data=res['samples'][:,3])
file.create_dataset('k', data=res['samples'][:,4])
file.create_dataset('iota', data=res['samples'][:,5])
file.create_dataset('distance', data=res['samples'][:,6])
file.create_dataset('pol', data=res['samples'][:,7])
file.create_dataset('tc', data=res['samples'][:,8])
file.create_dataset('logwt', data=res['logwt'])
file.create_dataset('logz', data=res['logz'])
file.create_dataset('logl', data=res['logl'])
file.create_dataset('Evidence', data=res['logz'][-1])
file.close()

et = time.time()

print('Done!!!')
print('Time taken:{} Hours.'.format((et-st)/3600.))    



