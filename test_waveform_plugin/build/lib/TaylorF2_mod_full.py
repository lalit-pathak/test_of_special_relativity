"""  A waveform model with extra phasing parameter using TaylorF2 as a base
"""
def TaylorF2_full(**params):
    
    """
    Function to generate a modified phase TaylorF2 waveform 
    
    """
    import lal
    import numpy as np
    from pycbc import waveform
    from pycbc.conversions import mtotal_from_mass1_mass2
    
    C = lal.C_SI  
    G = lal.G_SI
    Msun = lal.MSUN_SI
    Mpc = 3.086*1e22
    
    mtotal = mtotal_from_mass1_mass2(params['mass1'], params['mass2'])
    
    hp, _ = waveform.get_fd_waveform(approximant='TaylorF2', mass1=params['mass1'], mass2=params['mass2'], \
                                     spin1z=params['spin1z'], spin2z=params['spin2z'], distance=params['distance'], inclination=params['inclination'], \
                                     delta_f = params['delta_f'], f_lower=params['f_lower'])
    
    #--- amplitude and phase from template ---
    amp = waveform.utils.amplitude_from_frequencyseries(hp)
    phase = waveform.utils.phase_from_frequencyseries(hp)
    
    #--- index where the frequency is equal to lower cutoff frequency ---
    idx = np.where(phase.sample_frequencies.data==params['f_lower'])[0][0]
    
    #--- phase is zero for f < fLow so add_phase1 and add_phase2 are also zero for f < fLow ---
    add_phase1 = np.zeros(len(phase.sample_frequencies))  
    add_phase2 = np.zeros(len(phase.sample_frequencies))
    
    #--- defining additional phases ---
    x = (params['f_lower']/phase.sample_frequencies.data[idx:])**(1/3)
  
    add_phase1[idx:] = (-3*np.sin(params['inclination'])/(np.sqrt(2)*np.pi*C)) * (np.pi*G*mtotal*Msun*phase.sample_frequencies.data[idx:])**(1/3) * (- x**2 + x**6/20 - x**10/72 + 5*x**14/832)
    
    add_phase2[idx:] = ((3*params['k']*1e-9*params['distance']*Mpc*np.sin(params['inclination']))/(np.sqrt(2)*np.pi*C**2*G*mtotal*Msun)) * (np.pi*G*mtotal*Msun*phase.sample_frequencies.data[idx:])**(4/3) * (1 + x**2 + x**4/2*np.log(x) - x**6/8 + 5*x**8/128)
    
    #--- combining additional phases to get back template ---
    phase = phase + add_phase1 + add_phase2 
    hp = amp*np.exp(1j*phase)
    
    return hp, -1j*hp







