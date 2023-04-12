"""
Plotting utility functions
Author: Lalit Pathak(lalit.pathak@iitgn.ac.in)

"""

import corner
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import gaussian_kde
from collections import namedtuple

import h5py
import matplotlib.pyplot as plt
import matplotlib.lines as mpllines
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q

def pos_samples(filename, params):
    
    """
    Function to generate weighted posterior samples from raw samples coming from dynesty sampler.
    
    Parameter
    ---------
    filename: name of the file containing raw samples from dynesty PE run
    params: list of parameters of which weighted samples are sought
    
    Returns
    --------
    weighted_samples: samples (array) in the same order as given in the params (except s1z and s2z are replaced by chi_eff)
    
    """
    
    file = h5py.File(filename, 'r')
    
    samples = []
    
    i = 0
    
    for p in params:
        
        if(p =='k'):
            
            samples.append(np.array(file[p])*1e-9)
            
        else:
            
            samples.append(np.array(file[p]))
    
    logwt = np.array(file['logwt'])
    logz = np.array(file['logz'])
    
    file.close()
    
    samples = np.array(samples).T
    wts =  np.exp(logwt - logz[-1]) # posterior weights
    effective_size = int(len(wts)/(1+(wts/wts.mean() - 1)**2).mean()) # effective samples size
    
    seed = 0
    np.random.seed(seed)
    weighted_samples_index = np.random.choice(samples.shape[0], effective_size, p=wts, replace=False)
    weighted_samples = samples[weighted_samples_index]
    
    return weighted_samples

def plot_corner(samples, filename=None, save=False, dpi=None, **kwargs):
    
    """
    Function to generate beautiful corner plots using corner package
    
    Parameters
    ----------
    samples: weighted posterior samples (array shape: number of posterior samples x number of parameters)
    save: False (boolean) If want to save as a jpeg
    dpi: integer value, only works when we want to save
    **kwargs: keyword arguments for tweaking the corner plot
    
    Returns
    --------
    figure object
    
    """
    
    defaults_kwargs = dict(bins=20, smooth=0.9, label_kwargs=dict(fontsize=16),
            title_kwargs=dict(fontsize=16), color='#0072C1',
            truth_color='tab:orange', quantiles=None,
            levels=[1 - np.exp(-0.74**2/2), 1 - np.exp(-1.32**2/2)],
            plot_density=False, plot_datapoints=False, fill_contours=False,
            max_n_ticks=5, hist_kwargs=dict(density=True), show_titles=False, title_fmt=None)
    
    defaults_kwargs.update(kwargs)
    fig = corner.corner(samples, **defaults_kwargs)
    
    for ax in fig.get_axes():
    
        ax.tick_params(axis='both', labelsize=kwargs.get('label_kwargs').get('fontsize')-4)
    
    if save:
        
        if filename:
            
            fig.savefig(fname=filename, dpi=dpi, bbox_inches='tight', pad_inches=kwargs.get('pad_inches'))

        else:
            
            filename = 'corner_plot.jpeg'
            fig.savefig(fname=filename, dpi=200, bbox_inches='tight', pad_inches=kwargs.get('pad_inches'))

    return fig

def title_formats(samps, labels, titles, fmt_arr, bins, measure='map'):
    
    """
    Function to customize the title formats for the corner plots
    
    Parameter
    ---------
    samps: weighted posterior samples
    labels: list of labels
    titles: list of titles
    fmt_arr: list of formats

    Returns
    --------
    tlabels: titles with cutomized format
    
    """
    range_vals = []
    tlabels = []
    
    p = 0
    
    for t in titles:
        
        if(t == r'$k$'):
            
            kernel = gaussian_kde(samps[:,p])
            count, val = np.histogram(samps[:,p], bins)
            val_pdf = kernel.pdf(val)
            map_val = val[np.argmax(val_pdf)]/1e-18

            q_5, q_50, q_95 = np.quantile(samps[:,p]/1e-18, [0.05, 0.5, 0.95])
            
            if(measure=='median'):
              
              q_m, q_p = q_50-q_5, q_95-q_50

              title_fmt=".2f"
              fmt = "{{0:{0}}}".format(title_fmt).format
              title = r"${{{0}}}_{{-{1}}}^{{+{2}}} \, {{{3}}}$"
              title = title.format(fmt(q_50), fmt(q_m), fmt(q_p), fmt_func(1e-18))
              title = "{0} = {1}".format(titles[p], title)
              tlabels.append(title)
              p = p + 1
              
            else:
              
              q_m, q_p = map_val-q_5, q_95-map_val

              title_fmt=".2f"
              fmt = "{{0:{0}}}".format(title_fmt).format
              title = r"${{{0}}}_{{-{1}}}^{{+{2}}} \, {{{3}}}$"
              title = title.format(fmt(map_val), fmt(q_m), fmt(q_p), fmt_func(1e-18))
              title = "{0} = {1}".format(titles[p], title)
              tlabels.append(title)
              p = p + 1
        
        else:
        
            kernel = gaussian_kde(samps[:,p])
            count, val = np.histogram(samps[:,p], bins)
            val_pdf = kernel.pdf(val)
            map_val = val[np.argmax(val_pdf)]
            
            if(measure=='median'):
              
              q_5, q_50, q_95 = np.quantile(samps[:,p], [0.05, 0.5, 0.95])
              q_m, q_p = q_50-q_5, q_95-q_50

              title_fmt=fmt_arr[p]
              fmt = "{{0:{0}}}".format(title_fmt).format
              title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
              title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))
              title = "{0} = {1}".format(titles[p], title)
              tlabels.append(title)
              p = p + 1
              
            else:
              
              q_5, q_50, q_95 = np.quantile(samps[:,p], [0.05, 0.5, 0.95])
              q_m, q_p = map_val-q_5, q_95-map_val

              title_fmt=fmt_arr[p]
              fmt = "{{0:{0}}}".format(title_fmt).format
              title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
              title = title.format(fmt(map_val), fmt(q_m), fmt(q_p))
              title = "{0} = {1}".format(titles[p], title)
              tlabels.append(title)
              p = p + 1
            
    return tlabels

def fmt_func(x):
    
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    
    return r'\times 10^{{{}}}'.format(b)

def ecdf(samples):
    
    """
    Empirical cumulative distribution functions
    
    Parameter
    ---------
    samples: weighted posterior samples
    
    Returns
    -------
    
    values: values of the samples
    cdf: cdf values of the corresponding samples
    
    """
    
    vals = np.sort(samps)
    
    #calculate CDF values
    cdf = np.arange(len(samps)) / (len(samps))
    
    return vals, cdf
