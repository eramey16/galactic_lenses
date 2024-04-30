import matplotlib.pyplot as plt
import numpy as np 
from astropy.io import fits
from pandas import read_csv
import matplotlib
import glob
from scipy.stats import binned_statistic
from scipy.stats import ks_2samp

import fsps
import sedpy
import prospect
import numpy as np
from prospect.models import priors
from prospect.models.sedmodel import SedModel
import time
import scipy
import h5py
from scipy.special import gamma, gammainc
from decimal import Decimal
import matplotlib.ticker as mticker

import json
import os
from prospect.utils.obsutils import fix_obs
from scipy.stats import truncnorm

from prospect.models.templates import TemplateLibrary
from prospect.io.read_results import results_from, get_sps
from prospect.models.sedmodel import PolySedModel
from prospect.utils.plotting import quantile
from prospect.models.transforms import tage_from_tuniv

from prospect.sources import CSPSpecBasis
from sedpy.observate import load_filters
from prospect.plotting.utils import sample_posterior
from astropy.cosmology import WMAP9 as cosmo
print(cosmo)


import corner as triangle
from prospect.io.read_results import traceplot, subcorner


# Draw random values from the mass posterior distributions in the h5 files - do about 50 different cdfs and then take
# the median and 50% cdf

def age(tage, tau):
    """
    Function to convert from tage and tau to mass-weighted age
    Note that mass-weighted age = tage - age(tage, tau)
    This only works for a delayed-tau SFH
    """
    age = 2*tau - tage**2/(tau*np.exp(tage/tau)-tau-tage)
    return age

def tmax_to_tage(tmax=None,redshift=None,**kwargs):
    return WMAP9.age(redshift).value*(tmax) # in Gyr

def SFR(tage, tau, mass):
    """
    To determine SFR from delayed-tau SFH
    Takes in lists of tage, tau and mass
    Outputs SFR in M_sol/yr
    """
    psi_arr = []
    
    for i in np.arange(0, len(tage), 1):
        # for delay tau this function gives the (unnormalized) SFR 
        # for any t, tau combo in M_sun/Gyr
        tau_i = tau[i]
        sfr = lambda t,tau_i: (t) * np.exp(-t/tau_i)
        # now we numerically integrate this SFH from 0 to tage to get the mass formed
        times = np.linspace(0, tage[i], 1000)
        A = np.trapz(sfr(times, tau[i]), times)
        # and now we renormalize the formed mass to the actual mass value 
        # to get the the SFR in M_sun per Gyr 
        psi = mass[i] * sfr(tage[i], tau[i]) / A
        # if we want SFR in Msun/year
        psi /= 1e9
        psi_arr.append(psi)
    return np.array(psi_arr)

def sm(t_m, logMF):
    logtm = np.log10(t_m * 1e9)
    MF = 10**logMF
    logsm = 1.06 - 0.24 * logtm + 0.01*(logtm**2)
    
    sm = 10**logsm
    mass = sm * MF
    return np.log10(mass)

def total_dust(dust1_fraction, dust2):
    return dust1_fraction*dust2 + dust2


def specific_SFR(mass, z, sfr):
    # First solve for sSFR using full post. dist of mass and SFR
    # Take the mode perhaps? for redshift free
    sSFR = np.log10(np.array(sfr)/(10**np.array(mass)))
    # Find lookback time    
    type_SF = []
    
    t_lookback = tage_from_tuniv(zred=z, tage_tuniv=1.0)*1e9

    for s in sSFR:
        sf_cond = np.log10(1/3/(t_lookback))
        q_cond = np.log10(1/20/(t_lookback))

        if s >= sf_cond:
            type_SF.append('SF')
        elif (s < sf_cond) & (s > q_cond):
            type_SF.append('T')
        else: 
            type_SF.append('Q')          
    
    SF_mode = scipy.stats.mode(type_SF)[0][0]
    
    return SF_mode, sSFR

def theta_posteriors(results, **kwargs):
    # Get parameter names 
    parnames = np.array(results['theta_labels'], dtype='U20')

     # Get the arrays we need (trace, wghts)
    samples = results['chain']
    xx = samples.copy()
    wghts = results.get('weights', None)

    # Resample posterior using weights to 100k uniformly-sampled weights    
    theta_samp = sample_posterior(xx, weights=wghts, nsample=100000)
    
    dust_AV = total_dust(theta_samp[:, parnames.tolist().index('dust1_fraction')], theta_samp[:, parnames.tolist().index('dust2')])

    # Define some variables
    
    tau = theta_samp[:, parnames.tolist().index('tau')]
    mass = 10**theta_samp[:, parnames.tolist().index('massmet_1')]
    mass_log = theta_samp[:, parnames.tolist().index('massmet_1')]


    logzsol = theta_samp[:, parnames.tolist().index('massmet_2')]

    
    
    if 'zred' in parnames:
        zred = theta_samp[:, parnames.tolist().index('zred')]

        tmax_par = theta_samp[:, parnames.tolist().index('tmax')]
        t_age = tmax_to_tage(tmax=tmax_par,redshift=zred)
        tau = theta_samp[:, parnames.tolist().index('tau')]
        
        # Convert from tage to mass-weighted age
        t_m =  t_age - age(t_age,tau)
        # Calculate SFR
        SFR_calc = SFR(t_age, tau, mass) 

        stell_mass = sm(t_m, mass_log)   
        
        type_SF, spSFR = specific_SFR(mass=stell_mass, z=zred, sfr=SFR_calc)
        type_z = 'photo-z'
        
    elif 'zred' not in parnames:
        t_age_par = theta_samp[:, parnames.tolist().index('tage')]
        t_age = t_age_par.copy()
        
        # Convert from tage to mass-weighted age
        t_m =  t_age - age(t_age,tau)
        # Calculate SFR
        SFR_calc = SFR(t_age, tau, mass) 

        stell_mass = sm(t_m, mass_log)
        
        for i in np.arange(0, len(res['model_params']),1):
            if res['model_params'][i]['name'] == 'zred':
                if res['model_params'][i]['isfree'] == False:
                    zred = res['model_params'][i]['init'] 
                    type_z = 'spec-z'
        type_SF, spSFR = specific_SFR(mass=stell_mass, z=zred, sfr=SFR_calc)

    return dustAV, t_m, SFR_calc, stell_mass, logzsol, zred, spSFR, type_z, type_SF


def sample_pdf(dist, nsamp = 50, **kwargs):
    """
    Builds posterior distributions for distributions drawn from Prospector fit
    Use the theta_posteriors code first
    Draws randomly from the distribution a certain number of times and builds a cdf from that
    randomly drawn distribution.
    
    'param': parameter you want to sample, options: mass, tm, met, SFR, dust
    'nsamp': number of random draws from the pdf
    """
    if type(dist) != float:
        x = np.random.choice(dist, size=nsamp, replace = True)
    else:
        dist = np.array([dist])
        x = np.random.choice(dist, size=nsamp, replace = True)
        
    return x


# Stores all pdfs
host_all = {
    'name': [],
    'AV': [],
    'age': [],
    'SFR': [],
    'sSFR': [],
    'mass': [],
    'logzsol': [],
    'z': [],
    'type_SF': [],
    'rand_mass': [],
    'rand_age': [],
    'rand_met': [],
    'rand_SFR': [],
    'rand_dust': [],
    'med_AV': [],
    'med_age': [],
    'med_sfr': [],
    'med_mass': [],
    'med_Zsol': []
    }


files = glob.glob('prospector_fits/*.h5', recursive = True)

num_samp = 5000 

for file in files:
    run_params = {}
    name = os.path.basename(file)[0:-8]
    host_all['name'].append(os.path.basename(file)[0:-8])
    print(name)
    res, obs, mod = results_from(file, dangerous=False)

    # Extract the pdf's for each stellar population property
    AV, mw_age, starFR, m_stell, Z_stell, redshift, spSFR, type_z, type_SF = theta_posteriors(res)

    host_all['AV'].append(AV)
    host_all['age'].append(mw_age)
    host_all['SFR'].append(starFR)
    host_all['mass'].append(m_stell)
    host_all['logzsol'].append(Z_stell)
    host_all['type_SF'].append(type_SF)
    host_all['sSFR'].append(spSFR)
    host_all['z'].append(redshift)

    # Randomly sample from the pdf's
    rand_mass = sample_pdf(m_stell, nsamp = num_samp)
    rand_age = sample_pdf(mw_age, nsamp = num_samp)
    rand_SFR = np.log10(sample_pdf(starFR, nsamp = num_samp))
    rand_met = sample_pdf(10**Z_stell, nsamp = num_samp)
    rand_dust = sample_pdf(AV, nsamp = num_samp)
    
    host_all['rand_mass'].append(rand_mass)
    host_all['rand_age'].append(rand_age)
    host_all['rand_met'].append(rand_met)
    host_all['rand_SFR'].append(rand_SFR)
    host_all['rand_dust'].append(rand_dust)

    med_AV = np.median(AV)
    med_age = np.median(mw_age)
    med_sfr = np.median(starFR)
    med_mass = np.median(m_stell)
    med_Zsol = np.median(10**Z_stell)

    # Get the medians from each distribution 
    host_all['med_AV'].append(med_AV)
    host_all['med_age'].append(med_age)
    host_all['med_sfr'].append(med_sfr)
    host_all['med_mass'].append(med_mass)
    host_all['med_Zsol'].append(med_Zsol)

    
np.save('SNe_stellar_pop_properties.npy',  host_all)    