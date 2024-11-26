### prospect_conversion.py - functions to convert an h5 file result to parameter estimates
### Author - Emily Everetts
### Date - 11/14/24

import numpy as np
import pandas as pd
import corner

from prospect.models.transforms import tage_from_tuniv
from prospect.plotting.utils import sample_posterior
from astropy.cosmology import WMAP9

prosp_params = ['zred', 'dust', 'age', 'tau', 'sfr', 'mass', 'met']

def age(tage, tau):
    """
    Function to convert from tage and tau to mass-weighted age
    Note that mass-weighted age = tage - age(tage, tau)
    This only works for a delayed-tau SFH
    """
    age = 2*tau - tage**2/(tau*np.exp(tage/tau)-tau-tage)
    return age

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

def specific_SFR(mass, z, sfr):
    # First solve for sSFR using full post. dist of mass and SFR
    # Take the mode perhaps? for redshift free
    sSFR = np.log10(np.array(sfr)/(10**np.array(mass)))
    # Find lookback time    
    type_SF = []
    # If z was fixed in the fit, t_lookback is a single value
    if type(z) == float:
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
                
    else:
        for i in np.arange(0, len(z), 1):
            t_lookback = tage_from_tuniv(zred=z[i], tage_tuniv=1.0)*1e9
            
            sf_cond = np.log10(1/3/(t_lookback))
            q_cond = np.log10(1/20/(t_lookback))
            
            if sSFR[i] >= sf_cond:
                type_SF.append('SF')
            elif (sSFR[i] < sf_cond) & (sSFR[i] > q_cond):
                type_SF.append('T')
            else: 
                type_SF.append('Q')   
    
    SF_mode = scipy.stats.mode(type_SF)[0][0]
    
    return SF_mode, sSFR

def tmax_to_tage(tmax=None,redshift=None,**kwargs):
    return WMAP9.age(redshift).value*(tmax) # in Gyr

# Gives you parameter estimates and quantiles
def quantiles_phot(results, model):
    # Get parameter names 
    parnames = np.array(results['theta_labels'], dtype='U20')

     # Get the arrays we need (trace, wghts)
    samples = results['chain']
    xx = samples.copy()
    wghts = results.get('weights', None)

    # Resample posterior using weights to 100k uniformly-sampled weights    
    theta_samp = sample_posterior(xx, weights=wghts, nsample=100000)

    # Change from optical depth to dust: note must have dust2 as free parameter for this to work
    if 'dust1' in model.params:
        # Calculate total dust from dust1 and dust2 contributions: AV = (dust1 + dust2)*1.086 
        # dust1 = 0.5 * dust2
        dust2 = 1.5*1.086*theta_samp[:, parnames.tolist().index('dust2')]
    else:    
        dust2 = theta_samp[:, parnames.tolist().index('dust2')]*1.086

    
    mass = 10**theta_samp[:, parnames.tolist().index('massmet_1')]
    mass_log = theta_samp[:, parnames.tolist().index('massmet_1')]

    logzsol = theta_samp[:, parnames.tolist().index('massmet_2')]
    # We want to plot dust2, t_m, tau, SFR, M, Z , Z_gas and redshift if it exists

    if 'zred' in parnames:
        # Define some variables
        zred = theta_samp[:, parnames.tolist().index('zred')]

        tmax_par = theta_samp[:, parnames.tolist().index('tmax')]
        t_age = tmax_to_tage(tmax=tmax_par,redshift=zred)
        tau = theta_samp[:, parnames.tolist().index('tau')]
        
        # Convert from tage to mass-weighted age
        t_m =  t_age - age(t_age,tau)
        # Calculate SFR
        SFR_calc = SFR(t_age, tau, mass) 

        stell_mass = sm(t_m, mass_log)   
        
        new_theta = []
        for i in np.arange(0, len(mass_log), 1):
            new_idx = [zred[i], dust2[i], t_m[i], tau[i], SFR_calc[i], stell_mass[i], logzsol[i]]
            new_theta.append(new_idx)

    else:
        # Define some variables
        t_age_par = theta_samp[:, parnames.tolist().index('tage')]
        t_age = t_age_par.copy()
        tau = theta_samp[:, parnames.tolist().index('tau')]
        
        # Convert from tage to mass-weighted age
        t_m =  t_age - age(t_age,tau)
        # Calculate SFR
        SFR_calc = SFR(t_age, tau, mass) 

        stell_mass = sm(t_m, mass_log)
        
        new_theta = []
        for i in np.arange(0, len(mass_log), 1):
            new_idx = [dust2[i], t_m[i], tau[i], SFR_calc[i], stell_mass[i], logzsol[i]]
            new_theta.append(new_idx)
            
        print(np.median(stell_mass))

    new_theta = np.array(new_theta)
    n_theta = new_theta.shape[1]
    quant = []
    for i in range(n_theta):
        quant.append(corner.quantile(new_theta[:,i], q=[0.16, 0.50, 0.84]))
    if 'zred' not in parnames: # no redshift
        params = prosp_params[1:]
    else: params = prosp_params
    return pd.DataFrame({params[i]:quant[i] for i in range(len(params))})