# Imports
import matplotlib.pyplot as plt
import numpy as np 
from astropy.io import fits
from pandas import read_csv
import matplotlib
import glob
from scipy.stats import binned_statistic
from scipy.stats import ks_2samp
from prospect.models.sedmodel import SpecModel

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
from astropy.cosmology import WMAP9

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


import corner as triangle
from prospect.io.read_results import traceplot, subcorner


matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['axes.linewidth'] = 2.

class MassMet(priors.Prior):
    """A Gaussian prior designed to approximate the Gallazzi et al. 2005 
    stellar mass--stellar metallicity relationship.
    """

    prior_params = ['mass_mini', 'mass_maxi', 'z_mini', 'z_maxi']
    distribution = truncnorm
    massmet = np.loadtxt('gallazzi_05_massmet.txt')
    def __len__(self):
        
        """ Hack to work with Prospector 0.3
        """
        return 2

    def scale(self,mass):
        upper_84 = np.interp(mass, self.massmet[:,0], self.massmet[:,3]) 
        lower_16 = np.interp(mass, self.massmet[:,0], self.massmet[:,2])

        return (upper_84-lower_16)/2

    def loc(self,mass):
        return np.interp(mass, self.massmet[:,0], self.massmet[:,1])

    def get_args(self,mass):
        a = (self.params['z_mini'] - self.loc(mass)) / self.scale(mass)
        b = (self.params['z_maxi'] - self.loc(mass)) / self.scale(mass)

        return [a, b]

    @property

    def range(self):
        return ((self.params['mass_mini'], self.params['mass_maxi']),\

                (self.params['z_mini'], self.params['z_maxi']))

    def bounds(self, **kwargs):
        if len(kwargs) > 0:

            self.update(**kwargs)

        return self.range

    def __call__(self, x, **kwargs):

        """Compute the value of the probability density function at x and
        return the ln of that.
        :params x:
           x[0] = mass, x[1] = metallicity. Used to calculate the prior

        :param kwargs: optional
            All extra keyword arguments are used to update the `prior_params`.


        :returns lnp:

            The natural log of the prior probability at x, scalar or ndarray of
            same length as the prior object.

        """

        if len(kwargs) > 0:
            self.update(**kwargs)

        p = np.atleast_2d(np.zeros_like(x))
        a, b = self.get_args(x[...,0])
        p[...,1] = self.distribution.pdf(x[...,1], a, b, loc=self.loc(x[...,0]), scale=self.scale(x[...,0]))
        with np.errstate(invalid='ignore'):
            p[...,1] = np.log(p[...,1])

        return p

    def sample(self, nsample=None, **kwargs):
        """Draw a sample from the prior distribution.
        :param nsample: (optional)
            Unused
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = np.random.uniform(low=self.params['mass_mini'],high=self.params['mass_maxi'],size=nsample)
        a, b = self.get_args(mass)
        met = self.distribution.rvs(a, b, loc=self.loc(mass), scale=self.scale(mass), size=nsample)
        return np.array([mass, met])

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.
        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.
        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """

        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = x[0]*(self.params['mass_maxi'] - self.params['mass_mini']) + self.params['mass_mini']
        a, b = self.get_args(mass)
        met = self.distribution.ppf(x[1], a, b, loc=self.loc(mass), scale=self.scale(mass))
        return np.array([mass,met])
    
def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2
def dust2_to_dust1(dust2=None, **kwargs):
    return dust2
def massmet_to_mass(massmet=None, **extras):
    return 10**massmet[0]
def massmet_to_logzol(massmet=None,**extras):
    return massmet[1]
def gas_logz(gas_logz=None, **kwargs):
    return gas_logz
def tmax_to_tage(tmax=None,zred=None,**kwargs):
    return WMAP9.age(zred).value*(tmax) # in Gyr

# --------------
# SPS Object
# --------------

def load_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps


# -----------------
# Gaussian Process
# ------------------

def load_gp(**extras):
    return None, None


# --------------
# MODEL_PARAMS
# --------------

def load_model(obs, add_duste=True, opt_spec=False, smooth_spec = False,
               add_dust1 = False, massmet = True, add_agn = False,
               add_neb=True, luminosity_distance=None, **extras):
    
    model_params = TemplateLibrary["parametric_sfh"]
    
    #fixed values
    model_params["imf_type"]["init"] = 1 # Chabrier
    model_params["dust_type"]["init"] = 1 # Milky Way extinction law
    model_params["sfh"]["init"] = 4 # non delayed-tau 
    model_params["logzsol"]["isfree"] = True
    model_params["tau"]["isfree"] = True
    model_params["dust2"]["isfree"] = True
    model_params["tage"]["isfree"] = True
    model_params["mass"]["isfree"]= True
        
    # Setting redshift
    if obs['redshift'] is not None:
        model_params["zred"]['isfree'] = False
        model_params["zred"]['init'] = obs['redshift']
        
        # Find max age of the universe at this redshift
        tage_max = tage_from_tuniv(zred=obs['redshift'], tage_tuniv=1.0)
        
        # Set tage range
        model_params["tage"]["prior"] = priors.TopHat(mini=0.0, maxi=tage_max)
    
    elif obs['redshift'] is None:
        model_params["zred"]['isfree'] = True
        model_params["zred"]["prior"] = priors.TopHat(mini=0.6, maxi=2.0)
        model_params['tmax'] = {'N': 1,'isfree': True,'init': 0.5,
                                   'prior': priors.TopHat(mini=0, maxi=1.0)}
        
        model_params['tage']['isfree'] = False
        model_params['tage']['depends_on'] = tmax_to_tage


   
    # adjust priors
    model_params["tau"]["prior"] = priors.LogUniform(mini=0.1, maxi=30.0)
    
    model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1)
    model_params["dust_index"] = {"N": 1, 
                                  "isfree": True,
                                  "init": 0.0, "units": "power-law multiplication of Calzetti",
                                  "prior": priors.TopHat(mini=-1.0, maxi=0.4)}

    model_params['dust1'] = {"N": 1, 
                             "isfree": False, 
                             'depends_on': to_dust1,
                             "init": 0.0, "units": "optical depth towards young stars",
                             "prior": None}
    model_params['dust1_fraction'] = {'N': 1,
                                      'isfree': True,
                                      'init': 1.0,
                                      'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}
    

    # Add nebular emission parameters and turn nebular emission on
    if add_neb: # ALWAYS use!
        model_params.update(TemplateLibrary["nebular"])
        
        if opt_spec: # Do you have a spectrum?
            model_params['nebemlineinspec']['init'] = True
            model_params['gas_logu'] = {'N':1, 'init': -2, 'isfree':True,
                                        'prior': priors.TopHat(mini=-4, maxi=-1), 'units': 'Q_H/N_H'}
            model_params['gas_logz'] = {'N':1, 'init': 0.0, 'units': 'log Z/Z_\\odot', 'depends_on': gas_logz,
                                        'isfree':True, 'prior': priors.TopHat(mini=-2.0, maxi=0.5)}
        
            model_params['gas_logu']['isfree'] = True
            model_params['gas_logz']['isfree'] = True
        else:
            model_params['nebemlineinspec']['init'] = False
            model_params['gas_logu']['isfree'] = False
            model_params['gas_logz']['isfree'] = False
            
    
    # Adding massmet param - ALWAYS use! 
    if massmet:
        model_params['massmet'] = {"name": "massmet", "N": 2, "isfree": True, "init": [8.0, 0.0],
                                   "prior": MassMet(z_mini=-2.0, z_maxi=0.3, mass_mini=6, mass_maxi=13)}
        model_params['mass']['isfree']=False
        model_params['mass']['depends_on']= massmet_to_mass
        model_params['logzsol']['isfree'] =False
        model_params['logzsol']['depends_on']=massmet_to_logzol
        
    # Dust emission in FIR - use if you have well-sampled FIR region    
    if add_duste:
        model_params.update(TemplateLibrary["dust_emission"])
        model_params["duste_gamma"]["isfree"] = False
        #model_params["duste_gamma"]["prior"] = priors.LogUniform(mini=0.001, maxi=0.15)
        model_params["duste_qpah"]["isfree"] = True
        model_params["duste_qpah"]["prior"] = priors.TopHat(mini=0.5, maxi=7.0)
        model_params["duste_umin"]["isfree"] = False
        #model_params["duste_umin"]["prior"] = priors.TopHat(mini=0.1,maxi=25)
    
    # Optical depth in MIR - use if you know an AGN exists or have IR-FIR data
    if add_agn:
        model_params.update(TemplateLibrary["agn"])
        model_params['agn_tau']['isfree'] = True # optical depth
        model_params['agn_tau']['prior'] = priors.LogUniform(mini=10.0, maxi=90.0)
        model_params['fagn']['isfree'] = True
        # Note that fagn > 2 is unphysical
        model_params['fagn']['prior'] = priors.LogUniform(mini=1e-5, maxi=2)
        model_params['add_dust_agn'] = {'N':1, 'init':True, 'isfree':False, "units":" ", 'prior':None}
        
    if opt_spec:
        model_params.update(TemplateLibrary["optimize_speccal"])
        # fit for normalization of spectrum
        model_params['spec_norm'] = {'N': 1,'init': 1.0,'isfree': True,'prior': 
                                     priors.Normal(sigma=0.2, mean=1.0), 'units': 'f_true/f_obs'}
        # Increase the polynomial size to 12
        model_params['polyorder'] = {'N': 1, 'init': 6,'isfree': False}
        
        run_params["opt_spec"] = True
    
        # Now instantiate the model using this new dictionary of parameter specifications
        model = PolySpecModel(model_params)
        
        if smooth_spec:
            # Note that if you're using this method, you'll be optimizing "spec_norm" rather than fitting for it 
            model_params['spec_norm'] = {'N': 1,'init': 1.0,'isfree': False,'prior': 
                                         priors.Normal(sigma=0.2, mean=1.0), 'units': 'f_true/f_obs'}
            # Increase the polynomial size to 12
            model_params.update(TemplateLibrary['spectral_smoothing'])
            model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=40.0, maxi=400.0)
            
            # This matches the continuum in the model spectrum to the continuum in the observed spectrum.
            # Highly recommend using when modeling both photometry & spectroscopy
            model_params.update(TemplateLibrary['optimize_speccal'])
            
    elif opt_spec == False:
        model = SpecModel(model_params)
        run_params["opt_spec"] = False
        
    return model


def makeSED(result, obs_dict, model, stellarpop):
    flatchain = result['chain']

    post_pcts = [quantile(flatchain[:, i], percents=50, weights=result.get("weights", None))
                 for i in range(model.ndim)]
    
    # Converged spectrum and photometric values
    mspec_conv, mphot_conv, _ = model.mean_model(post_pcts, obs_dict, sps=stellarpop)

    # Photometry
    wphot = obs_dict["phot_wave"] # Model/Observed photometric wavelengths
    modphot = mphot_conv * 3631e6

    # If observed spectrum was used in fit
    if obs_dict['wavelength'] is not None:        
        modspec = (model._norm_spec)*3631e6 # Already normalized model spectrum
        wspec = stellarpop.wavelengths.copy()*(1+model.params.get('zred')) # Full wavelength range

    # If observed spectrum was NOT used in fit
    else:
        a = 1 + model.params.get('zred')
        wspec = stellarpop.wavelengths.copy()
        wspec *= a #redshift them
        modspec = mspec_conv * 3631e6 # Converted model spectrum
    
    return wphot, modphot, wspec, modspec



files_h5 = glob.glob('prospector_fits/*.h5', recursive = True)



for i in np.arange(0, len(files_h5),1):
    print(os.path.basename(files_h5[i]))

    run_params = {}
    # Load in the h5 file
    res, obs, _ = results_from(files_h5[i], dangerous=False)

    mod = load_model(obs, **run_params)
    sps = load_sps(**run_params)

    fig = plt.figure(figsize=(15,10))
    
    wphot, mphot, wspec, mspec = makeSED(result = res, obs_dict = obs, model = mod, stellarpop = sps)

    plt.errorbar(wphot, mphot, label='Model photometry',
                 marker='s', markersize=10, alpha=1.0, ls='', lw=3, 
                 markerfacecolor='none', markeredgecolor='#B631EA', markeredgewidth=3, zorder=3)
    
    plt.plot(wspec, mspec, color='#B631EA', label='Model Spectrum')
    
    plt.errorbar(wphot, obs['maggies']*3631e6, yerr=obs['maggies_unc']*3631e6, label='Observed photometry', 
                 ecolor='#F12F16', marker='o', markersize=10, ls='', lw=3, alpha=1.0, markerfacecolor='none', 
                 markeredgecolor='#F12F16', markeredgewidth=2, zorder=4)
    
    #Set observed wavelength min and max
    obs_wmin = np.min(wphot)-1000
    obs_wmax = np.max(wphot)+10000

    obs_smin = np.min(obs['maggies']*3631e5)
    obs_smax = np.max(obs['maggies']*3631e7)

    plt.xlim(obs_wmin, obs_wmax)
    plt.ylim(obs_smin, obs_smax)
    
    plt.yscale('log')
    plt.xscale('log')
    
    plt.legend(loc='lower right', fontsize=20)
    
    plt.xlabel(r'Observed Wavelength [$\AA$]', fontsize=20)
    plt.ylabel(r'Flux Density [$\mu$Jy]', fontsize=20)
    plt.title('SN {}, z = {}'.format(os.path.basename(files_h5[i][0:-8]), np.round_(obs['redshift'],2)), fontsize=20)
    
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=20, length=10, width=2, direction='in', 
                       bottom=True, top=True, left=True, right=True)

    path = 'SED/'
    plt.savefig(path+os.path.basename(files_h5[i][0:-8])+'.png', dpi=300)