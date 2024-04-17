import time, sys, os
import numpy as np
np.errstate(invalid='ignore')

from prospect.models import model_setup
from prospect.io import write_results
from prospect import fitting
from prospect.likelihood import lnlike_spec, lnlike_phot, write_log
from dynesty.dynamicsampler import stopping_function, weight_function, _kld_error
from dynesty.utils import *

import mpi4py
from schwimmbad import MPIPool

mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False

import numpy as np
import os
from pandas import read_csv
from prospect.models import priors, SedModel
from prospect.models.sedmodel import PolySedModel
from prospect.models.templates import TemplateLibrary
from prospect.sources import CSPSpecBasis
from sedpy.observate import load_filters
from prospect.utils.obsutils import fix_obs
from scipy.stats import truncnorm
from prospect.models.transforms import tage_from_tuniv
from prospect.models.sedmodel import PolySpecModel
from prospect.models.sedmodel import SpecModel
from astropy.cosmology import WMAP9

run_params = {'verbose':True,
              'debug':False,
              'output_pickles': False,
              # Optimization parameters
              'do_powell': False,
              'ftol':3e-16, 'maxfev': 5000,
              'do_levenburg': False,
              'nmin': 5,
              # dynesty Fitter parameters
              'nested_bound': 'multi', # bounding method
              'nested_sample': 'rwalk', # sampling method
              'nested_nlive_init': 500,
              'nested_nlive_batch': 500,
              'nested_bootstrap': 0,
              'nested_dlogz_init': 0.05,
              'nested_weight_kwargs': {"pfrac": 1.0},
              'nested_stop_kwargs': {"post_thresh": 0.1},
              # SPS parameters
              'zcontinuous': 1
              }

import argparse

parser = argparse.ArgumentParser(description='Job array for Prospector.')
parser.add_argument("--objname", help="Object number for redshift and photometry.", type=int)
args = parser.parse_args()

objid = args.objname

host_data = read_csv('test_job_array.dat', names=['num', 'name', 'z', 'filters', 'maggies', 'maggies_unc'], delimiter='&')
run_params['outfile'] = host_data['name'][objid].strip()
print(run_params['outfile'])

def load_obs(obj_idx=objid, data=host_data, **kwargs):
    # Read in data file  
    # Grab correct filternames and photometry
    filternames = eval(data['filters'][obj_idx])
    mags = eval(data['maggies'][obj_idx])
    flux_uncertainty = eval(data['maggies_unc'][obj_idx])
       
    # Build output dictionary.
    obs = {}
    
    # This is a list of sedpy filter objects.    See the
    # sedpy.observate.load_filters command for more details on its syntax.
    obs['filters'] = load_filters(filternames)
    # This is a list of maggies, converted from mags.  It should have the same
    # order as `filters` above.
    obs["phot_wave"] = [f.wave_effective for f in obs["filters"]]
    obs['maggies'] = np.array(mags)
    obs['maggies_unc'] = np.array(flux_uncertainty)
    # Here we mask out any NaNs or infs
    obs['phot_mask'] = np.isfinite(np.squeeze(mags))
    # Add in  spectrum.
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['unc'] = None
    
    # Add in redshift
    obs['redshift'] = data['z'][obj_idx] # put in a value
   
    return obs

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

        return (upper_84-lower_16)/2 # DIVIDING BY 2 FOR LOCAL GALAXIES

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

def load_model(add_duste=True, opt_spec=False, smooth_spec = False,
               massmet = True, add_agn = False,
               add_neb=True, luminosity_distance=None, **extras):
    
    model_params = TemplateLibrary["parametric_sfh"]
    
    #fixed values
    model_params["imf_type"]["init"] = 1 # Chabrier
    model_params['dust_type']['init'] = 4 # Kriek and Conroy
    model_params["sfh"]["init"] = 4 # non delayed-tau 
    model_params["logzsol"]["isfree"] = True
    model_params["tau"]["isfree"] = True
    model_params["dust2"]["isfree"] = True
    model_params["tage"]["isfree"] = True
    model_params["mass"]["isfree"]= True
    
    obs = load_obs()
    
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
        model_params["zred"]["prior"] = priors.TopHat(mini=0.1, maxi=2.0)
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
                                   "prior": MassMet(z_mini=-1.2, z_maxi=0.3, mass_mini=6, mass_maxi=13)}
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


# --------------
# Globals
# --------------
# GP instances as global
spec_noise, phot_noise = load_gp(**run_params)
# Model as global
global_model = load_model(**run_params)
# Obs as global
global_obs = load_obs(**run_params)
# SPS Model instance as global
sps = load_sps(**run_params)

# Run SPS over sparse grid to get necessary data in cache/memory
initial_theta_grid = np.around(np.arange(global_model.config_dict["logzsol"]['prior'].range[0], global_model.config_dict["logzsol"]['prior'].range[1], step=0.01), decimals=2)

for theta_init in initial_theta_grid:
    sps.ssp.params["sfh"] = global_model.params['sfh'][0]
    sps.ssp.params["imf_type"] = global_model.params['imf_type'][0]
    sps.ssp.params["logzsol"] = theta_init
    sps.ssp._compute_csp()

# -----------------
# LnP function as global
# ------------------

def lnprobfn(theta, model=None, obs=None, verbose=run_params['verbose']):
    """Given a parameter vector and optionally a dictionary of observational
    ata and a model object, return the ln of the posterior. This requires that
    an sps object (and if using spectra and gaussian processes, a GP object) be
    instantiated.

    :param theta:
        Input parameter vector, ndarray of shape (ndim,)

    :param model:
        bsfh.sedmodel model object, with attributes including ``params``, a
        dictionary of model parameters.  It must also have ``prior_product()``,
        and ``mean_model()`` methods defined.

    :param obs:
        A dictionary of observational data.  The keys should be
          *``wavelength``
          *``spectrum``
          *``unc``
          *``maggies``
          *``maggies_unc``
          *``filters``
          * and optional spectroscopic ``mask`` and ``phot_mask``.

    :returns lnp:
        Ln posterior probability.
    """
    if model is None:
        model = global_model
    if obs is None:
        obs = global_obs

    lnp_prior = model.prior_product(theta, nested=True)
    if np.isfinite(lnp_prior):
        # Generate mean model
        try:
            mu, phot, x = model.mean_model(theta, obs, sps=sps)
        except(ValueError):
            return -np.infty

        # Noise modeling
        if spec_noise is not None:
            spec_noise.update(**model.params)
        if phot_noise is not None:
            phot_noise.update(**model.params)
        vectors = {'spec': mu, 'unc': obs['unc'],
                   'sed': model._spec, 'cal': model._speccal,
                   'phot': phot, 'maggies_unc': obs['maggies_unc']}

        # Calculate likelihoods
        lnp_spec = lnlike_spec(mu, obs=obs, spec_noise=spec_noise, **vectors)
        lnp_phot = lnlike_phot(phot, obs=obs, phot_noise=phot_noise, **vectors)

        return lnp_phot + lnp_spec + lnp_prior
    else:
        return -np.infty


def prior_transform(u, model=None):
    if model is None:
        model = global_model
        
    return model.prior_transform(u)


def halt(message):
    """Exit, closing pool safely.
    """
    print(message)
    try:
        pool.close()
    except:
        pass
    sys.exit(0)


if __name__ == "__main__":

    # --------------
    # Setup
    # --------------
    rp = run_params
    try:
        rp['sps_libraries'] = sps.ssp.libraries
    except(AttributeError):
        rp['sps_libraries'] = None
    # Use the globals
    model = global_model
    obs = global_obs
    if rp.get('debug', False):
        halt('stopping for debug')

    # Try to set up an HDF5 file and write basic info to it
    outroot = run_params['outfile']
    odir = os.path.dirname(os.path.abspath(outroot))
    if (not os.path.exists(odir)):
        badout = 'Target output directory {} does not exist, please make it.'.format(odir)
        halt(badout)

    # -------
    # Sample
    # -------
    if rp['verbose']:
        print('dynesty sampling...')
    tstart = time.time()  # time it
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        nprocs = pool.size

        dynestyout = fitting.run_dynesty_sampler(lnprobfn, prior_transform, model.ndim,
                                                 pool=pool, queue_size=nprocs, 
                                                 stop_function=stopping_function,
                                                 wt_function=weight_function,
                                                 **rp)
    ndur = time.time() - tstart
    print('done dynesty in {0}s'.format(ndur))

    # -------------------------
    # Output HDF5 (and pickles if asked for)
    # -------------------------
    if rp.get("output_pickles", False):
        # Write the dynesty result object as a pickle
        import pickle
        with open(outroot + '_dns.pkl', 'w') as f:
            pickle.dump(dynestyout, f)
    
        # Write the model as a pickle
        partext = write_results.paramfile_string(**rp)
        write_results.write_model_pickle(outroot + '_model', model, powell=None,
                                         paramfile_text=partext)
    
    # Write HDF5
    hfile = outroot + '_mcmc.h5'
    write_results.write_hdf5(hfile, rp, model, obs, dynestyout,
                             None, tsample=ndur)
