import os
# os.environ["SPS_HOME"] = '/global/homes/e/eramey16/fsps/'
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

import fsps
import dynesty
import sedpy
import h5py, astropy
import numpy as np
import pandas as pd
import time
import pickle
# import util
from dl import queryClient as qc, helpers
from dl.helpers.utils import convert

from sedpy.observate import load_filters
from prospect.utils.obsutils import fix_obs
from prospect.utils import prospect_args

from prospect.models.templates import TemplateLibrary
from prospect.models import SpecModel
from prospect.models.transforms import tage_from_tuniv
from prospect.models.templates import TemplateLibrary
from prospect.models import priors
from prospect.sources import CSPSpecBasis
from prospect.likelihood import lnlike_spec, lnlike_phot, write_log
from dynesty.dynamicsampler import stopping_function, weight_function, _kld_error
from prospect.fitting import fit_model
from prospect import fitting
from scipy.stats import truncnorm

from astropy.cosmology import WMAP9
from dynesty.dynamicsampler import stopping_function, weight_function, _kld_error

### TODO next:
# - Check on load_obs and see if you can eliminate the need to pass args
# - Check where else args may be needed / used
# - What exactly is in args anyway?

# Get galaxy data
bands = ['g', 'r', 'i', 'z', 'w1', 'w2']
sdss = ['sdss_{}0'.format(b) for b in ['g','r','i','z']]
wise = ['wise_w1', 'wise_w2']
filternames = sdss + wise

# # Run SPS over sparse grid to get necessary data in cache/memory
# initial_theta_grid = np.around(np.arange(global_model.config_dict["logzsol"]['prior'].range[0], global_model.config_dict["logzsol"]['prior'].range[1], step=0.01), decimals=2)

# ls_id = 9906619143751091 # IPTF16geu
# ls_id = 9906620228572307 # random unlensed galaxy

iptf16 = {
    'ls_id': 9906619143751091,
    'ra': 316.066215680906,
    'dec': -6.34022115158153
}
unlensed = {
    'ls_id': 10995426317568971,
    'ra': 316.085565,
    'dec': -6.352871
}


output_dir = '/monocle/exports/'
input_dir = '/monocle/'
# input_dir = '/global/homes/e/eramey16/galactic_lenses/docker/'

gal = iptf16

# --------------
# RUN_PARAMS
# --------------

run_params = {'verbose':True,
              'debug':False,
              'outfile':'160411A',
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
              # 'nested_stop_kwargs': {"post_thresh": 0.1}, # Getting an error here
              'nested_stop_kwargs': {"target_n_effective": 100000},
              # SPS parameters
              'zcontinuous': 1
              }

class MassMet(priors.Prior):
    """A Gaussian prior designed to approximate the Gallazzi et al. 2005 
    stellar mass--stellar metallicity relationship.
    """

    prior_params = ['mass_mini', 'mass_maxi', 'z_mini', 'z_maxi']
    distribution = truncnorm
    massmet = np.loadtxt(input_dir+'gallazzi_05_massmet.txt')
    def __len__(self):
        
        """ Hack to work with Prospector 0.3
        """
        return 2

    def scale(self,mass):
        upper_84 = np.interp(mass, self.massmet[:,0], self.massmet[:,3]) 
        lower_16 = np.interp(mass, self.massmet[:,0], self.massmet[:,2])

        return (upper_84-lower_16)

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

def load_model(add_duste=False, opt_spec=False, smooth_spec = False,
               add_dust1 = True, massmet = True, add_agn = False,
               add_neb=True, luminosity_distance=None, obs=None, **extras):
    
    model_params = TemplateLibrary["parametric_sfh"]
    
    #fixed values
    model_params["imf_type"]["init"] = 1 # Chabrier
    model_params["dust_type"]["init"] = 2 #1 # Milky Way extinction law
    model_params["sfh"]["init"] = 4 # delayed-tau 
    model_params["logzsol"]["isfree"] = True
    model_params["tau"]["isfree"] = True
    model_params["dust2"]["isfree"] = True
    model_params["tage"]["isfree"] = True
    model_params["mass"]["isfree"]= True
    
    # Don't fit redshift (see original to change)
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
    model_params["tau"]["prior"] = priors.LogUniform(mini=0.1, maxi=10.0)
    # related to SFH - delayed tau SFH
    
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0,maxi=3.0)
    # Expect more dust attenuation from young stellar populations than old
    # Add dust1 - if star-forming or unknown galaxy type - set this on!
    if add_dust1: # Automatically 1/2 of dust2 since init is 0.5
        model_params['dust1'] = {'N':1, 'init':0.5, 'isfree':False,
                                'depends_on': dust2_to_dust1}
    # Total dust attenuation = 1.5*dust2 value (report in paper)
    
    # Add nebular emission parameters and turn nebular emission on
    if add_neb: # ALWAYS use!
        model_params.update(TemplateLibrary["nebular"])
        
        if opt_spec: # Do you have a spectrum? # fitting gas-phase metallicity
            model_params['nebemlineinspec']['init'] = True
            model_params['gas_logu'] = {'N':1, 'init': -2, 'isfree':True,
                                        'prior': priors.TopHat(mini=-4, maxi=-1), 'units': 'Q_H/N_H'}
            model_params['gas_logz'] = {'N':1, 'init': 0.0, 'units': 'log Z/Z_\\odot', 
                                        'depends_on': gas_logz,
                                        'isfree':True, 'prior': priors.TopHat(mini=-2.0, maxi=0.5)}
        
            model_params['gas_logu']['isfree'] = True
            model_params['gas_logz']['isfree'] = True
        else: # my case
            model_params['nebemlineinspec']['init'] = False
            model_params['gas_logu']['isfree'] = False
            model_params['gas_logz']['isfree'] = False # making these false sets to solar metallicity
            
    
    # Adding massmet param - ALWAYS use! 
    if massmet:
        model_params['massmet'] = {"name": "massmet", "N": 2, "isfree": True, "init": [8.0, 0.0],
                                   "prior": MassMet(z_mini=-1.0, z_maxi=0.19, mass_mini=7, mass_maxi=13)}
        model_params['mass']['isfree']=False
        model_params['mass']['depends_on']= massmet_to_mass
        model_params['logzsol']['isfree'] =False
        model_params['logzsol']['depends_on']=massmet_to_logzol
        # Useful in the case we have photometry / limited photometry
        # picks the metallicity first and then samples from the bounds for mass calculated from this
        # massmet_1 is mass formed within galaxy (see equation for mass formed -> stellar mass)
    
    
    if opt_spec: # not on by default - we don't have spectra
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

def load_obs(mag_in, mag_unc_in, object_redshift=None, spec = False, 
             spec_file = None, maskspec=False, phottable=None,
             luminosity_distance=0, snr=10, **kwargs):
    filters = load_filters(filternames)
    
    # if 'mag_in' not in args or args['mag_in'] is None:
    #     raise ValueError("--mag_in must be passed as an argument")
    # if 'mag_unc_in' not in args or args['mag_unc_in'] is None:
    #     raise ValueError("--mag_unc_in must be passed as an argument")
    
    if isinstance(mag_in, str):
        M_AB = np.array([float(x) for x in mag_in.strip('[]').split(',')])
        magerr = np.array([float(x) for x in mag_unc_in.strip('[]').split(',')])
    else:
        M_AB = np.array(mag_in)
        magerr = np.array(mag_unc_in)
    
    maggies = np.array(10**(-.4*M_AB))
    magerr = np.clip(magerr, 0.05, np.inf)
    maggies_unc = magerr * maggies / 1.086
    
    # # Redshift
    # if object_redshift is not None:
    #     z = object_redshift
    # else: z = None

    # Build obs
    obs = dict(wavelength=None, spectrum=None, unc=None, redshift=object_redshift,
               maggies=maggies, maggies_unc=maggies_unc, filters=filters)
    obs["phot_wave"] = [f.wave_effective for f in obs["filters"]]
    obs['phot_mask'] = np.isfinite(np.squeeze(maggies))
    # obs = fix_obs(obs)
    
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['unc'] = None
    
    print("Loaded observation")
    return(obs)

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

def load_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    print("Loaded SPS libraries")
    return sps

def load_gp(**extras):
    return None, None

# -----------------
# LnP function as global
# ------------------

def lnprobfn(theta, model=None, obs=None, def_sps=None, def_noise_model=None,
             verbose=run_params['verbose']):
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
    
    if def_sps is None:
        def_sps=sps # No idea how this global gets defined
    if def_noise_model is None:
        def_spec_noise, def_phot_noise = spec_noise, phot_noise
    else:
        def_spec_noise, def_phot_noise = def_noise_model
    
    lnp_prior = model.prior_product(theta, nested=True)
    if np.isfinite(lnp_prior):
        # Generate mean model
        try:
            mu, phot, x = model.mean_model(theta, obs, sps=def_sps)
        except(ValueError):
            return -np.infty

        # Noise modeling
        ### WHERE ARE ALL THESE GLOBALS DEFINED!?!?!
        if def_spec_noise is not None:
            def_spec_noise.update(**model.params)
        if def_phot_noise is not None:
            def_phot_noise.update(**model.params)
        vectors = {'spec': mu, 'unc': obs['unc'],
                   'sed': model._spec, 'cal': model._speccal,
                   'phot': phot, 'maggies_unc': obs['maggies_unc']}

        # Calculate likelihoods
        lnp_spec = lnlike_spec(mu, obs=obs, spec_noise=def_spec_noise, **vectors)
        lnp_phot = lnlike_phot(phot, obs=obs, phot_noise=def_phot_noise, **vectors)

        return lnp_phot + lnp_spec + lnp_prior
    else:
        return -np.infty


def prior_transform(u, model=None):
        
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

def load_all(mag_in, mag_unc_in, object_redshift=None, **run_params):
    spec_noise, phot_noise = load_gp(**run_params)
    obs = load_obs(mag_in, mag_unc_in, object_redshift=object_redshift, **run_params)
    model = load_model(obs=obs, **run_params)
    sps = load_sps(**run_params)
    noise_model = (spec_noise, phot_noise)
    
    return obs, model, sps, noise_model

# def run(mag_in, mag_unc_in, effective_samples, outfile, object_redshift=None, 
#         withmpi=False, run_params=run_params, output_dynesty=False, **kwargs):
#     print("\n\nRunning a task with mpi\n\n")
#     ### Get initial model, obs, sps, and noise
#     obs, model, sps, noise_model = load_all(mag_in, mag_unc_in, 
#                                             object_redshift=object_redshift,
#                                             **run_params)
#     spec_noise, phot_noise = noise_model
    
#     ### Get initial theta grid
#     initial_theta_grid = np.around(np.arange(model.config_dict["logzsol"]['prior'].range[0], 
#                                 model.config_dict["logzsol"]['prior'].range[1], step=0.01), decimals=2)
#     for theta_init in initial_theta_grid:
#         sps.ssp.params["sfh"] = model.params['sfh'][0]
#         sps.ssp.params["imf_type"] = model.params['imf_type'][0]
#         sps.ssp.params["logzsol"] = theta_init
#         sps.ssp._compute_csp()
    
#     ### Set stopping criterion
#     run_params['nested_stop_kwargs'] = {"target_n_effective": effective_samples}
    
#     ### Define prob and prior functions
#     def new_lnfn(x):
#         return lnprobfn(x, model=model, obs=obs, def_sps=sps, def_noise_model=noise_model)
#     def new_prior(u):
#         return prior_transform(u, model=model)
    
#     ### Run prospector
#     if withmpi: # Run with MPI
#         run_params['using_mpi'] = True
#         with MPIPool() as pool:
#             if not pool.is_master():
#                 pool.wait()
#                 sys.exit(0)
        
#             nprocs = pool.size
#             # The parent process will oversee the fitting
#             run_params.update(dict(nlive_init=400, nested_method="rwalk", nested_dlogz_init=0.05))
#             output = fitting.run_dynesty_sampler(new_lnfn, new_prior, model.ndim,
#                                                  pool=pool, queue_size=nprocs, 
#                                                  stop_function=stopping_function,
#                                                  wt_function=weight_function,
#                                                  **run_params)
#     else: # Run prospector without MPI
#         run_params.update(dict(nlive_init=400, nested_method="rwalk", nested_dlogz_init=0.05))
#         output = fitting.run_dynesty_sampler(new_lnfn, new_prior, model.ndim, 
#                                                  stop_function=stopping_function,
#                                                  wt_function=weight_function,
#                                                  **run_params)
#         runtime = (time.time()-start)/60.0
        
    
#     ### Pickle output (for importance sampling)
#     if output_dynesty:
#         outfile = outfile+'.pkl'
#         with open(outfile, 'wb') as file:
#             pickle.dump(output, file)
#     else:
#         from prospect.io import write_results as writer
#         outfile = outfile+'.h5'
#         writer.write_hdf5(outfile, {}, model, obs,
#                          output, None,
#                          sps=sps,
#                          tsample=None,
#                          toptimize=0.0)
#     print(f"\nWrote results to: {outfile}\n")

if __name__=='__main__':
    
    parser = prospect_args.get_parser()
    parser.add_argument('--object_redshift', type=float, default=None,
                        help=("Redshift for the model"))
    parser.add_argument('--mag_in', default=None, type=str)
    parser.add_argument('--mag_unc_in', default=None, type=str)
    parser.add_argument('--effective_samples', default=100000, type=int)
    parser.add_argument('--output_dynesty', action='store_true', 
                        help='Saves dynesty output file as pickle')
    
    args = parser.parse_args()
    print(args)
    
#     # Set up with MPI or without MPI
#     try:
#         import mpi4py
#         from mpi4py import MPI
#         from schwimmbad import MPIPool

#         mpi4py.rc.threads = False
#         mpi4py.rc.recv_mprobe = False

#         comm = MPI.COMM_WORLD
#         size = comm.Get_size()

#         withmpi = comm.Get_size() > 1
#         start = MPI.Wtime()
#     except ImportError as e:
#         print('Failed to start MPI; are mpi4py and schwimmbad installed? Proceeding without MPI.')
#         print(f'Message: {e}')
#         withmpi = False
#         start = time.time()
    
#     # if withmpi:
#     #     with MPIPool() as pool:
#     #         if not pool.is_master(): # TODO Emily: figure out what this does
#     #             pool.wait()
#     #             sys.exit(0)
#     run(args.mag_in, args.mag_unc_in, 
#         object_redshift=args.object_redshift,
#         outfile=args.outfile,
#         effective_samples=args.effective_samples,
#         withmpi=withmpi,
#         output_dynesty=args.output_dynesty
#        )
#     #     print(run_params)
#     #     runtime = (MPI.Wtime()-start)/60.0
#     # else:
#     #     run(args.mag_in, args.mag_unc_in, 
#     #             object_redshift=args.object_redshift,
#     #             outfile=args.outfile,
#     #             effective_samples=args.effective_samples,
#     #             mpi_pool=None,
#     #             output_dynesty=args.output_dynesty
#     #            )
#     runtime = (time.time()-start)/60.0
    
    
#     print(f"Prospector finished in {runtime:.4f} minutes")
    
    
    
    ######################## Older code - trying to replace ########################################
    # run_params['nested_target_n_effective'] = args.nested_target_n_effective
    
    obs, model, sps, noise_model = load_all(args.mag_in, args.mag_unc_in, args.object_redshift, 
                                            **run_params)
    spec_noise, phot_noise = noise_model
    
    initial_theta_grid = np.around(np.arange(model.config_dict["logzsol"]['prior'].range[0], 
                                model.config_dict["logzsol"]['prior'].range[1], step=0.01), decimals=2)
    
    for theta_init in initial_theta_grid:
        sps.ssp.params["sfh"] = model.params['sfh'][0]
        sps.ssp.params["imf_type"] = model.params['imf_type'][0]
        sps.ssp.params["logzsol"] = theta_init
        sps.ssp._compute_csp()
    
    # Set up mpi
    try:
        import mpi4py
        from mpi4py import MPI
        from schwimmbad import MPIPool

        mpi4py.rc.threads = False
        mpi4py.rc.recv_mprobe = False

        comm = MPI.COMM_WORLD
        size = comm.Get_size()

        withmpi = comm.Get_size() > 1
        start = MPI.Wtime()
    except ImportError as e:
        print('Failed to start MPI; are mpi4py and schwimmbad installed? Proceeding without MPI.')
        print(f'Message: {e}')
        withmpi = False
        start = time.time()

#     # Evaluate SPS over logzsol grid in order to get necessary data in cache/memory
#     # for each MPI process. Otherwise, you risk creating a lag between the MPI tasks
#     # caching data depending which can slow down the parallelization
#     if (withmpi) & ('logzsol' in model.free_params):
#         dummy_obs = dict(filters=None, wavelength=None)

#         logzsol_prior = model.config_dict["logzsol"]['prior']
#         lo, hi = logzsol_prior.range
#         logzsol_grid = np.around(np.arange(lo, hi, step=0.1), decimals=2)

#         sps.update(**model.params)  # make sure we are caching the correct IMF / SFH / etc
#         for logzsol in logzsol_grid:
#             model.params["logzsol"] = np.array([logzsol])
#             _ = model.predict(model.theta, obs=dummy_obs, sps=sps)

    # # ensure that each processor runs its own version of FSPS
    # # this ensures no cross-over memory usage
    # from prospect.fitting import lnprobfn, fit_model
    # from functools import partial
    # lnprobfn_fixed = partial(lnprobfn, sps=sps)
    
    run_params['nested_stop_kwargs'] = {"target_n_effective": args.effective_samples}
    def new_lnfn(x):
        return lnprobfn(x, model=model, obs=obs)
    def new_prior(u):
        return prior_transform(u, model=model)
    
    # run_params['nested_target_n_effective'] = args.nested_target_n_effective

    if withmpi:
        run_params["using_mpi"] = True
        with MPIPool() as pool:

            # The dependent processes will run up to this point in the code
            if not pool.is_master(): # TODO Emily: figure out what this means
                pool.wait()
                sys.exit(0)
            nprocs = pool.size
            # The parent process will oversee the fitting
            run_params.update(dict(nlive_init=400, nested_method="rwalk", nested_dlogz_init=0.05))
            output = fitting.run_dynesty_sampler(new_lnfn, new_prior, model.ndim,
                                                 pool=pool, queue_size=nprocs, 
                                                 stop_function=stopping_function,
                                                 wt_function=weight_function,
                                                 **run_params)
        print(run_params)
        runtime = (MPI.Wtime()-start)/60.0
    else:
        run_params.update(dict(nlive_init=400, nested_method="rwalk", nested_dlogz_init=0.05))
        output = fitting.run_dynesty_sampler(new_lnfn, new_prior, model.ndim, 
                                                 stop_function=stopping_function,
                                                 wt_function=weight_function,
                                                 **run_params)
        runtime = (time.time()-start)/60.0
    
    
    print(f"Prospector finished in {runtime:.4f} minutes")
    # print(f"Run params: {run_params}")

    # Pickle output (for importance sampling)
    if args.output_dynesty:
        outfile = args.outfile+'.pkl'
        with open(outfile, 'wb') as file:
            pickle.dump(output, file)
    else:
        from prospect.io import write_results as writer
        outfile = args.outfile+'.h5'
        writer.write_hdf5(outfile, {}, model, obs,
                         output, None,
                         sps=sps,
                         tsample=None,
                         toptimize=0.0)
    print(f"\nWrote results to: {outfile}\n")