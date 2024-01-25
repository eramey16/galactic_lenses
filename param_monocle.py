import os
# os.environ["SPS_HOME"] = '/global/homes/e/eramey16/fsps/'
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

import fsps
import dynesty
import sedpy
import h5py, astropy
import numpy as np
import pandas as pd
import classify
import util
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
from scipy.stats import truncnorm

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

gal = iptf16

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

def load_model(**run_params):
    
    model_params = TemplateLibrary["continuity_flex_sfh"]
    
    #fixed values
    model_params["imf_type"]["init"] = 1 # Chabrier
    model_params["dust_type"]["init"] = 2 #1 # Milky Way extinction law
    model_params["sfh"]["init"] = 4 # delayed-tau 
    model_params["logzsol"]["isfree"] = True
    model_params["tau"]["isfree"] = True
    model_params["dust2"]["isfree"] = True
    model_params["tage"]["isfree"] = True
    model_params["mass"]["isfree"]= True

def build(maggies, maggies_unc, object_redshift, **run_params):
    filters = load_filters(filternames)

    # Build obs
    obs = dict(wavelength=None, spectrum=None, unc=None, redshift=object_redshift,
               maggies=maggies, maggies_unc=maggies_unc, filters=filters)
    obs["phot_wave"] = [f.wave_effective for f in obs["filters"]]
    obs = fix_obs(obs)
    print(obs)

    # Build model
    # model_params = TemplateLibrary["parametric_sfh"]
    model_params = TemplateLibrary['continuity_flex_sfh']
    model_params.update(TemplateLibrary["nebular"])
    model_params["zred"]["init"] = obs["redshift"]

    model = SpecModel(model_params)
    noise_model = (None, None)

    print(model)

    # from prospect.sources import CSPSpecBasis
    from prospect.sources import FastStepBasis
    # sps = CSPSpecBasis(zcontinuous=1)
    sps = FastStepBasis(zcontinuous=1)
    print(sps.ssp.libraries)
    
    return obs, model, sps, noise_model

if __name__=='__main__':
    
    parser = prospect_args.get_parser()
    parser.add_argument('--object_redshift', type=float, default=None,
                        help=("Redshift for the model"))
    parser.add_argument('--mag_in', default=None, type=str)
    parser.add_argument('--mag_unc_in', default=None, type=str)
    
    args = parser.parse_args()
    
    if args.mag_in is None:
        gal_data = classify.query_galaxy(ra=gal['ra'], dec=gal['dec'])
        gal_data = util.clean_and_calc(gal_data, mode='dr9').iloc[0]
        print(gal_data)
        maggies = np.array([10**(-.4*gal_data['dered_mag_'+b]) for b in bands])
        magerr = 2.5 / np.log(10) / np.array([gal_data['snr_'+b] for b in bands])
        magerr = np.clip(magerr, 0.05, np.inf)
        maggies_unc = magerr * maggies / 1.086
        z = gal_data['z_phot_median']
        
    else:
        M_AB = np.array([float(x) for x in args.mag_in.strip('[]').split(',')])
        maggies = np.array(10**(-.4*M_AB))
        magerr = np.array([float(x) for x in args.mag_unc_in.strip('[]').split(',')])
        magerr = np.clip(magerr, 0.05, np.inf)
        z = args.object_redshift
        maggies_unc = magerr * maggies / 1.086

    
    obs, model, sps, noise_model = build(maggies, maggies_unc, z)
    
    try:
        import mpi4py
        from mpi4py import MPI
        from schwimmbad import MPIPool

        mpi4py.rc.threads = False
        mpi4py.rc.recv_mprobe = False

        comm = MPI.COMM_WORLD
        size = comm.Get_size()

        withmpi = comm.Get_size() > 1
    except ImportError as e:
        print('Failed to start MPI; are mpi4py and schwimmbad installed? Proceeding without MPI.')
        print(f'Message: {e}')
        withmpi = False

    # Evaluate SPS over logzsol grid in order to get necessary data in cache/memory
    # for each MPI process. Otherwise, you risk creating a lag between the MPI tasks
    # caching data depending which can slow down the parallelization
    if (withmpi) & ('logzsol' in model.free_params):
        dummy_obs = dict(filters=None, wavelength=None)

        logzsol_prior = model.config_dict["logzsol"]['prior']
        lo, hi = logzsol_prior.range
        logzsol_grid = np.around(np.arange(lo, hi, step=0.1), decimals=2)

        sps.update(**model.params)  # make sure we are caching the correct IMF / SFH / etc
        for logzsol in logzsol_grid:
            model.params["logzsol"] = np.array([logzsol])
            _ = model.predict(model.theta, obs=dummy_obs, sps=sps)

    # ensure that each processor runs its own version of FSPS
    # this ensures no cross-over memory usage
    from prospect.fitting import lnprobfn, fit_model
    from functools import partial
    lnprobfn_fixed = partial(lnprobfn, sps=sps)

    if withmpi:
        # run_params["using_mpi"] = True
        with MPIPool() as pool:

            # The dependent processes will run up to this point in the code
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            nprocs = pool.size
            # The parent process will oversee the fitting
            fitting_kwargs = dict(nlive_init=400, nested_method="rwalk", nested_target_n_effective=1000, nested_dlogz_init=0.05)
            output = fit_model(obs, model, sps, noise=noise_model, pool=pool, queue_size=nprocs, 
                               lnprobfn=lnprobfn_fixed, using_mpi=True, **fitting_kwargs)
    else:
        fitting_kwargs = dict(nlive_init=400, nested_method="rwalk", nested_target_n_effective=1000, nested_dlogz_init=0.05)
        output = fit_model(obs, model, sps, optimize=False, dynesty=True, lnprobfn=lnprobfn, 
                           noise=noise_model, **fitting_kwargs)

    result, duration = output["sampling"]

    from prospect.io import write_results as writer
    hfile = args.outfile+'.h5'
    writer.write_hdf5(hfile, {}, model, obs,
                     output["sampling"][0], None,
                     sps=sps,
                     tsample=output["sampling"][1],
                     toptimize=0.0)
