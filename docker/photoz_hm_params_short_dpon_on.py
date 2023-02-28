import time, sys

import numpy as np
from sedpy.observate import load_filters
import pandas as pd
from prospect.utils import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from astropy.cosmology import WMAP9
from scipy.special import gamma, gammainc
from scipy.stats import truncnorm
import numpy as np
import os
from prospect.models import priors

#--------
#MASSMET 
#--------

class MassMet(priors.Prior):
    """A Gaussian prior designed to approximate the Gallazzi et al. 2005 
    stellar mass--stellar metallicity relationship.
    """

    prior_params = ['mass_mini', 'mass_maxi', 'z_mini', 'z_maxi']
    distribution = truncnorm
    #massmet = np.loadtxt('gallazzi_05_massmet.txt')
    massmet = np.array([[ 8.870e+00, -6.000e-01, -1.110e+00, -0.000e+00], 
           [ 9.070e+00, -6.100e-01, -1.070e+00, -0.000e+00], 
           [ 9.270e+00, -6.500e-01, -1.100e+00, -5.000e-02], 
           [ 9.470e+00, -6.100e-01, -1.030e+00, -1.000e-02], 
           [ 9.680e+00, -5.200e-01, -9.700e-01,  5.000e-02], 
           [ 9.870e+00, -4.100e-01, -9.000e-01,  9.000e-02], 
           [ 1.007e+01, -2.300e-01, -8.000e-01,  1.400e-01], 
           [ 1.027e+01, -1.100e-01, -6.500e-01,  1.700e-01], 
           [ 1.047e+01, -1.000e-02, -4.100e-01,  2.000e-01], 
           [ 1.068e+01,  4.000e-02, -2.400e-01,  2.200e-01], 
           [ 1.087e+01,  7.000e-02, -1.400e-01,  2.400e-01], 
           [ 1.107e+01,  1.000e-01, -9.000e-02,  2.500e-01], 
           [ 1.127e+01,  1.200e-01, -6.000e-02,  2.600e-01], 
           [ 1.147e+01,  1.300e-01, -4.000e-02,  2.800e-01], 
           [ 1.168e+01,  1.400e-01, -3.000e-02,  2.900e-01], 
           [ 1.187e+01,  1.500e-01, -3.000e-02,  3.000e-01]]) 
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
    
    
def dust2_to_dust1(dust2=None, **kwargs):
    return dust2
def massmet_to_mass(massmet=None, **extras):
    return 10**massmet[0]
def massmet_to_logzol(massmet=None,**extras):
    return massmet[1]

# --------------
# Model Definition
# --------------
def build_model(object_redshift=None,massmet=True, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.

    :param add_dust: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.

    """
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel

    # --- Get a basic delay-tau SFH parameter set. ---
    # This has 5 free parameters:
    #   "mass", "logzsol", "dust2", "tage", "tau"
    # And two fixed parameters
    #   "zred"=0.1, "sfh"=4
    model_params = TemplateLibrary["parametric_sfh"]

    # First, redshift
    if object_redshift is None:
        model_params['zred']['isfree'] = True
        model_params['zred']['prior'] = priors.TopHat(mini=0.0,maxi=9.0)
    else:
        model_params['zred']['isfree'] = False         # this is the default, but we're being explicit
        model_params['zred']['init'] = object_redshift
    if massmet:
        model_params['massmet'] = {"name": "massmet", "N": 2, "isfree": True, "init": [8.0, 0.0],
                     "prior": MassMet(z_mini=-0.5, z_maxi=0.10, mass_mini=9, mass_maxi=11)}
        model_params['mass']['isfree']=False
        model_params['mass']['depends_on']= massmet_to_mass
        model_params['logzsol']['isfree'] =False
        model_params['logzsol']['depends_on']=massmet_to_logzol
    else:     
    # Mass
    # note by default this is *total mass formed*, not stellar mass
        model_params['mass']['prior'] = priors.LogUniform(mini=1e6, maxi=1e13)
        model_params['logzsol']['prior'] = priors.TopHat(mini=-1.98,maxi=0.19)
        
    model_params['imf_type']['init'] = 1 # assume a Chabrier IMF

    # Metallicity
    # prior simply reflects allowed template range for now.
    # Here we specify the metallicity interpolation to use a triangular weighting scheme
    # this avoids the challenges of interpolating over non-linear and non-monotonic
    # effect of stellar metallicity
    model_params['pmetals'] = {'N': 1,
                               'init': -99,
                               'isfree': False}

    # choose a delayed-tau SFH here
    # tau limits set following Wuyts+11
    model_params['sfh']['init'] = 4 # delayed-tau
    model_params["tau"]["prior"] = priors.LogUniform(mini=0.1, maxi=10)

    # Introduce new variable, `logtmax`
    # this allows us to set tage_max = f(z)
    # This assumes a WMAP9 cosmology for the redshift --> age of universe conversion
    def logtmax_to_tage(logtmax=None,zred=None,**kwargs):
        return WMAP9.age(zred).value*(10**logtmax) # in Gyr
    
    model_params['tage']['isfree'] = False
    #model_params["tage"]["prior"] = priors.TopHat(mini=0.0, maxi=6.0)
    model_params['tage']['depends_on'] = logtmax_to_tage
    model_params['logtmax'] = {'N': 1,
                               'isfree': True,
                               'init': 0.5,
                               'prior': priors.TopHat(mini=np.log10(0.73e-4), maxi=np.log10(1.0))}

    # Dust attenuation. We choose a standard dust screen model here.
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=4.0)

    # Young stars and nebular regions get extra attenuation
    def dust2_to_dust1(dust2=None,**kwargs):
        return dust2
    model_params['dust1'] = {'N': 1,
                             'init': 0.5,
                             'isfree': False,
                             'depends_on': dust2_to_dust1}

    # We let the dust attenuation law vary according to the amount of dust
    def dust2_to_dustindex(dust2=None,**kwargs):
        return -0.095 + 0.111*dust2 - 0.0066*dust2**2
    model_params['dust_type']['init'] = 4   # Calzetti with power-law modification
    model_params['dust_index'] = {'N': 1,
                                  'init': 0.0,
                                  'isfree': False,
                                  'depends_on': dust2_to_dustindex}

    # Nebular emission
    model_params.update(TemplateLibrary["nebular"])

    # Gas metallicity == stellar metallicity
    def logzsol_to_gaslogz(logzsol=None,**kwargs):
        return logzsol
    model_params['gas_logz']['isfree'] = False
    model_params['gas_logz']['depends_on'] = logzsol_to_gaslogz

    # Allow ionization parameter to vary based on sSFR
    def ssfr_to_gaslogu(logtmax=None,tau=None,zred=None,**kwargs):
        # calculate sSFR
        tage = logtmax_to_tage(logtmax=logtmax,zred=zred)
        ssfr = (tage/tau**2) * np.exp(-tage/tau) / (gamma(2) * gammainc(2, tage/tau)) * 1e-9

        # above calculation is missing a factor of (stellar mass / total mass formed)
        # this is a pain to estimate and typically is (0.64-1.0)
        # take a rough estimate here to split the difference
        # this is an ok approximation since it typically varies by ~7 orders of magnitude
        ssfr *= 0.82

        # now plug into relationship from Kaasinen+18
        gas_logu = np.log10(ssfr)*0.3125 + 0.9982   

        return np.clip(gas_logu,-4.0,-1.0)  # stay within allowed range
    model_params['gas_logu']['isfree'] = False
    model_params['gas_logu']['depends_on'] = ssfr_to_gaslogu

    # Make sure to add nebular emission to the spectrum
    # this takes extra runtime, but will be important when emulating the spectrum
   # model_params['nebemlineinspec']['init'] = True

    model_params['nebemlineinspec']={'N': 1,'isfree': False,'init': False,'prior': None}

    # We don't need to produce or emulate the infrared.
    model_params['add_dust_emission'] = {'N': 1, 'init': False, 'isfree': False}

    return sedmodel.SedModel(model_params)


def build_obs(objid=0,**kwargs):
    """Load photometry from an ascii file.  Assumes the following columns:
    `objid`, `filterset`, [`mag0`,....,`magN`] where N >= 11.  The User should
    modify this function (including adding keyword arguments) to read in their
    particular data format and put it in the required dictionary.

    :param objid:
        The object id for the row of the photomotery file to use.  Integer.
        Requires that there be an `objid` column in the ascii file.

    :param luminosity_distance: (optional)
        The Johnson 2013 data are given as AB absolute magnitudes.  They can be
        turned into apparent magnitudes by supplying a luminosity distance.

    :returns obs:
        Dictionary of observational data.
    """
    # Writes your code here to read data.  Can use FITS, h5py, astropy.table,
    # sqlite, whatever.
    # e.g.:
    # import astropy.io.fits as pyfits
    # catalog = pyfits.getdata(phottable)

    from prospect.utils.obsutils import fix_obs

    # Name the filters
    sdss = ['sdss_{}0'.format(b) for b in ['g','r','z']]
    wise = ['wise_w1', 'wise_w2']
    filternames = sdss + wise
    
    # fake fluxes!
    M_AB = np.array([float(item) for item in kwargs["mag_in"].strip("[]").split(',')])
    M_AB_unc = np.array([float(item) for item in kwargs["mag_unc_in"].strip("[]").split(',')])
    
    mags = 10**(-0.4*M_AB)
    
    mag_down = [x-y for (x,y) in zip(M_AB, M_AB_unc)]
    flux_down = [10**(-0.4*x) for x in mag_down]
    flux_uncertainty = [y-x for (x,y) in zip(mags, flux_down)]
    
    # Build output dictionary.
    obs = {}
    # This is a list of sedpy filter objects.    See the
    # sedpy.observate.load_filters command for more details on its syntax.
    obs['filters'] = load_filters(filternames)
    obs["phot_wave"] = [f.wave_effective for f in obs["filters"]]
    # This is a list of maggies, converted from mags.  It should have the same
    # order as `filters` above.
    obs['maggies'] = np.array(mags)
    # Hack, should use real flux uncertainties
    obs['maggies_unc'] = np.array(flux_uncertainty)
    # Here we mask out any NaNs or infs
    obs['phot_mask'] = np.isfinite(np.squeeze(mags))
    # We have no spectrum.
    obs['wavelength'] = None
    obs['spectrum'] = None

    # Add unessential 'bonus' info.  This will be stored in output
    obs['objid'] = objid

    # This ensures all required keys are present and adds some extra useful info
    obs = fix_obs(obs)

    return obs

# --------------
# SPS Object
# --------------
def build_sps(zcontinuous=2, compute_vega_mags=False, **extras):
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps

# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None

# -----------
# Everything
# ------------
def build_all(**kwargs):

    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


if __name__=='__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--object_redshift', type=float, default=0.0,
                        help=("Redshift for the model"))
    parser.add_argument('--objid', type=int, default=0,
                        help="zero-index row number in the table to fit.")
    parser.add_argument('--mag_in')
    parser.add_argument('--mag_unc_in')

    args = parser.parse_args()
    run_params = vars(args)
    
    # # ### EMILY'S EDITS    # Comment before uploading to Docker
    # run_params['nested_maxiter'] = 10
    # run_params['nested_walks'] = 5
    # run_params['nested_maxiter_batch'] = 10
    # run_params['niter'] = 10
    # run_params['nwalkers'] = 5
    # run_params['nested_maxcall'] = 10
    # run_params['nested_maxiter_init'] = 10
    # run_params['maxiter_init'] = 10
    # run_params['maxiter'] = 10
    # run_params['use_stop'] = False
    
    
    obs, model, sps, noise = build_all(**run_params)

    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__

    print(model)
    print(run_params)


    #hfile = setup_h5(model=model, obs=obs, **run_params)
    hfile = "{}.h5".format(args.outfile)
    output = fit_model(obs, model, sps, noise, **run_params)
    
    print("Writing h5 file from photoz_hm_params")

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1],
                      sps=sps)

    try:
        hfile.close()
    except(AttributeError):
        pass

