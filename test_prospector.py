import os
os.environ["SPS_HOME"] = '/global/homes/e/eramey16/fsps/'

import fsps
import dynesty
import sedpy
import h5py, astropy
import numpy as np
import pandas as pd
from docker import classify
from docker import util
from docker import util
from dl import queryClient as qc, helpers
from dl.helpers.utils import convert

from prospect.fitting import fit_model
from prospect.sources import CSPSpecBasis
from sedpy.observate import load_filters
from prospect.utils.obsutils import fix_obs
from prospect.fitting import lnprobfn, fit_model
from prospect.io import write_results as writer
from prospect.models.templates import TemplateLibrary
from prospect.models import SpecModel
from prospect.utils import prospect_args
from functools import partial

# Define bands
bands = ['g', 'r', 'i', 'z', 'w1', 'w2']
sdss = ['sdss_{}0'.format(b) for b in ['g','r','i','z']]
wise = ['wise_w1', 'wise_w2']
filternames = sdss + wise

# Test galaxies
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

def load_test():
    gal = iptf16

    gal_data = classify.query_galaxy(ra=gal['ra'], dec=gal['dec'])
    gal_data = util.clean_and_calc(gal_data, mode='dr9').iloc[0]
    print(gal_data)
    
    # Calculate maggies
    M_AB = np.array([gal_data[f'mag_{b}'] for b in bands])
    M_AB_unc = 2.5 / np.log(10) / np.array([gal_data['snr_'+b] for b in bands])
    
    return M_AB, M_AB_unc, gal_data['z_phot_median']

def run_prospector(M_AB, M_AB_unc, z=None):
    filters = load_filters(filternames)
    import pdb; pdb.set_trace()
    M_AB = np.array(M_AB)
    M_AB_unc = np.array(M_AB_unc)
    
    # Get maggies from AB Magnitude
    mags = 10**(-.4*M_AB)

    # Calculate lower limit of uncertainty
    mag_down = [x-y for (x,y) in zip(M_AB, M_AB_unc)]
    flux_down = [10**(-0.4*x) for x in mag_down]
    flux_uncertainty = [y-x for (x,y) in zip(mags, flux_down)]

    # magerr = [ 2.5 / (np.log(10) * gal_data['dered_flux_'+b] * 
    #                         np.sqrt(gal_data['flux_ivar_'+b])) for b in bands]
    # magerr = np.clip(magerr, 0.05, np.inf)

    # Build obs
    obs = dict(wavelength=None, spectrum=None, unc=None, redshift=z,
               maggies=np.array(mags), maggies_unc=np.array(flux_uncertainty), phot_mask=np.isfinite(np.squeeze(mags)),
               filters=filters)
    obs = fix_obs(obs)
    print(obs)

    # Build model
    model_params = TemplateLibrary["parametric_sfh"]
    model_params.update(TemplateLibrary["nebular"])
    model_params["zred"]["init"] = obs["redshift"]

    model = SpecModel(model_params)
    noise_model = (None, None)

    print(model)
    
    sps = CSPSpecBasis(zcontinuous=1)
    print(sps.ssp.libraries)

    # Set up MPI communication
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
    lnprobfn_fixed = partial(lnprobfn, sps=sps)

    run_params = dict(nlive_init=400, nested_method="rwalk", nested_target_n_effective=1000, nested_dlogz_init=0.05)

    if withmpi:
        run_params["using_mpi"] = True
        with MPIPool() as pool:

            # The dependent processes will run up to this point in the code
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            nprocs = pool.size
            # The parent process will oversee the fitting
            output = fit_model(obs, model, sps, noise=noise_model, pool=pool, queue_size=nprocs, 
                               dynesty=True, lnprobfn=lnprobfn_fixed, **run_params)
    else:
        # without MPI we don't pass the pool
        output = fit_model(obs, model, sps, noise=noise_model, dynesty=True, lnprobfn=lnprobfn, **run_params)

    # output = fit_model(obs, model, sps, optimize=False, dynesty=True, lnprobfn=lnprobfn, 
    #                    noise=noise_model, **fitting_kwargs) 
    result, duration = output["sampling"]

    hfile = "./test.h5"
    writer.write_hdf5(hfile, {}, model, obs,
                     output["sampling"][0], output['optimization'][0],
                      sps=sps,
                     tsample=output["sampling"][1],
                     toptimize=output['optimization'][1])

if __name__=='__main__':
    parser = prospect_args.get_parser()
    parser.add_argument('--object_redshift', type=float, default=None,
                        help=("Redshift for the model"))
    parser.add_argument('--mag_in', type=str)
    parser.add_argument('--mag_unc_in', type=str)
    
    args = parser.parse_args()
    
    if args.mag_in==None:
        M_AB, M_AB_unc, object_redshift = load_test()
    else:
        mag_in, mag_unc_in, object_redshift = args.mag_in, args.mag_unc_in, args.object_redshift
        
        M_AB = [float(x) for x in (mag_in[1:-1].split(','))]
        M_AB_unc = [float(x) for x in (mag_unc_in[1:-1].split(','))]
        object_redshift = args.object_redshift
    
    print(M_AB, M_AB_unc)
    
    run_prospector(M_AB, M_AB_unc, object_redshift)