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

# Try it with a more standard fit
from sedpy.observate import load_filters
from prospect.utils.obsutils import fix_obs

gal_data = classify.query_galaxy(ra=gal['ra'], dec=gal['dec'])
gal_data = util.clean_and_calc(gal_data, mode='dr9').iloc[0]
print(gal_data)
    # query =[f"""SELECT {classify.query_cols} FROM ls_dr9.tractor AS trac 
    #         INNER JOIN ls_dr9.photo_z AS phot_z ON trac.ls_id = phot_z.ls_id 
    #         WHERE trac.ls_id = {ls_id} """]
    # query = ''.join(query)
    # try:
    #     result = qc.query(sql=query) # result as string
    #     data = convert(result, "pandas") # result as dataframe
    #     if data.empty:
    #         raise ValueError(f"No objects matching ls_id {ls_id} in DESI Legacy Survey")
    #     gal_data = util.clean_and_calc(data, mode='dr9').iloc[0]
    # except:
    #     raise ValueError("Something went wrong")

# Get galaxy data
bands = ['g', 'r', 'i', 'z', 'w1', 'w2']
sdss = ['sdss_{}0'.format(b) for b in ['g','r','i','z']]
wise = ['wise_w1', 'wise_w2']
filternames = sdss + wise

filters = load_filters(filternames)
maggies = np.array([10**(-.4*gal_data[f'dered_mag_{b}']) for b in bands])
magerr = [ 2.5 / (np.log(10) * gal_data['dered_flux_'+b] * 
                        np.sqrt(gal_data['flux_ivar_'+b])) for b in bands]
magerr = np.clip(magerr, 0.05, np.inf)

# Build obs
obs = dict(wavelength=None, spectrum=None, unc=None, redshift=gal_data['z_phot_median'],
           maggies=maggies, maggies_unc=magerr * maggies / 1.086, filters=filters)
obs = fix_obs(obs)
print(obs)

# Build model
from prospect.models.templates import TemplateLibrary
from prospect.models import SpecModel
model_params = TemplateLibrary["parametric_sfh"]
model_params.update(TemplateLibrary["nebular"])
model_params["zred"]["init"] = obs["redshift"]

model = SpecModel(model_params)
noise_model = (None, None)

print(model)

from prospect.sources import CSPSpecBasis
sps = CSPSpecBasis(zcontinuous=1)
print(sps.ssp.libraries)

from prospect.fitting import lnprobfn, fit_model
fitting_kwargs = dict(nlive_init=400, nested_method="rwalk", nested_target_n_effective=1000, nested_dlogz_init=0.05)
output = fit_model(obs, model, sps, optimize=False, dynesty=True, lnprobfn=lnprobfn, 
                   noise=noise_model, **fitting_kwargs)
result, duration = output["sampling"]

from prospect.io import write_results as writer
hfile = "./dr10_lensed_param_csp.h5"
writer.write_hdf5(hfile, {}, model, obs,
                 output["sampling"][0], None,
                 sps=sps,
                 tsample=output["sampling"][1],
                 toptimize=0.0)