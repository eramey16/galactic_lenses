import os
import numpy as np
os.environ["SPS_HOME"] = '/global/homes/e/eramey16/fsps/'
h5_file = os.path.expandvars("10995462702105256_extra_samples.h5")

from docker import param_monocle as param
from docker import classify
import prospect.io.read_results as reader

import h5py
import pickle
import dynesty

########################

res1, _, _ = reader.results_from(h5_file)

mag_unc_1 = np.array([0.005395851583585794, 0.0034580493543346622, 0.004425482546800575,
                 0.004152162899013791, 0.011291867461568727, 0.04210798998385776])*5
args_1 = {
    "mag_in": [20.717947, 19.274029, 18.668655, 18.314737, 17.77136, 18.470882],
    "mag_unc_in": mag_unc_1,
    "object_redshift": None,
}
run_params = param.run_params
obs_1, model_1, sps_1, noise_model_1 = param.load_all(args=args_1, **run_params)
spec_noise_1, phot_noise_1 = noise_model_1

##########################

mag_unc_2 = np.array([0.005395851583585794, 0.0034580493543346622, 0.004425482546800575,
                 0.004152162899013791, 0.011291867461568727, 0.04210798998385776])
args_2 = {
    "mag_in": [20.717947, 19.274029, 18.668655, 18.314737, 17.77136, 18.470882],
    "mag_unc_in": mag_unc_2,
    "object_redshift": None,
}

obs_2, model_2, sps_2, noise_model_2 = param.load_all(args=args_2, **run_params)
spec_noise_2, phot_noise_2 = noise_model_2

##########################

def loglikelihood1(theta_vec):
    return param.lnprobfn(theta_vec, model=model_1, obs=obs_1, def_sps=sps_1,
                          def_noise_model=noise_model_1)
def loglikelihood2(theta_vec):
    return param.lnprobfn(theta_vec, model=model_2, obs=obs_2, def_sps=sps_2, 
                          def_noise_model=noise_model_2)

###########################

with open('10995462702105256_extra_samples.pkl', 'rb') as file:
    dres = pickle.load(file)

###########################

N = dres['samples'].shape[0]
logl2 = [loglikelihood2(dres['samples'][i]) for i in range(N)]
dres_rwt = dynesty.utils.reweight_run(dres, logp_new=logl2)

#############################

import prospect.io.write_results as writer
writer.write_hdf5("10995462702105256_extra_samples_rwt.h5", {}, model_2, obs_2, dres_rwt, None, sps=sps_2, tsample=None, toptimize=0.0)