import os
import sys

sys.path.insert(0, '/global/cfs/cdirs/m2218/eramey16/frankenblast-host')

root = '/global/cfs/cdirs/m2218/eramey16'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide all GPUs from TensorFlow
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['SPS_HOME'] = f'/global/homes/e/eramey16/fsps'
os.environ['PROST_PATH'] = f'{root}/Prost/src/astro_prost/data'
os.environ['SBIPP_ROOT'] = f'{root}/frankeblast-host/data/SBI'
os.environ['SBIPP_PHOT_ROOT'] = f'{root}/frankenblast-host/data/sbipp_phot'
os.environ['SBIPP_TRAINING_ROOT'] = f'{root}/frankenblast-host/data/sbi_training_sets'
os.environ['SED_OUTPUT_ROOT'] = f'/pscratch/sd/e/eramey16/data/frankenblast/sed'
os.environ['CUTOUT_ROOT'] = '/pscratch/sd/e/eramey16/data/frankenblast/cutouts/'

from datetime import datetime
import pandas as pd
from scipy.stats import gamma, halfnorm, uniform
from astropy.cosmology import WMAP9 as cosmo
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.cosmology import WMAP9 as cosmo
from astropy.coordinates import SkyCoord
import astropy.units as u
from classes import Transient, Host, Filter
import csv
import shutil
import glob
import pathlib
import astropy.units as u
from urllib.error import HTTPError
from mpire import WorkerPool
import time
import numpy as np
import requests
import get_host
import pandas
from get_host_images import download_and_save_cutouts, survey_list, get_cutouts
from create_apertures import construct_aperture
import sys
from do_photometry import do_global_photometry
import os
# from sedpy.observate import load_filters
from fit_host_sed import run_training_set, build_obs, build_model, fit_sbi_pp, build_model_nonparam
import prospect.io.read_results as reader
from prospect.sources import FastStepBasis
from prospect.utils.plotting import quantile
import sedpy
from postprocess_sbi import save_all as psbi_save
import  matplotlib.pyplot as plt
from fit_host_sed import fit_host, maggies_to_asinh
from prospect.plotting.utils import sample_posterior
import corner as triangle
from prospect.io.read_results import traceplot, subcorner
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import partial
import matplotlib.ticker as mticker
from mwebv_host import get_mwebv

################
data = pd.read_csv('/pscratch/sd/e/eramey16/data/monocle/test_data/lensed_dr10_all.dat')
gal = data.iloc[100]

###############
transient_class = Transient(
    name=str(gal.ls_id),
    coordinates = SkyCoord(ra=gal.ra*u.deg, dec=gal.dec*u.deg, frame='icrs'),
    transient_redshift= gal.z_phot_median,
    milkyway_dust_reddening=None,  # Initially None
    )


# Milky-Way reddening from transient coordinates
transient_class.milkyway_dust_reddening = get_mwebv(transient_class)

##############
output_dir = f'{root}/frankenblast-host/data/prostdb/' # where to save data
prost_results = get_host.run_prost(transient_class, output_dir, save=True)

###############
# Get best host..
best_host = Host(name=transient_class.name, 
				# redshift=transient_class.transient_redshift, 
                photometric_redshift = transient_class.transient_redshift,
                host_prob = prost_results['host_prob'],
                missedcat_prob = prost_results['missedcat_posterior'],
                smallcone_prob = prost_results['smallcone_posterior'],
                association_catalog = prost_results['best_cat'],
				milkyway_dust_reddening=transient_class.milkyway_dust_reddening,
				coordinates=SkyCoord(ra=prost_results['host_ra'] * u.deg, dec=prost_results['host_dec'] * u.deg, frame='icrs'))

# best_host = Host(name=str(gal.ls_id), photometric_redshift=gal.z_phot_median, 
#                  milkyway_dust_reddening=transient_class.milkyway_dust_reddening, 
#                  coordinates=SkyCoord(ra=gal.ra*u.deg, dec=gal.dec*u.deg, frame='icrs'))

transient_class.host = best_host

###############
# Make filters
survey_metadata_path = f"{root}/frankenblast-host/data/survey_frankenblast_metadata.yml"
surveys = survey_list(survey_metadata_path)
filters = Filter.all()
print(filters)

################
DOWNLOAD_CUTOUTS = True

if DOWNLOAD_CUTOUTS:
    download_and_save_cutouts(transient_class, filters=Filter.all())


###############
cutouts = transient_class.cutouts
global_apertures = []

for cutout in cutouts:
    print(cutout['filter'].name)
    aperture = construct_aperture(cutout, transient_class.host.coordinates)
    global_apertures.append(aperture)

transient_class.global_apertures = global_apertures



all_phot = []
for i in np.arange(0, len(cutouts),1):
    filt = cutouts[i]['filter']
    apr = transient_class.global_apertures[i] 
    phot = do_global_photometry(transient_class, filter=filt, aperture=apr, 
                fwhm_correction=False, show_plot=False)
    all_phot.append(phot)


transient_class.host_photometry = all_phot

################
phot_dictionary = {'names': transient_class.name,
                   'phot': transient_class.host_photometry,
                   'apertures': transient_class.global_apertures,
                   'filternames': [cutout['filter'].name for cutout in cutouts]
                  }

############
# Get roots
SBIPP_ROOT = os.environ.get("SBIPP_ROOT")
SBIPP_PHOT_ROOT = os.environ.get("SBIPP_PHOT_ROOT")
SBIPP_TRAINING_ROOT = os.environ.get("SBIPP_TRAINING_ROOT")
SED_OUTPUT_ROOT = os.environ.get("SED_OUTPUT_ROOT")

##############
if best_host.redshift == None:
    # Set sbi params
    sbi_params = {
                "anpe_fname_global": f"{SBIPP_ROOT}/SBI_model_zfree_GPD2W_global.pt",  # trained sbi model
                "train_fname_global": f"{SBIPP_PHOT_ROOT}/sbi_phot_zfree_GPD2W_global.h5",  # training set
                "nhidden": 500,  # architecture of the trained density estimator
                "nblocks": 15,  # architecture of the trained density estimator
            }
    
    train_fname = 'zfree_GPD2W'
else:
    # Set sbi params
    sbi_params = {
                "anpe_fname_global": f"{SBIPP_ROOT}/SBI_model_zfix_GPD2W_global.pt",  # trained sbi model
                "train_fname_global": f"{SBIPP_PHOT_ROOT}/sbi_phot_zfix_GPD2W_global.h5",  # training set
                "nhidden": 500,  # architecture of the trained density estimator
                "nblocks": 15,  # architecture of the trained density estimator
            }
    
    train_fname = 'zfix_GPD2W'

print(train_fname)

##############
phot_filters = np.array(phot_dictionary['filternames'])

# Set filter objects to transient class
available_filters = []
update_phot = []
for filternames in phot_filters:
    filter_obj = next((f for f in filters if f.name == filternames), None)
    if filter_obj:
        phot_flux = np.array(phot_dictionary['phot'])[np.where(phot_filters == filternames)][0]['flux']
        phot_err = np.array(phot_dictionary['phot'])[np.where(phot_filters == filternames)][0]['flux_error']
        # Make sure no bad photometry is in here
        if phot_flux is not None:
            if phot_flux > 0:
                available_filters.append({"filter": filter_obj})
                update_phot.append(np.array(phot_dictionary['phot'])[np.where(phot_filters == filternames)][0])
        

transient_class.host_phot_filters = np.array(available_filters)
transient_class.host_photometry = np.array(update_phot)

###############
# Run SBI
prev_dir = os.getcwd()
os.chdir(f'{root}/frankenblast-host')
start = time.time()

flag=fit_host(transient_class, sbi_params=sbi_params, fname=train_fname, all_filters = filters, mode='test', sbipp=True, 
            aperture_type='global', aperture=transient_class.global_apertures, save=True)


end = time.time()
length = end - start

print("It took", length/60, "minutes to run SBIPP!")
print('Host SED fit completed.')
os.chdir(prev_dir)



