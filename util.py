### util.py : Functions for processing classifier output
### Author : Emily Ramey
### Date : 4/19/22

import pandas as pd
import numpy as np
import glob
import os
import sys

# Old columns
cols_old = ["ls_id", "ra", "dec","type",
        "dered_mag_g","dered_mag_r","dered_mag_z",
        "dered_mag_w1", "dered_mag_w2",'unc_g','unc_r',
        'unc_z','unc_w1','unc_w2', "z_phot_median",
        "z_phot_std",'z_spec','dered_flux_g','dered_flux_r',
        'dered_flux_z','dered_flux_w1','dered_flux_w2'
       ]

# Previous updated columns
cols_middle = ["ls_id", "ra", "dec","type",
        "dered_mag_g","dered_mag_r","dered_mag_z",
        "dered_mag_w1", "dered_mag_w2",'snr_g','snr_r',
        'snr_z','snr_w1','snr_w2', "z_phot_median",
        "z_phot_std",'z_spec','dered_flux_g','dered_flux_r',
        'dered_flux_z','dered_flux_w1','dered_flux_w2', 
        'dchisq_1', 'dchisq_2', 'dchisq_3', 'dchisq_4', 'dchisq_5',
        'rchisq_g', 'rchisq_r', 'rchisq_z', 'rchisq_w1', 'rchisq_w2',
        'psfsize_g', 'psfsize_r', 'psfsize_z', 'sersic', 'sersic_ivar',
        'shape_e1', 'shape_e1_ivar', 'shape_e2', 'shape_e2_ivar', 
        'shape_r', 'shape_r_ivar',
       ]

# New updated columns
bands = ['g', 'r', 'z', 'w1', 'w2']
trac_cols = ['ls_id', 'ra', 'dec', 'type'] \
            + ['dered_mag_'+b for b in bands] \
            + ['dered_flux_'+b for b in bands] \
            + ['snr_'+b for b in bands] \
            + ['flux_ivar_'+b for b in bands] \
            + ['dchisq_'+str(i) for i in range(1,6)] \
            + ['rchisq_'+b for b in bands] \
            + ['sersic', 'sersic_ivar'] \
            + ['psfsize_g', 'psfsize_r', 'psfsize_z'] \
            + ['shape_r', 'shape_e1', 'shape_e2'] \
            + ['shape_r_ivar', 'shape_e1_ivar', 'shape_e2_ivar']
phot_z_cols = ['z_phot_median', 'z_phot_std', 'z_spec']

cols_new = trac_cols+phot_z_cols


bands = ['g', 'r', 'z', 'w1', 'w2']
use_cols = ['dered_mag_g', 'dered_mag_r', 'dered_mag_z', 'dered_mag_w1', 'dered_mag_w2', 
            'z_phot_median', 'min_dchisq']
filter_cols = use_cols+['unc_'+b for b in bands]+['flux_ivar_'+b for b in bands]+['ls_id', 'ra', 'dec', 'type']
dchisq_labels = [f'dchisq_{i}' for i in range(1,6)]
rchisq_labels = ['rchisq_g', 'rchisq_r', 'rchisq_z', 'rchisq_w1', 'rchisq_w2']

def collect_lensed(path):
    """ Collects lensed galaxies in one CSV """
    # Get generic path for filenames
    key = os.path.join(path, '[0-9]*.csv')
    # Collect filenames
    filenames = glob.glob(key)
    
    # Load into list
    data = []
    for file in filenames:
        data.append(pd.read_csv(file))
    # Concatenate into dataframe
    data = pd.concat(data, ignore_index = True)
    
    # # Save result
    # dest = os.path.join(path, 'total_lensed.csv')
    # data.to_csv(dest, index=False)
    
    print(f"Total lensed: {np.sum(data['lensed'])} / {len(data)}")
    
    return data

def collect_gals(path):
    """ Collects galaxy data into a CSV """
    # Get generic path for filenames
    key = os.path.join(path, 'galaxies_*.csv')
    filenames = glob.glob(key)
    
    # Load into list
    data = []
    for file in filenames:
        df = pd.read_csv(file, index_col=0, header=None)
        df.columns = np.arange(len(df.columns))
        data.append(df)
    
    # Concatenate data
    data = pd.concat(data, ignore_index=True)
    
    # Fix columns
    if len(data.columns)==len(cols_old):
        data.columns = cols_old
    elif len(data.columns)==len(cols_middle):
        data.columns = cols_middle
    elif len(data.columns)==len(cols_new):
        data.columns = cols_new
    else:
        raise ValueError("The given csv files do not match a known column scheme.")
    
    # Return result
    return data

def filter_data(data):
    
    # Colors
    data['g-r'] = data['dered_mag_g']-data['dered_mag_r']
    data['r-z'] = data['dered_mag_r']-data['dered_mag_z']
    data['r-w1'] = data['dered_mag_r']-data['dered_mag_w1']
    data['r-w2'] = data['dered_mag_r']-data['dered_mag_w2']
    
    # Uncertainties
    for b in bands:
        data['unc_'+b] = 1 / data['snr_'+b]
        data['flux_sigma_'+b] = 1 / np.sqrt(data['flux_ivar_'+b])
    
    # Calculate minimum dchisq
    dchisq = np.array(data[dchisq_labels])
    data['min_dchisq'] = np.min(dchisq, axis=1)
    
    # Calculate sum rchisq
    rchisq = np.array(data[rchisq_labels])
    data['sum_rchisq'] = np.sum(rchisq, axis=1)
    
    # Calculate abs mag in r band
    dm = 5\*np.log10(300000*data.z_phot_median/70)+25
    data['abs_mag_r'] = data.dered_mag_r - dm
    
    # Remove bad / duplicate entries
    data = data[data.type!='DUP']
    data = data[data.type!='PSF']
    data = data[data.dered_mag_r<=22]
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna(subset=filter_cols)
    data.drop_duplicates(subset=['ls_id'], inplace=True)
    
    return data

def merge_gals(path):
    """ Merges output files into one CSV """
    # Get both data tables
    gals = collect_gals(path)
    lensed = collect_lensed(path)
    
    # Merge dataframes
    