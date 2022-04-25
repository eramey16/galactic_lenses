### util.py : Functions for processing classifier output
### Author : Emily Ramey
### Date : 4/19/22

import pandas as pd
import numpy as np
import glob
import os
import sys

cols = ["ls_id", "ra", "dec","type",
        "dered_mag_g","dered_mag_r","dered_mag_z",
        "dered_mag_w1", "dered_mag_w2",'unc_g','unc_r',
        'unc_z','unc_w1','unc_w2', "z_phot_median",
        "z_phot_std",'z_spec','dered_flux_g','dered_flux_r',
        'dered_flux_z','dered_flux_w1','dered_flux_w2'
       ]

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

def collect_gals(path, tag='tst'):
    """ Collects galaxy data into a CSV """
    # Get generic path for filenames
    key = os.path.join(path, 'galaxies_*.csv')
    filenames = glob.glob(key)
    
    # Load into list
    data = []
    for file in filenames:
        data.append(pd.read_csv(file, index_col=0, names=cols))
    data = pd.concat(data, ignore_index=True)
    
    data['tag'] = tag
    
    # Return result
    return data

def merge_gals(path, tag='tst'):
    """ Merges output files into one CSV """
    # Get both data tables
    gals = collect_gals(path, tag=tag)
    lensed = collect_lensed(path)
    
    # Merge dataframes
    