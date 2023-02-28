### util.py : Functions for processing classifier output
### Author : Emily Ramey
### Date : 4/19/22

import pandas as pd
import numpy as np
import glob
import os
import sys
import h5py
import psycopg2
import corner
from sqlalchemy import create_engine, Column, Table, MetaData
from sqlalchemy.types import BIGINT, FLOAT, REAL, VARCHAR
import prospect.io.read_results as reader

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

theta_labels = ['dust2', 'tau', 'massmet_1', 'massmet_2', 'logtmax']

# Column labels for h5 files
h5_cols = ['ls_id'] + \
        [f'maggies_{i}' for i in range(5)] + \
        [f'maggies_unc_{i}' for i in range(5)] + \
        [f'maggies_fit_{i}' for i in range(5)] + \
        [f'{theta}_med' for theta in theta_labels] + \
        [f'{theta}_sig_minus' for theta in theta_labels] + \
        [f'{theta}_sig_plus' for theta in theta_labels] + \
        ['chisq_maggies']

# Setup for columns
bands = ['g', 'r', 'z', 'w1', 'w2']

# Columns to use
use_cols = ['dered_flux_'+b for b in bands] + ['g/r', 'r/z', 'r/w1', 'r/w2'] + ['z_phot_median', 'min_dchisq']
# Columns to filter on
filter_cols = use_cols+['flux_ivar_'+b for b in bands]+['ls_id', 'ra', 'dec', 'type']
dchisq_labels = [f'dchisq_{i}' for i in range(1,6)]
rchisq_labels = ['rchisq_g', 'rchisq_r', 'rchisq_z', 'rchisq_w1', 'rchisq_w2']

res_cols = []
obs_cols = []
model_cols = [
    'zred',
    'mass',
    'logzsol',
    'sfh',
    'tage',
    'imf_type',
    'dust_type',
    'pmetals',
    'dust1',
    'dust_index',
    'gas_logz',
    'gas_logu',
    'dust2',
    'tau',
    'massmet',
    'logtmax'
]

# Get database columns from file columns
db_cols = [
    Column('ls_id', BIGINT, nullable=False),
    Column('ra', FLOAT),
    Column('dec', FLOAT),
    Column('type', VARCHAR(4)),
    *[Column(col, FLOAT) for col in cols_new[4:]+h5_cols[1:]], # All other cols
    Column('id', BIGINT, primary_key=True, autoincrement=True)
]

# Connection to database
# conn_string = 'host = nerscdb03.nersc.gov dbname=lensed_db user=lensed_db_admin'
conn_string = 'postgresql+psycopg2://lensed_db_admin@nerscdb03.nersc.gov/lensed_db'

def collect_lensed(path, lim=None):
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

def collect_gals(path, lim=None):
    """ Collects galaxy data into a CSV """
    if not os.path.exists(path):
        raise ValueError("Incorrect path.")
    # Get generic path for filenames
    key = os.path.join(path, '[0-9]*.csv')
    filenames = glob.glob(key)
    
    # Check for limit
    if lim and lim<len(filenames):
        filenames = filenames[:lim]
    
    # Load into list
    data = []
    for file in filenames:
        try:
            df = pd.read_csv(file)
            data.append(df)
        except:
            print(f"File {file} did not parse correctly.")
    
    # Concatenate data
    data = pd.concat(data, ignore_index=True)
    
    # Return result
    return data

def collect_h5(path, lim=None):
    """ Collects h5 data into a large file """
    if not os.path.exists(path):
        raise ValueError("Incorrect path.")
    # Get all paths for filenames
    key = os.path.join(path, '[0-9]*.h5')
    filenames = glob.glob(key)
    
    # Check for limit
    if lim and lim<len(filenames):
        filenames = filenames[:lim]
    
    # Append to master file
    model_file = os.path.join(path, 'model_data.h5')
    model_data = h5py.File(model_file, 'w')
    
    # Create empty dataframe
    all_data = pd.DataFrame(index=range(len(filenames)), columns=h5_cols)
    
    # Load data
    for i,file in enumerate(filenames):
        try:
            # Get ls_id from file path
            ls_id = file.split('/')[-1][:-3]

            # Read from file
            data = reader.results_from(file)
            
            # Save results in dataframe
            row = load_data(data)
            all_data.iloc[i] = [ls_id]+row
        except:
            print(f"Failed to process {file}")
            continue
        
        # Save model info in an h5 file 
        if i==0:
            # Add initial model parameters
            params = data[2].params
            for col in model_cols:
                model_data.create_dataset(col, data=params[col])
            model_data.create_dataset('theta', data=data[2].theta)
            model_data.close()
    
    # Output
    return all_data

def load_data(gal_results):
    """ Loads data from H5 prospector results into new dataframe """
    # Separate data structures
    res, obs, model = gal_results
    bf = res['bestfit']
    
    # Calculate secondary results
    # Get reduced chi^2 between maggies and fit
    red_chisq = np.sum((bf['photometry'] - obs['maggies'])**2 / obs['maggies_unc']**2) / (len(bf['photometry'])-1)
    
    # Define function for quantiles (0.16, 0.5, 0.84)
    def calc_quantiles(x):
        """ Calculates .16, .5, and .84 quantiles on slice x """
        return corner.quantile(x, q=[.16, .5, .84], weights=res['weights'])
    
    # Get quantiles & calc lower and upper sigmas
    quantiles = np.apply_along_axis(calc_quantiles, 0, res['chain'])
    mid = quantiles[1,:]
    sig_minus = mid-quantiles[0,:]
    sig_plus = quantiles[2,:]-mid
    
    # Set up data structure to load into df
    row = list(obs['maggies']) + \
        list(obs['maggies_unc']) + \
        list(bf['photometry']) + \
        list(mid) + \
        list(sig_minus) + \
        list(sig_plus) + \
        [red_chisq]
    
    # Load into dataframe
    return row

def clean_and_calc(data, duplicates=False, filter_cols=filter_cols, 
                   mode='all', cut_mag=False):
    
    ### Filtering for DS9 data
    if mode in ['all', 'dr9']:
        # Calculate colors
        data['g/r'] = data['dered_flux_g']/data['dered_flux_r']
        data['r/z'] = data['dered_flux_r']/data['dered_flux_z']
        data['r/w1'] = data['dered_flux_r']/data['dered_flux_w1']
        data['r/w2'] = data['dered_flux_r']/data['dered_flux_w2']
        
        # Uncertainties
        for b in bands:
            # data['unc_'+b] = 1 / data['snr_'+b]
            data['flux_sigma_'+b] = 1 / np.sqrt(data['flux_ivar_'+b])
        
        # Calculate minimum dchisq
        dchisq = np.array(data[dchisq_labels])
        data['min_dchisq'] = np.min(dchisq, axis=1)

        # Calculate sum rchisq
        rchisq = np.array(data[rchisq_labels])
        data['sum_rchisq'] = np.sum(rchisq, axis=1)
        
        # Calculate abs mag in r band
        dm = 5*np.log10(300000*data.z_phot_median/70)+25
        data['abs_mag_r'] = data.dered_mag_r - dm
        
        # Remove bad / duplicate entries
        data = data[data.type!='DUP']
        data = data[data.type!='PSF']
        
        if cut_mag:
            data = data[data.dered_mag_r<=22]
    
    ### Calculations for prospector data
    if mode in ['all', 'prospector']:
        for theta in theta_labels:
            data[theta+"_sig_diff"] = data[theta+"_sig_plus"]-data[theta+"_sig_minus"]
    
    # Remove bad / duplicate entries
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna(subset=filter_cols)
    if not duplicates:
        data.drop_duplicates(subset=['ls_id'], inplace=True)
    
    return data

def collect_all(path, db_name=None, lim=None):
    """ Merges output files into one CSV """
    # Get both data tables
    basic = collect_gals(path, lim)
    # lensed = collect_lensed(path)
    prospector = collect_h5(path, lim)
    prospector['ls_id'] = prospector.ls_id.astype(int) # Fix dtype
    
    # Merge dataframes
    all_data = basic.merge(prospector, on='ls_id')
    
    if db_name: # Save in database
        # Assign other column types
        # db_cols.update({col: REAL for col in list(all_data.columns)[4:]})
        
        # Connect to database
        engine = create_engine(conn_string)
        all_data.to_sql(db_name, engine, if_exists='append', index=False)
        
        # # Add primary key
        # try:
        #     with engine.connect() as conn:
        #         conn.execute(f'ALTER TABLE {db_name} ADD PRIMARY KEY (ls_id);')
        # except:
        #     raise ValueError("Could not add primary key")
    else:
        return all_data

def setup_table(table_name):
    """ Sets up a new database table if one does not exist """
    # Metadata objects
    meta = MetaData()
    engine = create_engine(conn_string)
    
    # Table columns
    tbl = Table(
       table_name, meta, 
       *db_cols
    )
    
    # Create table
    meta.create_all(engine)

def save_to_db(path, db_name, lim=None, delete=False):
    """ Saves a prospector folder's contents to a database """
    # Get filenames in folder
    key = os.path.join(path, '[0-9]*.h5')
    h5_files = glob.glob(key)
    
    # Check for limit
    if lim and lim<len(h5_files):
        h5_files = h5_files[:lim]
    
    # Connect to database
    engine = create_engine(conn_string)
    
    # Make new table if one doesn't exist
    with engine.connect() as conn:
        if not engine.dialect.has_table(conn, db_name):
            setup_table(db_name)
    
    # Loop through filenames
    for h5_file in h5_files:
        # Get ls id and prospector file
        ls_id = h5_file.split('/')[-1][:-3]
        basic_file = os.path.join(path, f"{ls_id}.csv")
        
        try:
            # Read in basic data
            df = pd.read_csv(basic_file)
            # Add new columns
            df[h5_cols[1:]] = None

            # Read in h5 data
            h5_data = reader.results_from(h5_file)
            # Get results
            h5_row = load_data(h5_data)
        except:
            continue
        
        # Put in dataframe
        df.loc[0, h5_cols[1:]] = h5_row
        
        # Only allowed columns
        cols = [col.name for col in db_cols if col.name!='id']
        df = df[cols]
        
        # Save to database
        df.to_sql(db_name, engine, if_exists='append', index=False)
        
        # Delete files
        if delete:
            os.remove(h5_file)
            os.remove(basic_file)

def read_table(db_name):
    """ Reads a pandas table in from a database """
    
    engine = create_engine(conn_string)
    
    data = pd.read_sql(f"SELECT * from {db_name}", engine)
    
    return data

def write_table(data, db_name, if_exists='append'):
    
    engine = create_engine(conn_string)
    
    data.to_sql(db_name, engine, if_exists=if_exists, index=False)
    