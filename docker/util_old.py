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
import numpy.ma as ma
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Table, MetaData, text
from sqlalchemy import String, DateTime
from sqlalchemy.types import BIGINT, FLOAT, REAL, VARCHAR, BOOLEAN, INT
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
import prospect.io.read_results as reader
from prospect.plotting.utils import sample_posterior

from psycopg2.extensions import register_adapter, AsIs
register_adapter(np.int64, AsIs)

Base = declarative_base()

# New updated columns
bands = ['g', 'r', 'i', 'z', 'w1', 'w2']
colors = ['g_r', 'i_z', 'r_i', 'r_z', 'w1_w2', 'z_w1']
# bands = ['g', 'r', 'z', 'w1', 'w2']
trac_cols = ['ls_id', 'ra', 'dec', 'type'] \
            + ['dered_mag_'+b for b in bands] \
            + ['dered_flux_'+b for b in bands] \
            + colors \
            + ['snr_'+b for b in bands] \
            + ['flux_ivar_'+b for b in bands] \
            + ['dchisq_'+str(i) for i in range(1,6)] \
            + ['rchisq_'+b for b in bands] \
            + ['sersic', 'sersic_ivar'] \
            + ['psfsize_g', 'psfsize_r', 'psfsize_z'] \
            + ['shape_r', 'shape_e1', 'shape_e2'] \
            + ['shape_r_ivar', 'shape_e1_ivar', 'shape_e2_ivar']
phot_z_cols = ['z_phot_median', 'z_phot_std', 'z_spec']

query_cols = trac_cols+phot_z_cols

theta_labels = ['zred', 'dust2', 'tau', 'tmax', 'massmet_1', 'massmet_2']

# Column labels for h5 files
h5_cols = ['ls_id'] + \
        [f'maggies_{i}' for i in range(len(bands))] + \
        [f'maggies_unc_{i}' for i in range(len(bands))] + \
        [f'maggies_fit_{i}' for i in range(len(bands))] + \
        [f'{theta}_med' for theta in theta_labels] + \
        [f'{theta}_sig_minus' for theta in theta_labels] + \
        [f'{theta}_sig_plus' for theta in theta_labels] + \
        ['chisq_maggies']

# Columns to use for ML ### TODO: fix these
use_cols = ['z_phot_median', 'chisq_maggies'] + \
            colors + \
            [f"rchisq_{band}" for band in bands] + \
            [theta+"_med" for theta in theta_labels] + \
            [theta+"_sig_diff" for theta in theta_labels]

# Columns to filter on
filter_cols = use_cols+['dered_mag_'+b for b in bands]+['ls_id', 'ra', 'dec', 'type']

dchisq_labels = [f'dchisq_{i}' for i in range(1,6)]
rchisq_labels = ['rchisq_g', 'rchisq_r', 'rchisq_z', 'rchisq_w1', 'rchisq_w2']

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
    *[Column(col, FLOAT) for col in query_cols[4:]+h5_cols[1:]], # All other cols
    Column('lensed', BOOLEAN),
    Column('id', BIGINT, primary_key=True, autoincrement=True),
    Column('lens_grade', VARCHAR(1)),
]

# Connection to database
# conn_string = 'host = nerscdb03.nersc.gov dbname=lensed_db user=lensed_db_admin'
conn_string = 'postgresql+psycopg2://lensed_db_admin@nerscdb03.nersc.gov/lensed_db'

########################## Conversion functions ##############################
def age(tage, tau):
    """
    Function to convert from tage and tau to mass-weighted age
    Note that mass-weighted age = tage - age(tage, tau)
    This only works for a delayed-tau SFH
    """
    age = 2*tau - tage**2/(tau*np.exp(tage/tau)-tau-tage)
    return age

def SFR(tage, tau, mass):
    """
    To determine SFR from delayed-tau SFH
    Takes in lists of tage, tau and mass
    Outputs SFR in M_sol/yr
    """
    psi_arr = []
    
    for i in np.arange(0, len(tage), 1):
        # for delay tau this function gives the (unnormalized) SFR 
        # for any t, tau combo in M_sun/Gyr
        tau_i = tau[i]
        sfr = lambda t,tau_i: (t) * np.exp(-t/tau_i)
        # now we numerically integrate this SFH from 0 to tage to get the mass formed
        times = np.linspace(0, tage[i], 1000)
        A = np.trapz(sfr(times, tau[i]), times)
        # and now we renormalize the formed mass to the actual mass value 
        # to get the the SFR in M_sun per Gyr 
        psi = mass[i] * sfr(tage[i], tau[i]) / A
        # if we want SFR in Msun/year
        psi /= 1e9
        psi_arr.append(psi)
    return np.array(psi_arr)

def sm(t_m, logMF):
    logtm = np.log10(t_m * 1e9)
    MF = 10**logMF
    logsm = 1.06 - 0.24 * logtm + 0.01*(logtm**2)
    
    sm = 10**logsm
    mass = sm * MF
    return np.log10(mass)

def specific_SFR(mass, z, sfr):
    # First solve for sSFR using full post. dist of mass and SFR
    # Take the mode perhaps? for redshift free
    sSFR = np.log10(np.array(sfr)/(10**np.array(mass)))
    # Find lookback time    
    type_SF = []
    # If z was fixed in the fit, t_lookback is a single value
    if type(z) == float:
        t_lookback = tage_from_tuniv(zred=z, tage_tuniv=1.0)*1e9

        for s in sSFR:
            sf_cond = np.log10(1/3/(t_lookback))
            q_cond = np.log10(1/20/(t_lookback))
            
            if s >= sf_cond:
                type_SF.append('SF')
            elif (s < sf_cond) & (s > q_cond):
                type_SF.append('T')
            else: 
                type_SF.append('Q')  
                
    else:
        for i in np.arange(0, len(z), 1):
            t_lookback = tage_from_tuniv(zred=z[i], tage_tuniv=1.0)*1e9
            
            sf_cond = np.log10(1/3/(t_lookback))
            q_cond = np.log10(1/20/(t_lookback))
            
            if sSFR[i] >= sf_cond:
                type_SF.append('SF')
            elif (sSFR[i] < sf_cond) & (sSFR[i] > q_cond):
                type_SF.append('T')
            else: 
                type_SF.append('Q')   
    
    SF_mode = scipy.stats.mode(type_SF)[0][0]
    
    return SF_mode, sSFR

def tmax_to_tage(tmax=None,redshift=None,**kwargs):
    return WMAP9.age(redshift).value*(tmax) # in Gyr

def quantiles_phot(results, model):
    # Get parameter names 
    parnames = np.array(results['theta_labels'], dtype='U20')

     # Get the arrays we need (trace, wghts)
    samples = results['chain']
    xx = samples.copy()
    wghts = results.get('weights', None)

    # Resample posterior using weights to 100k uniformly-sampled weights    
    theta_samp = sample_posterior(xx, weights=wghts, nsample=100000)

    # Change from optical depth to dust: note must have dust2 as free parameter for this to work
    if 'dust1' in model.params:
        # Calculate total dust from dust1 and dust2 contributions: AV = (dust1 + dust2)*1.086 
        # dust1 = 0.5 * dust2
        dust2 = 1.5*1.086*theta_samp[:, parnames.tolist().index('dust2')]
    else:    
        dust2 = theta_samp[:, parnames.tolist().index('dust2')]*1.086

    
    mass = 10**theta_samp[:, parnames.tolist().index('massmet_1')]
    mass_log = theta_samp[:, parnames.tolist().index('massmet_1')]

    logzsol = theta_samp[:, parnames.tolist().index('massmet_2')]
    # We want to plot dust2, t_m, tau, SFR, M, Z , Z_gas and redshift if it exists

    if 'zred' in parnames:
        # Define some variables
        zred = theta_samp[:, parnames.tolist().index('zred')]

        tmax_par = theta_samp[:, parnames.tolist().index('tmax')]
        t_age = tmax_to_tage(tmax=tmax_par,redshift=zred)
        tau = theta_samp[:, parnames.tolist().index('tau')]
        
        # Convert from tage to mass-weighted age
        t_m =  t_age - age(t_age,tau)
        # Calculate SFR
        SFR_calc = SFR(t_age, tau, mass) 

        stell_mass = sm(t_m, mass_log)   
        
        new_theta = []
        for i in np.arange(0, len(mass_log), 1):
            new_idx = [zred[i], dust2[i], t_m[i], tau[i], SFR_calc[i], stell_mass[i], logzsol[i]]
            new_theta.append(new_idx)

    else:
        # Define some variables
        t_age_par = theta_samp[:, parnames.tolist().index('tage')]
        t_age = t_age_par.copy()
        tau = theta_samp[:, parnames.tolist().index('tau')]
        
        # Convert from tage to mass-weighted age
        t_m =  t_age - age(t_age,tau)
        # Calculate SFR
        SFR_calc = SFR(t_age, tau, mass) 

        stell_mass = sm(t_m, mass_log)
        
        new_theta = []
        for i in np.arange(0, len(mass_log), 1):
            new_idx = [dust2[i], t_m[i], tau[i], SFR_calc[i], stell_mass[i], logzsol[i]]
            new_theta.append(new_idx)
            
        print(np.median(stell_mass))

    new_theta = np.array(new_theta)
    return corner.quantile(new_theta, q=[0.16, 0.50, 0.84])#, weights=TODO?

########################################################################

# def collect_lensed(path, lim=None):
#     """ Collects lensed galaxies in one CSV """
#     # Get generic path for filenames
#     key = os.path.join(path, '[0-9]*.csv')
#     # Collect filenames
#     filenames = glob.glob(key)
    
#     # Load into list
#     data = []
#     for file in filenames:
#         data.append(pd.read_csv(file))
#     # Concatenate into dataframe
#     data = pd.concat(data, ignore_index = True)
    
#     # # Save result
#     # dest = os.path.join(path, 'total_lensed.csv')
#     # data.to_csv(dest, index=False)
    
#     print(f"Total lensed: {np.sum(data['lensed'])} / {len(data)}")
    
#     return data

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
    phot = ma.masked_invalid(bf['photometry'])
    mags = ma.masked_invalid(obs['maggies'])
    mags_unc = ma.masked_invalid(obs['maggies_unc'])
    rchisq = np.sum((phot - mags)**2 / mags_unc**2) / (phot.count()-1)
    
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
        [rchisq]
    
    # Load into dataframe
    return row

def clean_and_calc(data, duplicates=False, filter_cols=filter_cols, 
                   mode='all', cut_mag=False, dropna=False):
    
    data = data.copy()
    filter_cols = [col for col in filter_cols if col in data.columns]
    
    ### Filtering for DS9 data
    if mode in ['all', 'dr10']:
        # Calculate colors
        for c in colors:
            c_ = c.split('_')
            data[c] = data['dered_mag_'+c_[0]]-data['dered_mag_'+c_[-1]]
        
        data.loc[data.z_phot_median<0, 'z_phot_median'] = None
        data.loc[data.z_phot_std<0, 'z_phot_std'] = None
        data.loc[data.z_spec<0, 'z_spec'] = None
        
        # Uncertainties
        for b in bands:
            data['flux_sigma_'+b] = 1 / np.sqrt(data['flux_ivar_'+b])
        
        # Calculate minimum dchisq
        dchisq = np.array(data[dchisq_labels])
        data['min_dchisq'] = np.min(dchisq, axis=1)

        # Calculate sum rchisq
        rchisq = np.array(data[rchisq_labels])
        data['sum_rchisq'] = np.sum(rchisq, axis=1)
        data.loc[data.sum_rchisq>100, 'sum_rchisq'] = np.nan
        
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
    if dropna:
        data.dropna(subset=filter_cols, inplace=True)
    if not duplicates:
        data.drop_duplicates(subset=['ls_id'], inplace=True)
    
    return data

def collect_all(path, db_name=None, lim=None):
    """ Merges output files into one database """
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
        engine = create_engine(conn_string, poolclass=NullPool)
        all_data.to_sql(db_name, engine, if_exists='append', index=False)
        
        # # Add primary key
        # try:
        #     with engine.connect() as conn:
        #         conn.execute(f'ALTER TABLE {db_name} ADD PRIMARY KEY (ls_id);')
        # except:
        #     raise ValueError("Could not add primary key")
    else:
        return all_data

def close_connections(engine):
    with engine.connect() as conn:
        conn.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_get_activity(NULL::integer)" \
                     " WHERE datid=(SELECT oid from pg_database where datname = 'lensed_db');")

def setup_table(conn, table_name):
    """ Sets up a new database table if one does not exist """
    # Metadata object
    meta = MetaData(conn)
    
    # Table columns
    tbl = Table(
       table_name, meta, 
       *db_cols
    )
    
    # Create table
    meta.create_all(conn, checkfirst=True)
    
    return tbl

def bookkeeping_setup(table_name, engine=None, train=False, data=None, tag=None):
    """ Start bookkeeping for a series of galaxies and a given pandas table """
    # Metadata object
    if engine is None:
        engine = create_engine(conn_string, poolclass=NullPool, future=True)
    conn = engine.connect()
    meta = MetaData(conn)
    
    # Bookkeeping table columns
    bktbl = sa.Table('bookkeeping', sa.MetaData(), autoload_with=engine)
    
    # Create table if needed
    meta.create_all(conn, checkfirst=True)
    data_tbl = setup_table(conn, table_name)
    
    colnames = [col.name for col in db_cols]
    cols = [col for col in data.columns if col in colnames]
    data = data[cols]
    print(data.columns)
    
    if data is None:
        pass
    else:
        for i,row in data.iterrows():
            stmt = data_tbl.insert().values(**row)
            result = conn.execute(stmt)
            pkey = result.inserted_primary_key[0]

            # For now just assume we have dr9 data
            stmt = bktbl.insert().values(tbl_id=pkey, 
                                  tbl_name=table_name,
                                  ls_id=row.ls_id,
                                  stage=1,
                                  train=train,
                                  tag=tag
                                )
            conn.execute(stmt)
            conn.commit()

def add_to_table(table_name, data, engine=None, train=False, tag=None):
    if engine is None:
        engine = create_engine(conn_string, poolclass=NullPool)
    
    bktbl = sa.Table('bookkeeping', sa.MetaData(), autoload_with=engine)
    tbl = sa.Table(table_name, sa.MetaData(), autoload_with=engine)
    pass

def save_to_db(path, db_name, lim=None, delete=False):
    """ Saves a prospector folder's contents to a database """
    # Get filenames in folder
    key = os.path.join(path, '[0-9]*.h5')
    h5_files = glob.glob(key)
    
    # Check for limit
    if lim and lim<len(h5_files):
        h5_files = h5_files[:lim]
    
    # Connect to database
    engine = create_engine(conn_string, poolclass=NullPool)
    
    # Make new table if one doesn't exist
    with engine.connect() as conn:
        if not engine.dialect.has_table(conn, db_name):
            setup_table(conn, db_name)
    
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
    
    conn.close()

def print_cmd(ls_id, engine=None, tag=None):
    
    if engine is None:
        engine = create_engine(conn_string, poolclass=NullPool)
    
    bktbl = sa.Table('bookkeeping', sa.MetaData(), autoload_with=engine)
    
    stmt = sa.select(bktbl).where(bktbl.c.ls_id==ls_id)
    bkdata = pd.read_sql(stmt, engine)
    
    if bkdata.empty: raise ValueError(f"LS ID not found: {ls_id}")
    elif tag is not None:
        bkdata = bkdata[bkdata.tag==tag]
    if len(bkdata) > 1:
        raise ValueError(f"Too many values found for LS ID {ls_id}, " \
                            "please provide a tag (or fix the database)")
    
    tbl_name = bkdata.tbl_name[0]
    tbl = sa.Table(tbl_name, sa.MetaData(), autoload_with=engine)
    stmt = sa.select(tbl).where(tbl.c.id==bkdata.tbl_id[0])
    gal = pd.read_sql(stmt, engine).iloc[0]
    
    return f"--mag_in=\"{list(gal[['dered_mag_'+b for b in bands]])}\" " \
          f"--mag_unc_in=\"{list(2.5 / np.log(10) / np.array([gal['snr_'+b] for b in bands]))}\" "
    

def read_table(tbl_name, engine=None):
    """ Reads a pandas table in from a database """
    
    if engine is None:
        engine = create_engine(conn_string, poolclass=NullPool)
    
    data = pd.read_sql(f"SELECT * from {tbl_name}", engine)
    
    return data

def write_table(data, db_name, if_exists='append', engine=None):
    
    if engine is None:
        engine = create_engine(conn_string, poolclass=NullPool)
    
    data.to_sql(db_name, engine, if_exists=if_exists, index=False)
    