#import statements
from dl import queryClient as qc, helpers
from dl import authClient as ac
from dl import storeClient as sc
from dl.helpers.utils import convert
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.ensemble import GradientBoostingClassifier
import h5py
import requests
import subprocess
import random
import datetime
import argparse
import glob
import sys
import os
import math
import time

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import OperationalError
import util
import pickle
import pandas as pd
import numpy as np

import prospect.io.read_results as reader

# from psycopg2.extensions import register_adapter, AsIs
# register_adapter(np.int64, AsIs)

bands = ['g', 'r', 'i', 'z', 'w1', 'w2']
colors = ['g_r', 'i_z', 'r_i', 'r_z', 'w1_w2', 'z_w1']
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
phot_z_cols = ['z_phot_median_i AS z_phot_median', 'z_phot_std_i AS z_phot_std', 'z_spec']
all_cols = ['trac.'+col for col in trac_cols] + ['phot_z.'+col for col in phot_z_cols]

query_cols = ', '.join(all_cols)
######################################################
output_dir = '/monocle/exports/' # Uncomment these two lines before pushing to docker
input_dir = '/monocle/'
# output_dir = '/global/cscratch1/sd/eramey16/gradient/' # Comment these two lines before pushing to docker
# input_dir='/global/cscratch1/sd/eramey16/gradient/'
prosp_file = 'param_monocle.py'
one_as = 0.0002777777778 # degrees per arcsecond

#given an RA and DEC, pull magnitudes, magnitude uncertainties, redshifts from NOAO

import numpy
from psycopg2.extensions import register_adapter, AsIs
def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
register_adapter(numpy.float64, addapt_numpy_float64)
register_adapter(numpy.int64, addapt_numpy_int64)

def _sub_query(query):
    # Send query to NOIRlab
    try:
        result = qc.query(sql=query) # result as string
        data = convert(result, "pandas") # result as dataframe
        return data
    # Connection error(s)
    except requests.exceptions.ConnectionError as e:
        raise Exception(f"ConnectionError: {e}")
    except qc.queryClientError as e:
        raise Exception(f"Query Client Error: {e}")

def send_query(where, cols=util.trac_cols, tbl='ls_dr10.tractor AS trac', extras=''):
    query = f"SELECT {','.join(cols)} FROM {tbl} WHERE {where} {extras}"
    data = _sub_query(query)
    return data

def query_galaxy(ra,dec,radius=one_as,cols=all_cols,trac_table="ls_dr10.tractor",
                 phot_table="ls_dr10.photo_z", data_type=None,limit=1,
                 engine=None, save=None):
    """Queries the NOAO Legacy Survey Data Release 10 Tractor and Photo-Z tables

    Parameters:
        ra (float,int): Right ascension (RA) used in the q3c radial query to NOAO
        dec (float,int): Declination (DEC) used in the q3c radial query to NOAO
        radius (float,int): Radius in degrees used in the q3c radial query to NOAO.
            Defaults to 1 arcseconds/0.0002777 degrees.
        data_type (str): Data type to filter query results from NOAO. Should really
            only use types "PSF","REX","DEV","EXP","COMP", or "DUP". Defaults to
            "None" in order to return all results. You can bundle types to return
            data from multiple sources; follow PostGres formatting to do so.
        limit (int):Limit the number of rows returned from NOAO query. Defaults to 1.
    Returns:
        df_final: Pandas dataframe contained query results
    Raises:
        ValueError: If no objects are found
        ConnectionError: If NOAO times out. You can usually re-run the function and
            it'll work.
    """
    # Set up basic query
    query =[f"""SELECT {','.join(cols)} FROM {trac_table} AS trac """
    f"""INNER JOIN {phot_table} as phot_z ON trac.ls_id=phot_z.ls_id """,
    f"""WHERE (q3c_radial_query(ra,dec,{ra},{dec},{radius})) """,
    f"""ORDER BY q3c_dist({ra}, {dec}, trac.ra, trac.dec) ASC """,
    f"""LIMIT {limit}"""]
    # Add data type, if specified
    if data_type is not None:
        query.insert(1,f""" AND (type = '{data_type}') """)
    # Join full query
    query = ''.join(query)
    
    trac_data = _sub_query(query)
    if trac_data.empty:
        raise ValueError(f"No objects within {radius/one_as:.2f} arcsec of {ra},{dec}")
    
    # Save to DB
    if save is not None:
        if engine is None:
            engine = sa.create_engine(util.conn_string, poolclass=NullPool)
        
    
    return trac_data
    

def get_galaxy(ls_id, tag=None, engine=None):
    """ Gets a galaxy and its data from the psql database """
    # Make a database connection
    if engine is None:
        engine = sa.create_engine(util.conn_string, poolclass=NullPool)
    
    # Sleep first
    sleeptime = 5 + 15*np.random.rand()
    time.sleep(sleeptime)
    
    # Set up loop
    tries = 0
    while tries<15:
        tries+=1
        
        try: # Try to make a database connection
            conn = engine.connect()
            # Put in a try catch loop, sleep for a random time and try again if it doesn't work
            stmt = text(f"SELECT * FROM bookkeeping WHERE ls_id={ls_id}")
            bkdata = pd.DataFrame(conn.execute(stmt))

            if bkdata.empty:
                raise ValueError(f"No galaxy with LSID {ls_id} is present in the bookkeeping table")
            elif len(bkdata)>1:
                bkdata = bkdata[bkdata.tag==tag]
                if len(bkdata)!=1:
                    raise ValueError(f"Too many entries for LSID {ls_id}. Enter a tag.")

            gal_meta = bkdata.iloc[0]

            # if gal_meta['stage'] != 1:
            #     raise ValueError(f"Stage is wrong for galaxy {ls_id}. Current stage: {gal_meta['stage']}")
            stmt = text(f"SELECT * FROM {gal_meta['tbl_name']} WHERE id={gal_meta['tbl_id']}")
            tbldata = pd.DataFrame(
                conn.execute(stmt))

            conn.close()

            return bkdata, tbldata[util.query_cols]
            
        except OperationalError as e: # If too many connections, sleep and try again
            sleeptime = 5 + 5*np.random.rand() # Sleep 5-10 seconds
            print(f"Received Error {e}, sleeping {sleeptime} seconds and trying again.")
            time.sleep(sleeptime)
    raise ValueError("Could not connect to the database")

def merge_prospector(dr10_data, h5_file=None, redo=False):
    """ Collects prospector data on a galaxy and merges it with dr10 data """
    basic_data = dr10_data.iloc[0] # Series of values
    
    # Find 
    print(f'Object found: {basic_data.ls_id}')
    
    # magnitudes, uncertainties, and fluxes
    mags = [basic_data['dered_mag_'+b] for b in bands]
    mag_uncs = list(2.5 / np.log(10) / np.array([basic_data['snr_'+b] for b in bands]))
    # fluxes = [basic_data['dered_flux_'+b] for b in bands]
    
    # Print
    print(f'Redshift: {basic_data.z_phot_median}')
    
    # Data shortcuts
    # red_value = basic_data.z_phot_median
    # Run Prospector
    if h5_file is None: h5_file = f'{output_dir}{basic_data.ls_id}.h5'
    if redo or not os.path.exists(h5_file):
        outfile = h5_file.replace('.h5', '') if '.h5' in h5_file else h5_file
        h5_file = run_prospector(basic_data.ls_id, mags, mag_uncs, outfile=outfile)
    
    import pdb; pdb.set_trace()
    
    # Read prospector file
    h5_data = reader.results_from(h5_file)
    prosp_data = util.load_data(h5_data)
    
    dr10_data.loc[0, util.h5_cols[1:]] = prosp_data
    
    return dr10_data

def update_db(bkdata, gal_data, engine=None):
    """ Updates the database using the bookkeeping and galaxy data provided """
    # Make a database connection
    if engine is None:
        engine = sa.create_engine(util.conn_string, poolclass=NullPool)
    
    # Sleep first
    sleeptime = 5 + 15*np.random.rand()
    time.sleep(sleeptime)
    
    tries = 0
    while tries < 15:
        tries += 1
        try:
            conn = engine.connect()

            bkdata = bkdata.iloc[0]
            gal_data = gal_data.iloc[0]

            # Update galaxy table
            db_cols = [col.name for col in util.db_cols if col.name in list(gal_data.index)] # get usable columns of db
            print(db_cols)
            update_data = gal_data[db_cols].copy()
            
            # Assemble psql statement
            import pdb; pdb.set_trace()
            stmt = f"UPDATE {bkdata.tbl_name} SET "
            if 'type' in update_data: # string needs to be in quotes
                update_data['type'] = f"'{update_data['type']}'"
            stmt += ', '.join([f"{col} = {update_data[col]}" 
                               for col in db_cols])
            stmt += f" WHERE id = {bkdata.tbl_id}"
            
            if 'inf' in stmt:
                stmt = stmt.replace('inf', "'infinity'")
            if '-inf' in stmt:
                stmt = stmt.replace('-inf', "'-infinity'")
            
            conn.execute(text(stmt))

            # Update bookkeeping table
            stmt = f"UPDATE bookkeeping SET stage = 2 WHERE id = {str(bkdata.id)}"
            conn.execute(text(stmt))
            
            conn.commit()

            conn.close()
            return
            
        except OperationalError as e:
            sleeptime = 5 + 5*np.random.rand() # Sleep 5-10 seconds
            print(f"Received Error {e}, sleeping {sleeptime} seconds and trying again.")
            time.sleep(sleeptime)
    raise ValueError("Could not connect to the database")
    

def run_prospector(ls_id, mags, mag_uncs, prosp_file=prosp_file, redshift=None, outfile=None):
    """ Runs prospector with provided parameters """
    # Input and output filenames
    pfile = os.path.join(input_dir, prosp_file)
    if outfile is None:
    	outfile = os.path.join(output_dir, str(ls_id))
    if output_dir not in outfile:
        outfile = os.path.join(output_dir, outfile)
    
    # Run prospector with parameters
    # mags = ', '.join([str(x) for x in mags])
    # mag_uncs = ', '.join([str(x) for x in mag_uncs])
    
    if redshift is not None:
        cmd =  ['python', pfile, f'--object_redshift={redshift}', f'--mag_in={mags}', 
                f' --mag_unc_in={mag_uncs}', f'--outfile={outfile}']
        print("Running: ", ' '.join(cmd))
        subprocess.run(cmd, shell=False, check=True)
    else:
        cmd = ['python', pfile, f'--mag_in={mags}',
                f'--mag_unc_in={mag_uncs}', f'--outfile={outfile}']
        print("Running: ", cmd) 
        subprocess.run(cmd, shell=False, check=True) 
    
    return outfile +'.h5'

def predict(gal_data, model_file=None, thresh=0.05):
    if model_file is None:
        model_file = os.path.join(input_dir, "rf.model")
    
    with open(model_file, 'rb') as file:
        clf = pickle.load(file)
    
    pred_data = util.clean_and_calc(gal_data)[clf.feature_names_in_]
    
    pred = clf.predict_proba(pred_data)[:,0] < thresh
    
    return pred


### MAIN FUNCTION
if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--ls_id", type=int, help="LS ID of the galaxy")
    parser.add_argument("-r","--ra",type=float, help = "RA selection")
    parser.add_argument("-d","--dec",type=float, help="DEC selection")
    parser.add_argument("-rd", "--radius",type=float, default=0.0002777777778, help = "Radius for q3c radial query")
    parser.add_argument('-p', '--predict', action='store_true')
    parser.add_argument('-n', '--nodb', action='store_true')
    parser.add_argument("-s","--save",type=str,default=None, help="Database table to use")
    
    # Start sqlalchemy engine
    engine = sa.create_engine(util.conn_string, poolclass=NullPool)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.ls_id is not None:
        bkdata, tbldata = get_galaxy(args.ls_id, engine=engine)
    elif args.ra is not None and args.dec is not None:
        query_galaxy(args.ra, args.dec, save=args.save) # TODO: fix this
    else:
        raise ValueError("User must provide either a valid LSID or values for RA and DEC.")
    
    # Output file names
    h5_file = os.path.join(output_dir, f'{args.ls_id}.h5')
    # basic_file = os.path.join(output_dir, f'{args.ls_id}.csv')
    
    # Run/read prospector file and get full dataframe
    gal_data = merge_prospector(tbldata)
    
    # Test on RF model
    if args.predict:
        pred = predict(gal_data)
        gal_data['lensed'] = pred
    
    # Write output to database
    if not args.nodb:
        update_db(bkdata, gal_data, engine=engine)
    
    engine.dispose()
