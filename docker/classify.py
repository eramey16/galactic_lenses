# TODO: make it so you can give an lsid and have it search database
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
import dynesty

import prospect.io.read_results as reader
from prospect.io import write_results as writer
import param_monocle as param

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

output_dir = '/monocle/exports/' # Uncomment these two lines before pushing to docker
input_dir = '/monocle/'
# output_dir = '/global/cscratch1/sd/eramey16/gradient/' # Comment these two lines before pushing to docker
# input_dir='/global/cscratch1/sd/eramey16/gradient/'
default_pfile = os.path.join(input_dir, 'param_monocle.py')
one_as = 0.0002777777778 # degrees per arcsecond

#given an RA and DEC, pull magnitudes, magnitude uncertainties, redshifts from NOAO

import numpy
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
def nan_to_null(f,
        _NULL=psycopg2.extensions.AsIs('NULL'),
        _Float=psycopg2.extensions.Float):
    if not np.isnan(f):
        return _Float(f)
    return _NULL

register_adapter(float, nan_to_null)
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

def query_galaxy(ra=None,dec=None,ls_id=None,radius=one_as,cols=all_cols,trac_table="ls_dr10.tractor",
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
    
    # Assemble query
    query_base = [f"""SELECT {','.join(cols)} FROM {trac_table} AS trac """,
                  f"""INNER JOIN {phot_table} as phot_z ON trac.ls_id=phot_z.ls_id """]
    if ls_id is not None:
        if ls_id[0]=='9':
            raise ValueError("Emily fix this, it's a dr9 ls_id")
        query = query_base + [f"""WHERE trac.ls_id={ls_id}"""]
    elif ra is not None and dec is not None:
        query = query_base + [f"""WHERE (q3c_radial_query(ra,dec,{ra},{dec},{radius})) """,
                              f"""ORDER BY q3c_dist({ra}, {dec}, trac.ra, trac.dec) ASC """]
    else:
        raise ValueError("Must run query_galaxy with either ra and dec or ls_id")
    query += [f"""LIMIT {limit}"""]
    
    # # Set up basic query
    # query =[f"""SELECT {','.join(cols)} FROM {trac_table} AS trac """,
    # f"""INNER JOIN {phot_table} as phot_z ON trac.ls_id=phot_z.ls_id """,
    # f"""WHERE (q3c_radial_query(ra,dec,{ra},{dec},{radius})) """,
    # f"""ORDER BY q3c_dist({ra}, {dec}, trac.ra, trac.dec) ASC """,
    # f"""LIMIT {limit}"""]
    # Add data type, if specified
    if data_type is not None:
        query.insert(1,f""" AND (type = '{data_type}') """)
    # Join full query
    query = ''.join(query)
    
    trac_data = _sub_query(query)
    if trac_data.empty:
        if 
        raise ValueError(f"No objects within {radius/one_as:.2f} arcsec of {ra},{dec}")
    
    # Save to DB
    if save is not None:
        if engine is None:
            engine = sa.create_engine(util.conn_string, poolclass=NullPool)
        ### TODO: FIX THIS, what is going on here?
    
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
            bk_data = pd.DataFrame(conn.execute(stmt))

            if bk_data.empty:
                raise ValueError(f"No galaxy with LSID {ls_id} is present in the bookkeeping table")
            elif len(bk_data)>1:
                bk_data = bk_data[bk_data.tag==tag]
                if len(bk_data)!=1:
                    raise ValueError(f"Too many entries for LSID {ls_id}. Enter a tag.")

            gal_meta = bk_data.iloc[0]

            # if gal_meta['stage'] != 1:
            #     raise ValueError(f"Stage is wrong for galaxy {ls_id}. Current stage: {gal_meta['stage']}")
            stmt = text(f"SELECT * FROM {gal_meta['tbl_name']} WHERE id={gal_meta['tbl_id']}")
            tbl_data = pd.DataFrame(
                conn.execute(stmt))

            conn.close()

            return bk_data, tbl_data[util.query_cols]
            
        except OperationalError as e: # If too many connections, sleep and try again
            sleeptime = 5 + 5*np.random.rand() # Sleep 5-10 seconds
            print(f"Received Error {e}, sleeping {sleeptime} seconds and trying again.")
            time.sleep(sleeptime)
    raise ValueError("Could not connect to the database")

def calc_mag(basic_data, inflate_err=False):
    # magnitudes, uncertainties, and fluxes
    mags = [basic_data['dered_mag_'+b] for b in bands]
    mag_uncs = list(2.5 / np.log(10) / np.array([basic_data['snr_'+b] for b in bands]))
    if inflate_err>1:
        mag_uncs = [x*inflate_err for x in mag_uncs]
    # fluxes = [basic_data['dered_flux_'+b] for b in bands]
    return mags, mag_uncs

def prepare_prospector(tbl_data, basename=None, redo=False, nodes=0, 
                       resample_output=False, **kwargs):
    basic_data = tbl_data.iloc[0] # Series of values
    
    # Find 
    print(f'Object found: {basic_data.ls_id}')
    
    mags, mag_uncs = calc_mag(basic_data, kwargs['inflate_err'])
    
    # Print
    print(f'Redshift: {basic_data.z_phot_median}')
    
    redshift = basic_data.z_phot_median if kwargs['use_redshift'] else None
    
    # Data shortcuts
    # red_value = basic_data.z_phot_median
    # Run Prospector
    if basename is None: basename = f'{output_dir}{basic_data.ls_id}'
    if '.h5' in basename: basename = basename.replace('.h5', '')
    # Save pkl file if resampling
    outfile = basename + (".h5" if not resample_output else '.pkl')
    
    if os.path.exists(outfile) and not redo:
        print(f"File {outfile} exists, not re-running")
    else:
        run_prospector(basic_data.ls_id, mags, mag_uncs, nodes=nodes, resample_output=resample_output,
                                 redshift=redshift, basename=basename, **kwargs)
    
    return outfile

def merge_prospector(tbl_data, basename=None):
    """ Collects prospector data on a galaxy and merges it with dr10 data """
    h5_file = basename + '.h5'
    
    # Read prospector file
    h5_data = reader.results_from(h5_file)
    prosp_data = util.load_data(h5_data) # TODO: update this process
    
    tbl_data.loc[0, util.h5_cols[1:]] = prosp_data
    tbl_data = tbl_data.replace(np.nan, None) # DB doesn't like NaNs
    
    return tbl_data

def update_db(bk_data, gal_data, engine=None):
    """ Updates the database using the bookkeeping and galaxy data provided """
    # Make a database connection
    if engine is None:
        engine = sa.create_engine(util.conn_string, poolclass=NullPool)
    
    bktbl = sa.Table('bookkeeping', sa.MetaData(), autoload_with=engine)
    tbl = sa.Table(bk_data.tbl_name[0], sa.MetaData(), autoload_with=engine)
    
    # Sleep first
    sleeptime = 5 + 15*np.random.rand()
    time.sleep(sleeptime)
    
    tries = 0
    while tries < 15:
        tries += 1
        try:
            conn = engine.connect()

            bk_data = bk_data.iloc[0]
            gal_data = gal_data.iloc[0]

            # Update galaxy table
            db_cols = [col.name for col in util.db_cols if col.name in list(gal_data.index)] # get usable columns of db
            update_data = gal_data[db_cols].copy()
            
            # Assemble psql statement
            # import pdb; pdb.set_trace()
            stmt = sa.update(tbl).where(
                tbl.c.id==bk_data.tbl_id).values(**update_data)
            # stmt = f"UPDATE {bk_data.tbl_name} SET "
            # if 'type' in update_data: # string needs to be in quotes
            #     update_data['type'] = f"'{update_data['type']}'"
            # stmt += ', '.join([f"{col} = {update_data[col]}" 
            #                    for col in db_cols])
            # stmt += f" WHERE id = {bk_data.tbl_id}"
            
            # if 'inf' in stmt:
            #     stmt = stmt.replace('inf', "'infinity'")
            # if '-inf' in stmt:
            #     stmt = stmt.replace('-inf', "'-infinity'")
            
            conn.execute(stmt)

            # Update bookkeeping table
            stmt = f"UPDATE bookkeeping SET stage = 2 WHERE id = {str(bk_data.id)}"
            conn.execute(text(stmt))
            
            conn.commit()

            conn.close()
            return
            
        except OperationalError as e:
            sleeptime = 5 + 5*np.random.rand() # Sleep 5-10 seconds
            print(f"Received Error {e}, sleeping {sleeptime} seconds and trying again.")
            time.sleep(sleeptime)
    raise ValueError("Could not connect to the database")
    
def resample_prospector(pkl_file, resample_file, tbl_data, inflate_err=5, engine=None):
    print(f"Resampling file {pkl_file}")
    if not isinstance(resample_file, str):
        raise ValueError("Resample file must be a string with ls_ids to resample")
    if engine is None:
        engine = sa.create_engine(util.conn_string, poolclass=NullPool)
    
    # Open pickle file
    with open(pkl_file, 'rb') as file:
        dres = pickle.load(file)
        N = dres['samples'].shape[0]
    
    # Get mags and mag_uncs
    basic_data = tbl_data.iloc[0]
    # First run has inflated error bars # TODO: don't hardcode
    mags1, mag_uncs1 = calc_mag(basic_data, inflate_err=inflate_err)
    
    # Set up original model
    run_params = param.run_params
    args1 = {
        "mag_in": mags1,
        "mag_unc_in": mag_uncs1,
        "object_redshift": None, # TODO
    }
    obs1, model1, sps1, noise_model1 = param.load_all(args=args1, **run_params)
    spec_noise1, phot_noise1 = noise_model1
    
    def loglikelihood1(theta_vec):
        return param.lnprobfn(theta_vec, model=model1, obs=obs1, def_sps=sps1,
                              def_noise_model=noise_model1)
    
    # Read resampled galaxies
    resamp_gals = pd.read_csv(resample_file) # should be file with ls_id col
    # Loop through resampled galaxies
    for ls_id in resamp_gals['ls_id']:
        # Get database info
        bk_data2, tbl_data2 = get_galaxy(ls_id, engine=engine)
        mags2, mag_uncs2 = calc_mag(tbl_data2.iloc[0])
        
        # Set up new model
        args2 = {
            "mag_in": mags2,
            "mag_unc_in": mag_uncs2,
            "object_redshift": None, # TODO
        }
        print(f"Resampling with specs: {args2}")
        obs2, model2, sps2, noise_model2 = param.load_all(args=args2, **run_params)
        spec_noise2, phot_noise2 = noise_model2
        
        # Define log likelihood
        def loglikelihood2(theta_vec):
            return param.lnprobfn(theta_vec, model=model2, obs=obs2, def_sps=sps2, 
                                  def_noise_model=noise_model2)
        
        # Resample prospector output
        logl2 = [loglikelihood2(dres['samples'][i]) for i in range(N)]
        # TODO: Try with JAX on a GPU
        # Reweight samples
        dres_rwt = dynesty.utils.reweight_run(dres, logp_new=logl2)
        
        outfile = output_dir+f"{ls_id}_rwt.h5"
        writer.write_hdf5(outfile, {}, model2, obs2, dres_rwt, 
                          None, sps=sps2, tsample=None, toptimize=0.0)
        print(f"Wrote file: {outfile}")
    
    print("Remove pickle file here.")

def run_prospector(ls_id, mags, mag_uncs, prosp_file=default_pfile, redshift=None, 
                   nodes=0, basename=None, effective_samples=1000, resample_output=False,
                   **kwargs):
    """ Runs prospector with provided parameters """
    # Input and output filenames
    if prosp_file is None: prosp_file = default_pfile
    if basename is None:
        basename = os.path.join(output_dir, str(ls_id))
    if output_dir not in basename:
        basename = os.path.join(output_dir, basename)
    
    # Run prospector with parameters
    # mags = ', '.join([str(x) for x in mags])
    # mag_uncs = ', '.join([str(x) for x in mag_uncs])
    
    cmd = ['python', prosp_file, f'--effective_samples={effective_samples}',
           f'--mag_in="{mags}"', f'--mag_unc_in="{mag_uncs}"',
           f'--outfile={basename}']
    
    if redshift is not None:
        cmd.insert(2, f'--object_redshift={redshift}')
    if resample_output:
        cmd.insert(2, f'--output_dynesty')
    if nodes!=0:
        cmd.insert(0, f'mpirun -n {nodes}')
    print("Running: ", ' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True, env=os.environ)
    
    outfile = basename + ('.h5' if not resample_output else '.pkl')
    if not os.path.exists(outfile):
        raise ValueError("Prospector run failed.")

def predict(gal_data, model_file=None, thresh=0.05):
    if model_file is None:
        model_file = os.path.join(input_dir, "rf.model")
    
    with open(model_file, 'rb') as file:
        clf = pickle.load(file)
    
    pred_data = util.clean_and_calc(gal_data)[clf.feature_names_in_]
    
    pred = clf.predict_proba(pred_data)[:,0] < thresh
    
    return pred

def run(ls_id=None, ra=None, dec=None, radius=None, nodes=0, predict=False, 
        nodb=False, save=None, basename=None, resample_output=False, **kwargs):
    
    # Start sqlalchemy engine
    engine = sa.create_engine(util.conn_string, poolclass=NullPool)
    
    start = time.time()
    if ls_id is not None:
        bk_data, tbl_data = get_galaxy(ls_id, engine=engine)
    elif ra is not None and dec is not None:
        raise NotImplementedError("Emily needs to implement this!")
        query_galaxy(ra, dec, save=save) # TODO: fix this
    else:
        raise ValueError("User must provide either a valid LSID or values for RA and DEC.")
    
    # Output file names
    if basename is not None:
        basename = os.path.join(output_dir, basename)
    else:
        basename = os.path.join(output_dir, str(ls_id))
    
    # Set settings for resampling, if chosen
    if resample_output:
        predict = False
        nodb = True
        kwargs['inflate_err'] = 5
        kwargs['effective_samples'] = 500000
    
    # Run/read prospector file and get full dataframe
    outfile = prepare_prospector(tbl_data, nodes=nodes, basename=basename, 
                       resample_output=resample_output, **kwargs)
    if resample_output:
        print(f"Resampling with galaxies from {resample_output}")
        resample_prospector(pkl_file=outfile, resample_file=resample_output, 
                            tbl_data=tbl_data, engine=engine)
        return
    else:
        gal_data = merge_prospector(tbl_data, basename=basename)
    
    # Test on RF model
    if predict:
        pred = predict(gal_data)
        gal_data['lensed'] = pred
    
    # Write output to database
    if not nodb:
        update_db(bk_data, gal_data, engine=engine)
    
    engine.dispose()
    print(f"\nFinished {gal_data['ls_id'][0]} in {(time.time()-start)/60} min\n")


### MAIN FUNCTION
if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--ls_id", type=int, help="LS ID of the galaxy")
    parser.add_argument("-r","--ra",type=float, help = "RA selection")
    parser.add_argument("-d","--dec",type=float, help="DEC selection")
    parser.add_argument("-rd", "--radius",type=float, default=0.0002777777778, help = "Radius for q3c radial query")
    parser.add_argument('-n', '--nodes', type=int, default=0,
                        help='Number of nodes for MPI run (0 means no MPI)')
    parser.add_argument('--resample_output', type=str, default=None,
                        help='Provide a file name with LS IDs of galaxies to resample from original. \
                        Also applies --inflate_err=5, --effective_samples=500000, --nodb=True')
    parser.add_argument('-e', '--effective_samples', type=int, default=100000, 
                        help='Same as --nested_target_n_effective in Prospector run')
    parser.add_argument('--inflate_err', type=int, default=1, 
                        help="Factor to inflate errors for prospector run")
    parser.add_argument('--use_redshift', action='store_true', 
                        help="If present, prospector will use redshift in its calculations")
    parser.add_argument('--prosp_file', type=str, default=None, help='File to use for running prospector')
    parser.add_argument('-p', '--predict', action='store_true')
    parser.add_argument('--nodb', action='store_true')
    parser.add_argument("-s","--save",type=str,default=None, help="Database table to use")
    parser.add_argument("--basename", type=str, default=None, help="Output filename (no .h5)")
    
    # Parse arguments
    args = parser.parse_args()
    # print(args)
    run(**vars(args))
