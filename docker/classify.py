### Classify.py - this is the reboot - takes a galaxy's info, fetches it, runs prospector, classifies it
### Author - Emily Everetts
### Date Created - 11/4/24
import os
import sys
import logging
import numpy as np
import pandas as pd
import importlib
import subprocess
import argparse

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import OperationalError

import requests
from dl import queryClient as qc, helpers
from dl.helpers.utils import convert

from prospect import fitting
from prospect.io import write_results as writer, read_results as reader
import dynesty
from dynesty.dynamicsampler import stopping_function, weight_function
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt

import db_util as util
import prospect_conversion as conv

from psycopg2.extensions import register_adapter, AsIs
register_adapter(np.int64, AsIs)

BKTBL = 'bookkeeping'

default_param = 'param_monocle' # TODO: add this to the python path

class Classifier:
    
    one_as = 0.0002777777778 # degrees per arcsecond
    
    def __init__(self, output_dir=os.getcwd(), gal_df=None, trac_table="ls_dr10.tractor",
                 phot_table="ls_dr10.photo_z", query_cols=util.desi_cols, engine=None,
                 prosp_package=default_param, code_dir='/monocle', 
                 logger=logging.getLogger(__name__)):
        # Set up output dir and data structures
        # gal_df must have either ls_id or ra, dec as columns
        # if no tbl_name is given, no db will be used
        self.outdir = output_dir
        
        self.trac_table = trac_table
        self.phot_table = phot_table
        self.query_cols = query_cols
        
        if isinstance(logger, int):
            logging.basicConfig(level=logger)
            log = logging.getLogger(__name__)
            log.setLevel(logger)
            self.logger = log
        else: self.logger = logger
        
        if gal_df is not None:
            self.gal_data = gal_df
        else:
            self.gal_data = pd.DataFrame() # column names?
        
        if engine is None:
            self.engine = sa.create_engine(util.conn_string, poolclass=NullPool, future=True)
        self.meta = sa.MetaData()
        
        if '.py' in prosp_package: raise ValueError("Delete .py from package name")
        try:
            self.param = importlib.import_module(prosp_package)
            self.logger.debug(f"Loading parameters from file {self.param.__file__}")
        except Exception as e:
            raise ValueError(f"Can't import prospector module {prosp_package}. Error {e}")
    
    def parallel_setup(self):
        # Set up mpi
        try:
            import mpi4py
            from mpi4py import MPI
            from schwimmbad import MPIPool
            self.MPIPool = MPIPool

            mpi4py.rc.threads = False
            mpi4py.rc.recv_mprobe = False

            self.comm = MPI.COMM_WORLD
            size = self.comm.Get_size()
            
            self.rank = self.comm.Get_rank()

            self.withmpi = size > 1
            start = MPI.Wtime()
        except ImportError as e:
            print('Failed to start MPI; are mpi4py and schwimmbad installed? Proceeding without MPI.')
            print(f'Message: {e}')
            self.withmpi = False
            self.rank = None
            self.comm = None
            start = time.time()
        return self.withmpi
    
    def get_galaxy(self, ls_id, tag=None, loop=5):
        bk_tbl = sa.Table(BKTBL, self.meta, autoload_with=self.engine)
        
        if loop is None: loop=1
        for _ in range(loop):
            try:
                with self.engine.connect() as conn:
                    # Get entry in bookkeeping table and narrow down
                    stmt = sa.select(bk_tbl).where(bk_tbl.c.ls_id==ls_id)
                    bk_data = pd.DataFrame(conn.execute(stmt))
                    if tag is not None and not bk_data.empty:
                        bk_data = bk_data.loc[bk_data.tag==tag]
                    if bk_data.empty: return None
                    elif len(bk_data)>1:
                        raise ValueError(f"Multiple definitions of ls_id {ls_id} for tag {tag}.")

                    # Check if right table exists
                    tbl_name = bk_data.tbl_name[0]
                    tbl_id = bk_data.tbl_id[0]
                    if self.engine.dialect.has_table(conn, tbl_name):
                        gal_tbl = sa.Table(tbl_name, self.meta, autoload_with=self.engine)
                        stmt = sa.select(gal_tbl).where(gal_tbl.c.id==tbl_id)
                        gal_data = pd.DataFrame(conn.execute(stmt))
                        if len(gal_data)==0: raise ValueError(f"No entry {tbl_id} in table {tbl_name}")
                    else:
                        raise ValueError(f"{ls_id} is defined in the bookkeeping table, " 
                                         f"but there is no table named {tbl_name} in the database.")

                return bk_data, gal_data
        
            except OperationalError as e: # If too many connections, sleep and try again
                sleeptime = 5 + 5*np.random.rand() # Sleep 5-10 seconds
                self.logger.warning(f"Received Error {e}, sleeping {sleeptime} seconds and trying again.")
                time.sleep(sleeptime)
        
        # Failed all tries
        raise ValueError("Could not connect to the database")
    
    def _sub_query(self, query):
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
    
    def send_query(self, where, cols=util.trac_cols, tbl='ls_dr10.tractor AS trac', extras=''):
        query = f"SELECT {','.join(cols)} FROM {tbl} WHERE {where} {extras}"
        data = self._sub_query(query)
        return data
    
    def query_galaxy(self, ra=None, dec=None, ls_id=None, radius=one_as,
                     data_type=None, limit=1):
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

        ### Assemble query
        query_base = [f"""SELECT {','.join(self.query_cols)} FROM {self.trac_table} AS trac """,
                      f"""INNER JOIN {self.phot_table} as phot_z ON trac.ls_id=phot_z.ls_id """]
        if ls_id is not None: # Query by ls_id
            if str(ls_id)[0]=='9': # check for dr9
                raise ValueError("Emily fix this, it's a dr9 ls_id")
            query = query_base + [f"""WHERE trac.ls_id={ls_id}"""]
        elif ra is not None and dec is not None: # Query by RA and DEC
            query = query_base + [f"""WHERE (q3c_radial_query(ra,dec,{ra},{dec},{radius})) """,
                                  f"""ORDER BY q3c_dist({ra}, {dec}, trac.ra, trac.dec) ASC """]
        else: # Something went wrong
            raise ValueError("Must run query_galaxy with either ra and dec or ls_id")
        query += [f"""LIMIT {limit}"""]

        if data_type is not None:
            query.insert(1, f""" AND (type = '{data_type}') """)
        # Join full query
        query = ''.join(query)

        trac_data = self._sub_query(query)
        if trac_data.empty:
            raise ValueError(f"No objects were found matching the query")
        db_data = util.desi_to_db(trac_data)

        return db_data
    
    def update_db(self, bk_data, gal_data, outfile, use_redshift=False, **kwargs):
        """ Now thread-safe """
        # Get model
        run_params = self.param.run_params
        res, obs, _ = reader.results_from(outfile)
        model = self.param.load_model(obs=obs, **run_params)
        if not self.withmpi or self.rank==0:
            util.update_prosp(bk_data, gal_data, (res, obs, model), 
                              meta=self.meta, engine=self.engine)
            self.logger.info(f"Wrote file data to table {bk_data.tbl_name[0]}")
    
    def _calc_mag(self, gal_data):
        # magnitudes, uncertainties, and fluxes
        mags = [gal_data['dered_mag_'+b] for b in util.bands]
        mag_uncs = list(2.5 / np.log(10) / np.array([gal_data['snr_'+b] for b in util.bands]))
        return mags, mag_uncs
    
    def run_prospector(self, gal_data, outfile, effective_samples=None,
                       use_redshift=False, verbose=False, **kwargs):
        """gal_data should be a pandas series"""
        
        mags, mag_uncs = self._calc_mag(gal_data)
        redshift = gal_data.z_phot_median if use_redshift else None
        
        run_params = self.param.run_params
        obs, model, sps, noise_model = self.param.load_all(mags, mag_uncs, redshift=redshift, **run_params)
        spec_noise, phot_noise = noise_model
        self.log_mpi("Loaded obs, model, SPS, and noise model", logging.INFO)

        initial_theta_grid = np.around(np.arange(model.config_dict["logzsol"]['prior'].range[0],
                                                 model.config_dict["logzsol"]['prior'].range[1],
                                                 step=0.01), decimals=2)

        for theta_init in initial_theta_grid:
            sps.ssp.params["sfh"] = model.params['sfh'][0]
            sps.ssp.params["imf_type"] = model.params['imf_type'][0]
            sps.ssp.params["logzsol"] = theta_init
            sps.ssp._compute_csp()
        
        print(effective_samples, use_redshift)
        if effective_samples is not None:
            run_params['nested_stop_kwargs'] = {"target_n_effective": effective_samples}
        else:
            run_params['nested_stop_kwargs'] = {"post_thresh": 0.05}
        
        run_params['verbose'] = verbose
        
        # Probability and prior functions
        global new_lnfn
        global new_prior
        def new_lnfn(x):
            return self.param.lnprobfn(x, model=model, obs=obs,
                                       def_sps=sps, def_noise_model=noise_model)
        def new_prior(u):
            return self.param.prior_transform(u, model=model)
        
        run_params.update(dict(nlive_init=400, nested_method="rwalk", nested_dlogz_init=0.05))
        
        # Return stuff for MPI
        if self.withmpi:
            run_params["using_mpi"] = True
            # return run_params, obs, model, sps, noise_model, new_lnfn, new_prior
            
            with self.MPIPool() as pool:
                # The dependent processes will run up to this point in the code
                if not pool.is_master():
                    pool.wait()
                    sys.exit(0)
                nprocs = pool.size
                self.log_mpi(f"Starting prospector in parallel with {nprocs} threads."+
                             f" Params {run_params}", logging.DEBUG)
                # The parent process will oversee the fitting
                output = fitting.run_dynesty_sampler(new_lnfn, new_prior, model.ndim,
                                                     pool=pool, queue_size=nprocs, 
                                                     stop_function=stopping_function,
                                                     wt_function=weight_function,
                                                     **run_params)
                # Write results to file
                writer.write_hdf5(outfile, {}, model, obs,
                                 output, None,
                                 sps=sps,
                                 tsample=None,
                                 toptimize=0.0)
                self.logger.info(f"\nWrote results to: {outfile}\n")
        else: # Run without MPI
            self.logger.debug(f"Starting prospector in series with params: {run_params}")

            output = fitting.run_dynesty_sampler(new_lnfn, new_prior, model.ndim,
                                                 stop_function=stopping_function,
                                                 wt_function=weight_function,
                                                 **run_params)

            # Write results to file
            writer.write_hdf5(outfile, {}, model, obs,
                             output, None,
                             sps=sps,
                             tsample=None,
                             toptimize=0.0)
            self.log_mpi(f"\nWrote results to: {outfile}\n", logging.INFO)
        
        if not os.path.exists(outfile):
            raise RuntimeError(f"Prospector run failed. Can't find {outfile}")
    
    def check_db(self, ls_id=None, ra=None, dec=None, tag=None):
        if ls_id is None and ra is None and dec is None:
            raise ValueError("Must input either ls_id or ra and dec pair")
        elif ls_id is not None:
            in_db = self.get_galaxy(ls_id, tag=tag)
            if in_db is not None: return True
        return False
    
    def ready_for_prospector(self, bk_data, gal_data, redo=False):
        # Check if the stage is right
        if bk_data.stage[0] < util.Status.TRAC_DONE:
            raise NotImplementedError("Trac data isn't in the database. Not sure how this got here. "
                                      "Remove or update existing DB entry.")
        elif bk_data.stage[0] == util.Status.PROCESSED and not redo:
            self.log_mpi(f"Galaxy {bk_data.ls_id[0]} already processed. Not re-running", 
                         logging.WARNING)
            return False
        return True
    
    def ensure_db(self, ls_id=None, ra=None, dec=None, tag=None, train=False, 
                  tbl_name=None):
        """ Thread-safe (I hope) """
        if not self.withmpi or self.rank==0:
            if not self.check_db(ls_id, ra, dec, tag):
                # Need a table name for new entry
                if tbl_name is None:
                    raise ValueError("Must supply a table name for inserting galaxies into the database")

                # Query galaxy from NOIRLab
                q_data = self.query_galaxy(ra, dec, ls_id)
                if q_data is None or q_data.empty:
                    raise ValueError(f"No data found for galaxy\n ls_id: {ls_id}\n RA {ra}, DEC {dec}")
                # Set ls_id for getting database entry
                ls_id = q_data.ls_id[0]

                if not self.check_db(ls_id, tag=tag): # Save to DB
                    # Calculate database quantities
                    q_data['tag'] = tag
                    util.insert_trac(q_data, tbl_name, engine=self.engine, meta=self.meta,
                                         train=train, bk_tbl=BKTBL, logger=self.logger)
            # Galaxy should be in db now
            result = self.get_galaxy(ls_id, tag=tag)
        else:
            result = None
        # Broadcast to other threads
        if self.withmpi:
            result = self.comm.bcast(result, root=0)
        
        # Return results of get_galaxy
        return result
    
    def run(self, ls_id=None, ra=None, dec=None, tag=None, outfile=None, 
                tbl_name=None, nodb=False, redo=False, train=False, **kwargs):
        """ Runs a galaxy through prospector regardless of what stage of processing it's in """
        self.parallel_setup()
        
        # Check if galaxy is already in database
        if nodb: raise NotImplementedError("Need to figure out how to deal with this")
        
        bk_data, gal_data = self.ensure_db(ls_id, ra, dec, tag=tag, train=train, tbl_name=tbl_name)
        if not self.ready_for_prospector(bk_data, gal_data, redo=redo): return
        
        # Get output filename, if not provided
        if outfile is None: outfile = f'{self.outdir}{gal_data.ls_id[0]}.h5'
        
        # If we already have the h5 file, don't re-run
        if os.path.exists(outfile) and not redo:
            self.logger.warning(f"File {outfile} exists, not re-running")
        else: # Run prospector
            # TODO: add in other kwargs when you get them working
            self.run_prospector(gal_data.iloc[0], outfile=outfile, **kwargs)
        
        # Update database with prospector predictions
        self.logger.info(f"Updating database with file contents: {outfile}")
        self.update_db(bk_data, gal_data, outfile=outfile, **kwargs)

    def log_mpi(self, message, level):
        if self.rank==0:
            self.logger.log(level, message)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--ls_id", type=int, default=None, help="LS ID of the galaxy")
    parser.add_argument("-r","--ra",type=float, default=None, help = "RA selection")
    parser.add_argument("-d","--dec",type=float, default=None, help="DEC selection")
    parser.add_argument("-t", "--tag", type=str, default=None, help="Galaxy tag in database")
    parser.add_argument('--nodb', action='store_true')
    parser.add_argument("-tn","--tbl_name",type=str,default=None, help="Database table to save to")
    parser.add_argument("--train", action='store_true')
    parser.add_argument("-o", "--outfile", type=str, default=None, help="output filename for prospector")
    parser.add_argument("--redo", action='store_true')
    parser.add_argument("--use_redshift", action='store_true')
    parser.add_argument("--effective_samples", type=int, default=None, help="# of prospector iterations")
    parser.add_argument("--output_dir", type=str, default='/monocle/exports/', 
                        help="Directory to store output files")
    parser.add_argument("--log_level", type=int, default=logging.INFO, help="Level for the logging object")
    parser.add_argument("-v", "--verbose", type=bool, default=True, help="Verbosity of dynesty")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize object
    classy = Classifier(logger=args.log_level, output_dir=args.output_dir)
    # TODO: find a way to pass the other ones as kwargs
    classy.run(**vars(args))
        
        
        
