### db_util.py - uses sqlalchemy to interact with the database
### Author - Emily Everetts
### Date Created - 11/4/24

import pandas as pd
import numpy as np
from enum import IntEnum

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import OperationalError

import logging

conn_string = 'postgresql+psycopg2://lensed_db_admin@nerscdb03.nersc.gov/lensed_db'

class Status(IntEnum):
    INIT = 0
    TRAC_DONE = 1
    PROCESSED = 2

bands = ['g', 'r', 'i', 'z', 'w1', 'w2']
colors = ['g_r', 'i_z', 'r_i', 'r_z', 'w1_w2', 'z_w1']

# cols we pull from tractor catalog
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
phot_cols = ['z_phot_median_i AS z_phot_median', # cols we pull from phot catalog
               'z_phot_std_i AS z_phot_std', 'z_spec']

dchisq_labels = [f'dchisq_{i}' for i in range(1,6)]
rchisq_labels = ['rchisq_g', 'rchisq_r', 'rchisq_z', 'rchisq_w1', 'rchisq_w2']

desi_cols = ['trac.'+col for col in trac_cols] + ['phot_z.'+col for col in phot_cols]

def desi_to_db(data):
    
    data = data.copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Calculate colors
    for c in colors:
        c_ = c.split('_')
        data[c] = data['dered_mag_'+c_[0]]-data['dered_mag_'+c_[1]]

    data.loc[data.z_phot_median<0, 'z_phot_median'] = np.nan
    data.loc[data.z_phot_std<0, 'z_phot_std'] = np.nan
    data.loc[data.z_spec<0, 'z_spec'] = np.nan

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
    
    return data

def insert_trac(gal_data, tbl_name, bk_tbl='bookkeeping', train=False,
                    meta=sa.MetaData(), engine=None, logger=logging.getLogger('main')):
    """
    Inserts a pandas dataframe of galaxies into the database.
    """
    if engine is None:
        engine = sa.create_engine(conn_string, poolclass=NullPool)
    
    with engine.connect() as conn:
        # Init tables
        if engine.dialect.has_table(conn, tbl_name):
            bk_tbl = sa.Table(bk_tbl, meta, autoload_with=engine)
            gal_tbl = sa.Table(tbl_name, meta, autoload_with=engine)
        else: raise ValueError(f"No table with name {tbl_name} in database") # TODO Em fix this
        
        # Match db and data columns # TODO - maybe a better way?s
        db_cols = [col.name for col in gal_tbl.columns.values() 
                       if col.name in gal_data.columns]
        tags = gal_data['tag']
        gal_data = gal_data[db_cols]
        
        # Insert each galaxy
        for i,row in gal_data.iterrows():
            # Insert into galaxy table
            logger.info(f"Inserting galaxy {row.ls_id} into data table")
            stmt = gal_tbl.insert().values(**row)
            result = conn.execute(stmt)
            pkey = result.inserted_primary_key[0]

            # Insert into bookkeeping
            logger.info(f"Inserting galaxy {row.ls_id} into bookkeeping table")
            stmt = bk_tbl.insert().values(tbl_id=pkey, 
                                  tbl_name=tbl_name,
                                  ls_id=row.ls_id,
                                  stage=Status.TRAC_DONE,
                                  train=train,
                                  tag=tags[i]
                                )
            conn.execute(stmt)
            conn.commit()