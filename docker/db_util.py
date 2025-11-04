### db_util.py - uses sqlalchemy to interact with the database
### Author - Emily Everetts
### Date Created - 11/4/24

import pandas as pd
import numpy as np
import numpy.ma as ma
from enum import IntEnum

import sqlalchemy as sa
from sqlalchemy import Column, Table, text
from sqlalchemy import String, DateTime
from sqlalchemy.types import BIGINT, FLOAT, REAL, VARCHAR, BOOLEAN, INT
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import OperationalError

import logging

from prospect.io import read_results as reader
import prospect_conversion as conv

conn_string = 'postgresql+psycopg2://lensed_db_admin@nerscdb03.nersc.gov/lensed_db'

from psycopg2.extensions import register_adapter, AsIs
register_adapter(np.int64, AsIs)

class Status(IntEnum):
    INIT = 0
    TRAC_DONE = 1
    PROCESSED = 2

bands = ['g', 'r', 'i', 'z', 'w1', 'w2']
colors = ['g_r', 'i_z', 'r_i', 'r_z', 'w1_w2', 'z_w1']

# More tractor labels
dchisq_labels = [f'dchisq_{i}' for i in range(1,6)]
rchisq_labels = ['rchisq_'+b for b in bands]

# cols we pull from tractor catalog
trac_cols = ['ls_id', 'ra', 'dec', 'type'] \
            + ['dered_mag_'+b for b in bands] \
            + ['dered_flux_'+b for b in bands] \
            + colors \
            + ['snr_'+b for b in bands] \
            + ['flux_ivar_'+b for b in bands] \
            + dchisq_labels \
            + rchisq_labels \
            + ['sersic', 'sersic_ivar'] \
            + ['psfsize_g', 'psfsize_r', 'psfsize_z'] \
            + ['shape_r', 'shape_e1', 'shape_e2'] \
            + ['shape_r_ivar', 'shape_e1_ivar', 'shape_e2_ivar']
phot_cols = ['z_phot_median_i AS z_phot_median', # cols we pull from phot catalog
               'z_phot_std_i AS z_phot_std', 'z_spec']

# Add trac. and phot_z.
desi_cols = ['trac.'+col for col in trac_cols] + ['phot_z.'+col for col in phot_cols]
filter_cols = ['dered_mag_'+b for b in bands]+['ls_id', 'ra', 'dec', 'type'] + \
                ['rchisq_'+b for b in bands]

# # Column labels for h5 data # TODO: move to db func
h5_cols = [f'maggies_{b}' for b in bands] + \
        [f'maggies_unc_{b}' for b in bands] + \
        [f'maggies_fit_{b}' for b in bands] + \
        [f'{param}' for param in conv.prosp_params] + \
        [f'{param}_sig_minus' for param in conv.prosp_params] + \
        [f'{param}_sig_plus' for param in conv.prosp_params]

# Get database columns from file columns
db_cols = [
    Column('ls_id', BIGINT, nullable=False),
    Column('ra', FLOAT),
    Column('dec', FLOAT),
    Column('type', VARCHAR(4)),
    *[Column(col, FLOAT) for col in trac_cols[4:] +
      ['z_phot_median', 'z_phot_std', 'z_spec'] + h5_cols],
    # All other cols
    Column('lens_grade', VARCHAR(1)),
    Column('lensed', BOOLEAN),
    Column('id', BIGINT, primary_key=True, autoincrement=True),
]

def make_table(tbl_name, engine=None, meta=sa.MetaData(), bk_tbl='bookkeeping'):
    if engine is None: engine = sa.create_engine(conn_string, poolclass=NullPool)
    with engine.connect() as conn:
        # Table columns
        tbl = Table(
            tbl_name,
            meta,
            *db_cols
        )
        
        meta.create_all(conn, checkfirst=True)
        conn.commit()

def remove_gal(ls_id, tag, engine=None, meta=sa.MetaData(), logger=logging.getLogger('main')):
    if engine is None: engine = sa.create_engine(conn_string, poolclass=NullPool)
    
    with engine.connect() as conn:
        # Init tables
        bk_tbl = sa.Table('bookkeeping', meta, autoload_with=engine)
        stmt = sa.select(bk_tbl).where(bk_tbl.c.ls_id==ls_id)
        bk_data = pd.DataFrame(conn.execute(stmt))
        
        if tag is not None and not bk_data.empty:
            bk_data = bk_data.loc[bk_data.tag==tag]
        if bk_data.empty:
            logger.warning(f"No galaxy with ls_id {ls_id} and tag '{tag}' in database")
            return
        elif len(bk_data) > 1:
            raise ValueError(f"Too many defined entries of galaxy {ls_id} for tag '{tag}'")
        
        tbl_name = bk_data.tbl_name[0]
        if not engine.dialect.has_table(conn, tbl_name):
            raise ValueError(f"No table '{tbl_name}' in the database.")
        gal_tbl = sa.Table(tbl_name, meta, autoload_with=engine)
        
        # Delete bookkeeping entry
        stmt = sa.delete(bk_tbl).where(bk_tbl.c.id==bk_data.id[0])
        conn.execute(stmt)
        
        # Delete galaxy table entry
        stmt = sa.delete(gal_tbl).where(gal_tbl.c.id==bk_data.tbl_id[0])
        conn.execute(stmt)
        
        conn.commit()

def clean_desi(data, duplicates=False, dropna=True, filter_cols=filter_cols):
    """ Cleans data from desi query """
    data = data[data.type!='DUP']
    data = data[data.type!='PSF']

    data.loc[data.z_phot_median<0, 'z_phot_median'] = None
    data.loc[data.z_phot_std<0, 'z_phot_std'] = None
    data.loc[data.z_spec<0, 'z_spec'] = None

    # Remove bad / duplicate entries
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    if dropna:
        data.dropna(subset=filter_cols, inplace=True)
    if not duplicates:
        data.drop_duplicates(subset=['ls_id'], inplace=True)

    return data
    
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
    if engine is None: engine = sa.create_engine(conn_string, poolclass=NullPool)
    
    with engine.connect() as conn:
        # Init tables
        if not engine.dialect.has_table(conn, tbl_name):
            logger.warning(f"No table {tbl_name} in the database. Creating table {tbl_name}.")
            make_table(tbl_name, engine=engine, meta=meta)
        bk_tbl = sa.Table(bk_tbl, meta, autoload_with=engine)
        gal_tbl = sa.Table(tbl_name, meta, autoload_with=engine)
        
        # Match db and data columns # TODO - maybe a better way?
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

def update_prosp(bk_data, gal_data, gal_results, bk_tbl='bookkeeping', meta=sa.MetaData(), 
                 engine=None, logger=logging.getLogger("main")):
    
    # Separate data structures
    res, obs, model = gal_results
    bf = res['bestfit']
    
    # import pdb; pdb.set_trace()
    # Get model
    quantiles = conv.quantiles_phot(res, model)
    
    # Read observed and fitted photometry into the table
    gal_data.loc[0, [f'maggies_{b}' for b in bands]] = obs['maggies']
    gal_data.loc[0, [f'maggies_unc_{b}' for b in bands]] = obs['maggies_unc']
    gal_data.loc[0, [f'maggies_fit_{b}' for b in bands]] = bf['photometry']
    
    # Read predicted values into the table
    for col in quantiles.columns:
        gal_data.loc[0, f'{col}_sig_minus'] = quantiles[col][1] - quantiles[col][0]
        gal_data.loc[0, f'{col}'] = quantiles[col][1]
        gal_data.loc[0, f'{col}_sig_plus'] = quantiles[col][2] - quantiles[col][1]
    
    # Get database tables
    if engine is None: engine = sa.create_engine(conn_string, poolclass=NullPool)
    with engine.connect() as conn:
        bk_tbl = sa.Table(bk_tbl, meta, autoload_with=engine)
        if not engine.dialect.has_table(conn, bk_data.tbl_name[0]):
            raise ValueError(f"No table {bk_data.tbl_name[0]} in the database")
        gal_tbl = sa.Table(bk_data.tbl_name[0], meta, autoload_with=engine)
        
        for col in conv.prosp_params:
            if col not in [col.name for col in gal_tbl.columns.values()]:
                raise ValueError(f"The columns of table {bk_data.tbl_name[0]} "
                                 "do not match the output prospector parameters "
                                 "(likely a code version issue).")

        # Update gal table
        stmt = gal_tbl.update().where(gal_tbl.c.id==bk_data.tbl_id[0]).values(**gal_data.iloc[0])
        conn.execute(stmt)
        
        # Update bookkeeping table
        stmt = bk_tbl.update().where(bk_tbl.c.id==bk_data.id[0]).values(stage=Status.PROCESSED)
        conn.execute(stmt)
        conn.commit()