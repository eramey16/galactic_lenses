import os
import pandas as pd
import sqlalchemy
from sqlalchemy.pool import NullPool
from docker import classify
from docker import util

engine = sqlalchemy.create_engine(util.conn_string, poolclass=NullPool)
ls_ids = list(pd.read_csv('remaining_lsids.dat', header=None)[0].values)
print(len(ls_ids))

data_dir = os.path.expandvars('$SCRATCH/data/monocle/cosmos_160/')

for ls_id in ls_ids:
    classify._fix_db(ls_id=ls_id, data_dir=data_dir, 
                     model_file='docker/rf.model', engine=engine)