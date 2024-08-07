### prep_run.py: prepares a batch prospector run by generating a tasks.txt file in the specified directory and with the specified galactic parameters
### Author: Emily Ramey
### Date: 4/11/22

### Imports
import pandas as pd
import numpy as np
import argparse
import os
import sys
import sqlalchemy

conn_string = 'postgresql+psycopg2://lensed_db_admin@nerscdb03.nersc.gov/lensed_db'

### String to repeat in taskfarmer file
shft_cmd = "shifter --image=eramey16/monocle:latest --volume='{}:/monocle/exports' /opt/conda/bin/python /monocle/classify.py --ls_id={}"
task_filename = ["tasks", ".txt"]
coord_filename = ["coords", ".csv"]

def process_tsv(file, dest):
    """ Processes a tsv file from the masterlens database """
    # Read file from tsv
    galaxies = pd.read_csv(file, skiprows=1, sep='\t')
    # Format columns correctly
    galaxies.columns = galaxies.columns.str.replace('"', '')
    galaxies.columns = galaxies.columns.str.strip()
    
    # Create new csv with just ra and dec
    radec = galaxies[['ra_coord', 'dec_coord']]
    radec.columns = ['ra', 'dec']
    
    # Return dataframe with coordinates
    return radec

def fill_tasks(gal_data, dest, tag=''):
    """ Fills in shifter tasks with the data provided """
    # Read in galaxy file
    if isinstance(gal_data, str):
        print(gal_data)
        if os.path.exists(file):
            galaxies = pd.read_csv(file)
        else:
            print("The galaxy file does not exist")
            sys.exit(1)
    else:
        galaxies = gal_data
    
    # Format columns
    galaxies.columns = [c.lower() for c in galaxies.columns]
    
    if 'ls_id' not in galaxies.columns:
        print("Galaxy file does not contain ls_id column")
        sys.exit(1)
    
    line = shft_cmd
    if tag:
        line += f" --tag={tag}"
    
    # Format ra and dec into shifter commands
    tasks = [line.format(dest,ls_id) for ls_id in galaxies.ls_id]
    
    # Save file commands
    return tasks

def fill_from_db(dest, tag='', query=None):
    engine = sqlalchemy.create_engine(conn_string)
    conn = engine.connect()
    
    if query is None:
        query = f"SELECT * FROM bookkeeping WHERE stage=1"
    if tag: query+=f" AND tag={tag}"
    
    bkdata = pd.DataFrame(conn.execute(query))
    
    tasks = fill_tasks(bkdata, dest, tag=tag)
    
    conn.close()
    
    return tasks

def save_batches(tasks, n_batches, data_dir='.', dest_filename=task_filename):
    # if isinstance(tasks, list):
    #     tasks = pd.Series(tasks)
    if isinstance(dest_filename, str): dest_filename=dest_filename.split('.')
    dest_filename.insert(-1, '_0')
    
    N = len(tasks)
    start = 0
    # Loop through batches
    for i in range(n_batches):
        # Get limits
        end = int((i+1) / n_batches * N)

        # Get dest filename
        dest_filename[-2] = f'_{i}'
        dest = os.path.join(data_dir, ''.join(dest_filename))

        # Save batch to file
        with open(dest, 'w+') as file:
            file.write('\n'.join(tasks[start:end]))
        # tasks.iloc[start:end].to_csv(dest, sep='\n', header=False, index=False, quoting=3)
        print(f"File created: {dest}")

        # Reset start
        start = end

def prep_run(file=None, dest=None, tag='', radius=None, batch=None, masterlens=False, query=None):
    # Check arguments
    if dest is None:
        if file is not None:
            dest = os.path.dirname(file)
        else:
            dest = os.getcwd()
    dest = os.path.expandvars(dest)
    
    # Check source file and destination folder
    if file is not None and not os.path.isfile(file):
        print(f"Source file does not exist: {file}")
        sys.exit()
    if not os.path.isdir(dest):
        print(f"Destination folder does not exist or is not a directory: {dest}")
        sys.exit()
    
    # Check for masterlens database flag
    if masterlens:
        # Get csv with just RA, DEC
        radec = process_tsv(file, dest)
        # Save file
        filename = '.'.join(coord_filename)
        filename = os.path.join(dest, filename)
        radec.to_csv(filename, index=False)
        print(f"File created: {filename}")
        # Use new file for filling tasks
        file = filename
    
    if file is not None: # Fill tasks from data in file
        tasks = fill_tasks(file, dest, tag)
    else: # Fill tasks from data in db
        tasks = fill_from_db(dest=dest, tag=tag, query=query)
    
    
    # Save in batches
    if batch is not None:
        save_batches(tasks, batch, data_dir=dest)
    else:
        save_batches(tasks, 1, data_dir=dest)
    
if __name__=='__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create a task file from a table of galaxy coordinates")
    parser.add_argument("-m", "--masterlens", action='store_true', 
                        help = "Indicates original file is a TSV from the masterlens database.")
    parser.add_argument("-f", "--file", type=str, default = None,
                        help = "Source file for target galaxies (must include RA and DEC).")
    parser.add_argument("-d", "--dest", nargs='?', type=str, default=None,
                        help = "Destination folder for generated lens files.")
    parser.add_argument("-q", "--query", type=str, default=None, help='query for bookkeeping database')
    parser.add_argument("-t", "--tag", nargs='?', type=str, default = '', 
                        help = "Tag for database lookup.")
    parser.add_argument("-rd", "--radius", nargs='?', type=str, default='.000277778', 
                        help="Search radius (in degrees).")
    parser.add_argument("-b", "--batch", nargs='?', type=int, help = "Number of batches")
    args = parser.parse_args()
    
    prep_run(**vars(args))
    