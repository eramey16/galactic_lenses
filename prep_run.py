### prep_run.py: prepares a batch prospector run by generating a tasks.txt file in the specified directory and with the specified galactic parameters
### Author: Emily Ramey
### Date: 4/11/22

### Imports
import pandas as pd
import numpy as np
import argparse
import os
import sys

### String to repeat in taskfarmer file
shft_cmd = "shifter --image=eramey16/gradient:latest" \
" --volume='{}:/gradient_boosted/exports'" \
" /opt/conda/bin/python /gradient_boosted/classify.py -r {} -d {} -rd {}"
dest_filename = ["tasks", ".txt"]
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

def fill_tasks(file, dest, rd):
    """ Fills in shifter tasks with the ra and dec info in file """
    # Read in galaxy file
    if os.path.exists(file):
        galaxies = pd.read_csv(file)
    else:
        print("No galaxy file found")
        sys.exit(1)
    
    # Format columns
    galaxies.columns = [c.lower() for c in galaxies.columns]
    
    if 'ra' not in galaxies.columns or 'dec' not in galaxies.columns:
        print("Galaxy file does not contain ra, dec coordinate columns")
        sys.exit(1)
    
    # Format ra and dec into shifter commands
    galaxies['run'] = [shft_cmd.format(dest,a,b, rd) for a,b in zip(galaxies.ra, galaxies.dec)]
    
    # Save file commands
    return galaxies['run']
    
if __name__=='__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create a task file from a table of galaxy coordinates")
    parser.add_argument("-m", "--masterlens", action='store_true', help = "Indicates original file is a TSV from the masterlens database")
    parser.add_argument("-f", "--file", type=str, help = "Source file for target galaxies (must include RA and DEC)")
    parser.add_argument("-d", "--dest", nargs='?', type=str, help = "Destination folder for generated lens files")
    parser.add_argument("-t", "--tag", nargs='?', type=str, default = '', help = "Tag for destination files")
    parser.add_argument("-rd", "--radius", nargs='?', type=str, default='.000277778', help="Search radius (in degrees)")
    args = parser.parse_args()
    
    # Check arguments
    if args.dest is None:
        args.dest = os.path.dirname(args.file)
    if args.tag:
        dest_filename.insert(1, "_"+args.tag)
        coord_filename.insert(1, "_"+args.tag)
    
    # Check source file and destination folder
    if not os.path.isfile(args.file):
        print(f"Source file is incorrect: {args.file}")
        sys.exit(1)
    if not os.path.isdir(args.dest):
        print(f"Destination folder does not exist or is not a directory: {args.dest}")
        sys.exit(1)
    
    # Check for masterlens database flag
    if args.masterlens:
        # Get csv with just RA, DEC
        radec = process_tsv(args.file, args.dest)
        # Save file
        filename = ''.join(coord_filename)
        filename = os.path.join(args.dest, filename)
        radec.to_csv(filename, index=False)
        print(f"File created: {filename}")
        # Use new file for filling tasks
        args.file = filename
    
    # Fill tasks from ra, dec in file
    tasks = fill_tasks(args.file, args.dest, args.radius)
    
    # Save to destination (tasks.txt)
    filename = ''.join(dest_filename)
    dest = os.path.join(args.dest, filename)
    tasks.to_csv(dest, sep='\t', header=False, index=False, quoting=3)
    
    print(f"File created: {dest}")
    