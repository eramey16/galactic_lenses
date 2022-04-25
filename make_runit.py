### make_runit.py : makes a runit file for taskfarmer based on user-provided options
### Author: Emily Ramey
### Date: 4/13/22

import pandas as pd
import argparse
import os
import sys

runit_file = "runit_template"

def make_runit(dest, time, nodes, cores, const, taskfile):
    """ Makes a runit file from a template """
    # Load template file
    with open(runit_file, 'r') as file:
        runit_text = file.read()
    
    # Replace template info with user inputs
    runit_text = runit_text.format(dest, time, nodes, cores, const, dest, taskfile)
    
    # Save resulting runit file
    filename = os.path.join(dest, 'runit')
    with open(filename, 'w') as file:
        file.write(runit_text)
        print(f"File created: {filename}")

if __name__=='__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="runit file for a taskfarmer job")
    parser.add_argument("-s", "--taskfile", type=str, help = "Name of taskfile for taskfarmer")
    parser.add_argument("-d", "--dest", type=str, help = "Destination folder for generated files")
    parser.add_argument("-t", "--time", type=str, default = '04:00:00', help = "Time for job to run")
    parser.add_argument("-N", "--nodes", type=str, default='15', help = "Number of nodes (>=2)")
    parser.add_argument("-c", "--cores", type=str, default='32', help = "Number of cores")
    parser.add_argument("-o", "--constraint", type=str, choices=['knl', 'haswell'], default = 'knl', help = "System constraint")
    args = parser.parse_args()
    
    # Check arguments
    if args.dest is None:
        args.dest = os.path.dirname(args.taskfile)
    
    # Make new runit file
    make_runit(args.dest, args.time, args.nodes, args.cores, args.constraint, args.taskfile)