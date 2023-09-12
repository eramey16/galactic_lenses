### make_runit.py : makes a runit file for taskfarmer based on user-provided options
### Author: Emily Ramey
### Date: 4/13/22

import pandas as pd
import argparse
import os
import sys

runit_file = "runit_template"
task_filename = ["tasks", ".txt"]

def make_runit(dest, time, nodes, cores, 
               constraint, taskfile, tag=''):
    """ Makes a runit file from a template """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    runit_loc = os.path.join(dir_path, runit_file)
    
    # Load template file
    with open(runit_loc, 'r') as file:
        runit_text = file.read()
    
    # Replace template info with user inputs
    runit_text = runit_text.format(dest, time, nodes, cores, 
                                   constraint, dest, taskfile)
    
    # Save resulting runit file
    filename = os.path.join(dest, 'runit'+str(tag))
    with open(filename, 'w+') as file:
        file.write(runit_text)
        print(f"File created: {filename}")

if __name__=='__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="runit file for a taskfarmer job")
    parser.add_argument("-f", "--taskfile", type=str, help = "Name of taskfile for taskfarmer")
    parser.add_argument("-d", "--dest", type=str, help = "Destination folder for generated files")
    parser.add_argument("-t", "--time", type=str, default = '08:00:00', help = "Time for job to run")
    parser.add_argument("-N", "--nodes", type=str, default='15', help = "Number of nodes (>=2)")
    parser.add_argument("-c", "--cores", type=str, default='32', help = "Number of cores")
    parser.add_argument("-C", "--constraint", type=str, default='cpu', help = "Arch constraint")
    parser.add_argument("-b", "--batch", nargs='?', type=int, help = "Number of batches")
    parser.add_argument('--tag', nargs='?', type=str, default='', help='Tag for runit file')
    args = parser.parse_args()
    
    # Check arguments
    if args.dest is None:
        args.dest = os.getcwd()
    else:
        args.dest = os.path.expandvars(args.dest)
    
    # Make several runits
    if args.batch:
        task_filename.insert(-1, '_0')
        for i in range(args.batch):
            task_filename[-2] = f'_{i}' + ("_" if args.tag else "")
            taskfile = ''.join(task_filename)
            make_runit(args.dest, args.time, args.nodes, args.cores, args.constraint, 
                       ''.join(task_filename), tag=str(i))
    else: # Make one runit
        make_runit(args.dest, args.time, args.nodes, args.cores, args.constraint, 
                   args.taskfile, tag=args.tag)
