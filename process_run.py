### process_run.py - processes data from a prospector run
### Author - Emily Ramey
### Date - 6/6/2022

import pandas as pd
import numpy as np
import argparse
import os
import sys

if __name__=='__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process the files from a prospector run")
    parser.add_argument("-d", "--dir", type=str, help = "Directory where prospector output is stored")
    parser.add_argument("-o", "--output", nargs='?', type=str, choices=['db', 'csv'], 
                        help = "How files should be output (database or csv)")
    parser.add_argument("-t", "--tag", type=str, help="Name of output file (table name or file name)")
    args = parser.parse_args()
    
    