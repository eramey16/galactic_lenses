GALACTIC_LENSES
===============

A repo for running prospector on lens galaxies. Involves the [Cori](https://docs.nersc.gov/systems/cori/) system from [NERSC](https://www.nersc.gov/) and the [taskfarmer](https://docs.nersc.gov/jobs/workflow/taskfarmer/) module.

# Overview:
* prep_run.py : prepares a run of taskfarmer by making a tasks.py file
* classify.py : Audrey's classification program for galaxies

# How to use:
`prep_run.py [-m] -f <galaxy_info.csv> [-d </path/to/destination>] [-t <tag>]`

Where:
* -m is a tag indicating whether the file specified is a CSV file with normal RA and DEC columns or a TSV file from the master lens database. If the flag is included, a secondary CSV file (`coords.csv`) will be output to the destination with only the RA and DEC coordinates.
* galaxy_info.csv is either A) a CSV file containing columns RA and DEC (not case sensitive) for each galaxy if the -m flag is NOT included, or B) a TSV file downloaded from the Master Lens Database with columns ra_coord and dec_coord, if the -m flag IS included.
* /path/to/destination is the path to the folder where you would like the output files stored after the output tasks.txt is run.
* -t is a tag to be put on the tasks.txt file (and CSV file, if converting a TSV). For example, including `-t A` would output `tasks_A.txt` to the destination rather than `tasks.txt`.

# Work done:
* Created a python script (prep_run.py) to make tasks.py files for each downloaded galaxy list
* Downloaded non-lensed (random) galaxy data from the [DESI Legacy Survey](https://datalab.noirlab.edu/query.php) (RA 150, DEC 2, DEV type, 1 degree search radius, first 1000 hits)
* Downloaded  lensed galaxy data (classes A, B, and C) from the [Master Lens Database](https://test.masterlens.org/search.php?)
* Downloaded lensed galaxy data (classes A, B, and C) from [Huang et al 2021](https://sites.google.com/usfca.edu/neuralens/publications/lens-candidates-huang-2020b?authuser=0)
* Created a task file for each of the galaxy categories above

# Thanks to:
* Peter Nugent