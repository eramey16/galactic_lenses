GALACTIC_LENSES
===============

A repo for running prospector on lens galaxies. Involves the [Cori](https://docs.nersc.gov/systems/cori/) system from [NERSC](https://www.nersc.gov/) and the [taskfarmer](https://docs.nersc.gov/jobs/workflow/taskfarmer/) module.

# Overview:
* prep_run.py : prepares a run of taskfarmer by making a tasks.py file
* make_runit.py : creates a runit file given a tasks file
* classify.py : Audrey's classification program for galaxies

# How to use:
`python prep_run.py [-m] -f <galaxy_info.csv> [-d </path/to/destination>] [-t <tag>]`

Where:
* -m is a tag indicating whether the file specified is a CSV file with normal RA and DEC columns or a TSV file from the master lens database. If the flag is included, a secondary CSV file (`coords.csv`) will be output to the destination with only the RA and DEC coordinates.
* galaxy_info.csv is either A) a CSV file containing columns RA and DEC (not case sensitive) for each galaxy if the -m flag is NOT included, or B) a TSV file downloaded from the Master Lens Database with columns ra_coord and dec_coord, if the -m flag IS included.
* /path/to/destination is the path to the folder where you would like the output files stored after the output tasks.txt is run.
* -t is a tag to be put on the tasks.txt file (and CSV file, if converting a TSV). For example, including `-t A` would output `tasks_A.txt` to the destination rather than `tasks.txt`.

`python make_runit.py [-s] </path/to/taskfile> [-d </path/to/destination>] [-t <00:00:00>] [-N <# nodes>] [-c <# cores>] [-o <constraint>]`

Where:
* -s is the path to an existing taskfile
* -d is the (optional) path to the destination folder for the runit (default same as taskfile)
* -t is the time for the taskfarmer job (default 04:00:00)
* -N is the number of nodes for the job to run on (must be at least 2, default 15)
* -c is the number of cores for the job to run on (max 32 for haswell, 68 for knl, default 32)
* -o is the [constraint](https://docs.nersc.gov/performance/knl/getting-started/) the job will run on: Knight's Landing (KNL) or Cori Haswell (default knl)

# Work done:
* Created a python script (prep_run.py) to make tasks.py files for each downloaded galaxy list
* Downloaded non-lensed (random) galaxy data from the [DESI Legacy Survey](https://datalab.noirlab.edu/query.php) (RA 150, DEC 2, DEV type, 1 degree search radius, first 1000 hits)
* Downloaded  lensed galaxy data (classes A, B, and C) from the [Master Lens Database](https://test.masterlens.org/search.php?)
* Downloaded lensed galaxy data (classes A, B, and C) from [Huang et al 2021](https://sites.google.com/usfca.edu/neuralens/publications/lens-candidates-huang-2020b?authuser=0)
* Created a task file for each of the galaxy categories above
* Created a runit file maker with various user inputs
* Edited classify.py to include extra columns
* Combined all the lens galaxies in one single batch job


### Query for (unlensed) training data
```
SELECT trac.ls_id, trac.ra, trac.dec, trac.type, trac.dered_mag_g, trac.dered_mag_r, trac.dered_mag_z, trac.dered_mag_w1, trac.dered_mag_w2, trac.snr_g, trac.snr_r, trac.snr_z, trac.snr_w1, trac.snr_w2, phot_z.z_phot_median, phot_z.z_phot_std, phot_z.z_spec, trac.dered_flux_g, trac.dered_flux_r, trac.dered_flux_z, trac.dered_flux_w1, trac.dered_flux_w2, trac.dchisq_1, trac.dchisq_2, trac.dchisq_3, trac.dchisq_4, trac.dchisq_5, trac.rchisq_g, trac.rchisq_r, trac.rchisq_z, trac.rchisq_w1, trac.rchisq_w2, trac.psfsize_g, trac.psfsize_r, trac.psfsize_z, trac.sersic, trac.sersic_ivar, trac.shape_e1, trac.shape_e1_ivar, trac.shape_e2, trac.shape_e2_ivar, trac.shape_r, trac.shape_r_ivar
FROM ls_dr9.tractor AS trac
INNER JOIN ls_dr9.photo_z AS phot_z ON trac.ls_id = phot_z.ls_id
WHERE (q3c_radial_query(ra,dec, 150, 2, 5))
AND NOT trac.type='PSF'
```
* Changed the 1/NULLIF statements to just return the SNR for now

# Thanks to:
* Peter Nugent