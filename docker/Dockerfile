FROM continuumio/miniconda3:latest

SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y gfortran

# Install astroconda
RUN conda config --add channels http://ssb.stsci.edu/astroconda
RUN conda install python ipython
RUN conda install numpy pandas scipy astropy 
RUN conda install h5py hdf5

RUN conda install -c conda-forge emcee dynesty

RUN python -m pip install astro-sedpy

WORKDIR /
ENV SPS_HOME /
RUN git clone https://github.com/cconroy20/fsps.git
RUN python -m pip install fsps

# # pip installs
# RUN pip install --upgrade noaodatalab

# # conda installs
# RUN conda install ipython numpy scipy
# RUN conda install pandas=1.4.2 SQLAlchemy=1.4.32

# # astropy installs
# RUN conda install -c astropy corner emcee h5py matplotlib

# # conda-forge installs
# RUN conda install -c conda-forge dynesty hdf5 mpi mpi4py

# ADD classify.py gradient_boosted/classify.py
# ADD rf.model gradient_boosted/rf.model
# ADD photoz_hm_params_short_dpon_on.py gradient_boosted/photoz_hm_params_short_dpon_on.py
# ADD util.py gradient_boosted/util.py

# WORKDIR gradient_boosted
# WORKDIR exports 
# WORKDIR ../../

# # /bin/sh -c conda update conda && conda install -c conda-forge hdf5 mpi mpi4py
# # /bin/sh -c pip install --upgrade astropy emcee h5py matplotlib noaodatalab numpy pandas psycopg2 requests scipy six
# # 