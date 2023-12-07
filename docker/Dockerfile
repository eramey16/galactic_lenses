FROM continuumio/miniconda3:latest

SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y gfortran nano vim

# Install astroconda
RUN conda config --add channels http://ssb.stsci.edu/astroconda
RUN conda install python ipython
RUN conda install numpy pandas scipy astropy 
RUN conda install h5py hdf5

RUN conda install -c conda-forge emcee dynesty schwimmbad

RUN python -m pip install astro-sedpy

WORKDIR /
ENV SPS_HOME '/fsps'
RUN git clone https://github.com/cconroy20/fsps.git
RUN python -m pip install fsps

RUN conda install -c conda-forge mpi mpi4py mpich six curl

RUN python -m pip install astro-prospector noaodatalab

RUN conda install -c astropy astroquery corner
RUN conda install sqlalchemy psycopg2 requests scikit-learn

ADD classify.py /monocle/classify.py
ADD rf.model /monocle/rf.model
ADD photoz_hm_params_short_dpon_on.py /monocle/photoz_hm_params_short_dpon_on.py
ADD util.py /monocle/util.py
# ADD __init__.py /monocle/__init__.py

# ENV PATH="${PATH}:/mycode"
ENV PYTHONPATH="${PYTHONPATH}:/monocle"

WORKDIR /monocle
WORKDIR exports

WORKDIR ../../