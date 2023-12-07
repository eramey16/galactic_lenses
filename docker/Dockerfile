FROM continuumio/miniconda3:latest

SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y gfortran nano vim

# ADD https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.0.tar.bz2 .
# RUN tar xf openmpi-5.0.0.tar.bz2 \
#     && cd openmpi-5.0.0 \
#     && ./configure --prefix=/ \
#     && make -j4 all \
#     && make install \
#     && cd .. && rm -rf \
#     openmpi-5.0.0 openmpi-5.0.0.tar.bz2 /tmp/*

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

ADD classify.py gradient_boosted/classify.py
ADD rf.model gradient_boosted/rf.model
ADD photoz_hm_params_short_dpon_on.py gradient_boosted/photoz_hm_params_short_dpon_on.py
ADD util.py gradient_boosted/util.py

ENV PATH="${PATH}:/gradient_boosted"

WORKDIR gradient_boosted
WORKDIR exports

WORKDIR ../../