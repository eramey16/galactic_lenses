FROM continuumio/miniconda3:latest

SHELL ["/bin/bash", "-c"]

# RUN apt-get update
# RUN apt-get install -y gfortran nano vim

RUN \
    apt-get update        && \
    apt-get install --yes    \
        build-essential gfortran nano vim wget && \
    apt-get clean all

# Install astroconda
RUN conda config --add channels http://ssb.stsci.edu/astroconda
RUN conda install -y python ipython numpy pandas scipy astropy h5py hdf5

RUN conda install -y -c conda-forge emcee dynesty=2.1.4 schwimmbad tdqm

RUN python -m pip install astro-sedpy

WORKDIR /
ENV SPS_HOME '/fsps'
RUN git clone https://github.com/cconroy20/fsps.git
RUN python -m pip install fsps

ENV mpich=4.0.2
ENV mpich_prefix=mpich-$mpich

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    ./configure  FFLAGS=-fallow-argument-mismatch  FCFLAGS=-fallow-argument-mismatch  && \
    make -j 4                                                               && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix

RUN conda install -c conda-forge mpi4py six curl schwimmbad

RUN python -m pip install astro-prospector noaodatalab

RUN conda install -y -c astropy astroquery corner
RUN conda install -y sqlalchemy psycopg2 requests scikit-learn

ADD classify.py /monocle/classify.py
ADD gallazzi_05_massmet.txt /monocle/gallazzi_05_massmet.txt
ADD rf.model /monocle/rf.model
ADD param_monocle.py /monocle/param_monocle.py
ADD util.py /monocle/util.py
# ADD __init__.py /monocle/__init__.py

# ENV PATH="${PATH}:/mycode"
ENV PYTHONPATH="${PYTHONPATH}:/monocle"

WORKDIR /monocle
WORKDIR exports

WORKDIR ../../
