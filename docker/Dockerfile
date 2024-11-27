FROM continuumio/miniconda3:latest

SHELL ["/bin/bash", "-c"]

# RUN apt-get update
# RUN apt-get install -y gfortran nano vim

RUN \
    apt-get update        && \
    apt-get install --yes    \
        less build-essential gfortran nano vim wget postgresql-client && \
    apt-get clean all

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

# RUN pip install --ignore-installed --no-cache-dir astro-datalab
RUN conda install --solver=classic conda-forge::conda-libmamba-solver conda-forge::libmamba conda-forge::libmambapy conda-forge::libarchive
RUN conda install -y python=3.11 ipython numpy pandas scipy astropy h5py hdf5
RUN conda install -y ipykernel sqlalchemy psycopg2 requests scikit-learn
RUN conda install -y -c conda-forge emcee dynesty schwimmbad mpi4py six curl
RUN conda install -y -c astropy astroquery corner

WORKDIR /
RUN git clone https://github.com/cconroy20/fsps.git /fsps
ENV SPS_HOME='/fsps'

RUN python -m pip install astro-sedpy tdqm astro-prospector httplib2 pycurl-requests pyvo specutils mkl fsps

RUN git clone https://github.com/astro-datalab/datalab.git && \
    cd datalab && \
    pip install . && \
    cd .. && \
    rm -rf datalab

# ADD classify.py /monocle/classify.py
# ADD gallazzi_05_massmet.txt /monocle/gallazzi_05_massmet.txt
# ADD rf.model /monocle/rf.model
# ADD param_monocle.py /monocle/param_monocle.py
# ADD prospect_conversion.py /monocle/prospect_conversion.py
# ADD db_util.py /monocle/db_util.py
# ADD __init__.py /monocle/__init__.py
ENV PYTHONPATH="/monocle/"
ENV PGPASSFILE="/credentials/pgpass.conf"
WORKDIR /monocle
WORKDIR exports

WORKDIR ../../
