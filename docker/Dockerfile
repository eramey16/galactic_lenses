FROM audreykoz/prospy:latest
RUN pip install --upgrade sklearn xgboost
RUN apt-get update
RUN apt-get -y install libssl-dev libcurl4-openssl-dev python-dev
RUN pip install --ignore-installed noaodatalab
ADD docker/classify.py gradient_boosted/classify.py
ADD docker/grad.model gradient_boosted/grad.model
ADD docker/photoz_hm_params_short_dpon_on.py gradient_boosted/photoz_hm_params_short_dpon_on.py
ADD util.py gradient_boosted/util.py
WORKDIR gradient_boosted 
WORKDIR exports 
WORKDIR ../../
