#import statements
from dl import queryClient as qc, helpers
from dl import authClient as ac
from dl import storeClient as sc
from dl.helpers.utils import convert
from prospect.io.read_results import results_from
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import h5py
import psycopg2
import requests
import random
import datetime
import random
import argparse
import pickle
import glob
import sys
import os
import math

bands = ['g', 'r', 'z', 'w1', 'w2']
trac_cols = ['ls_id', 'ra', 'dec', type] \
            + ['dered_mag_'+b for b in band] \
            + ['dered_flux+'+b for b in band] \
            + ['snr_'+b for b in band] \
            + ['flux_ivar_'+b for b in band]
phot_z_cols = ['z_phot_median', 'z_phot_std', 'z_spec']

#given an RA and DEC, pull magnitudes, magnitude uncertainties, redshifts from NOAO

def get_galaxy(ra,dec,radius=0.0002777777778,data_type=None,limit=1):
    """Queries the NOAO Legacy Survey Data Release 8 Tractor and Photo-Z tables

    Parameters:
        ra (float,int): Right ascension (RA) used in the q3c radial query to NOAO
        dec (float,int): Declination (DEC) used in the q3c radial query to NOAO
        radius (float,int): Radius in degrees used in the q3c radial query to NOAO.
            Defaults to 1 arcseconds/0.0002777 degrees.
        data_type (str): Data type to filter query results from NOAO. Should really
            only use types "PSF","REX","DEV","EXP","COMP", or "DUP". Defaults to
            "None" in order to return all results. You can bundle types to return
            data from multiple sources; follow PostGres formatting to do so.
        limit (int):Limit the number of rows returned from NOAO query. Defaults to 1.
    Returns:
        df_final: Pandas dataframe contained query results
    Raises:
        ValueError: If no objects are found
        ConnectionError: If NOAO times out. You can usually re-run the function and
            it'll work.
    """
    data = pd.DataFrame()

    query =["""SELECT trac.ls_id, trac.ra, trac.dec,trac.type,trac.dered_mag_g,trac.dered_mag_r,trac.dered_mag_z,
    trac.dered_mag_w1,trac.dered_mag_w2,1/NULLIF(trac.snr_g,0),1/NULLIF(trac.snr_r,0),1/NULLIF(trac.snr_z,0),
    1/NULLIF(trac.snr_w1,0),1/NULLIF(trac.snr_w2,0),phot_z.z_phot_median,phot_z.z_phot_std,phot_z.z_spec, 
    trac.dered_flux_g, trac.dered_flux_r, trac.dered_flux_z, trac.dered_flux_w1, trac.dered_flux_w2,
    trac.dchisq_1, trac.dchisq_2, trac.dchisq_3, trac.dchisq_4, trac.dchisq_5,
    trac.rchisq_g, trac.rchisq_r, trac.rchisq_z, trac.rchisq_w1, trac.rchisq_w2,
    trac.sersic, trac.sersic_ivar,
    trac.psfsize_g, trac.psfsize_r, trac.psfsize_z,
    trac.shape_r, trac.shape_r_ivar, trac.shape_e1, trac.shape_e1_ivar, trac.shape_e2, trac.shape_e2_ivar
    FROM ls_dr9.tractor AS trac 
    INNER JOIN ls_dr9.photo_z AS phot_z ON trac.ls_id = phot_z.ls_id WHERE (q3c_radial_query(ra,dec,{},{},{})) """,
    """ ORDER BY q3c_dist({}, {}, trac.ra, trac.dec) ASC""",
    """ LIMIT {}"""]

    if data_type is not None:
        query.insert(1,""" AND (type = '{}') """)
    else:
        data_type=''
        query.insert(1,"""{}""")
    query = ''.join(query)
    try:
        result10 = qc.query(sql=query.format(ra,dec,radius,data_type,ra,dec,limit))
        a = convert(result10, "pandas")
        if a.empty:
            raise ValueError(f"No objects within {radius} of {ra},{dec}")
        else:
            data = data.append(a,ignore_index=True)
            data.columns = ["ls_id", "ra", "dec","type","dered_mag_g","dered_mag_r","dered_mag_z","dered_mag_w1",
                            "dered_mag_w2",'unc_g','unc_r','unc_z','unc_w1','unc_w2',
                                 "z_phot_median","z_phot_std",'z_spec','dered_flux_g','dered_flux_r', 'dered_flux_z','dered_flux_w1','dered_flux_w2']
            data.to_csv(f"/gradient_boosted/exports/galaxies_{datetime.datetime.now().time()}.csv",mode="a+",header=False)
            return data
    except requests.exceptions.ConnectionError as e:
        return f"ConnectionError: {e} $\nOn RA:{ra},DEC:{dec}"
    except qc.queryClientError as e:
        with open("lensed_ordered_final.csv","a+") as f:
            f.write(f"Error on:{ra},{dec}: {e}\n")
        print(f"Query Client Error: {e} $\nRecorded on {ra},{dec} ")

#arg-parse section

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r","--ra",type=float, help = "RA selection")
    parser.add_argument("-d","--dec",type=float, help="DEC selection")
    parser.add_argument("-rd", "--radius",type=float, default=0.0002777777778, help = "Radius for q3c radial query")
    parser.add_argument("-t","--type",help="Object type to search for",type=str,choices=["DUP","DEV","EXP","REX","COMP","PSF"])

    args = parser.parse_args()

    data = get_galaxy(args.ra,args.dec,args.radius,args.type,1)
    print(f'Object found: {data["ls_id"].values[0]}')
    mags = [float(data['dered_mag_g'].values[0]),
float(data['dered_mag_r'].values[0]),
float(data['dered_mag_z'].values[0]),
float(data['dered_mag_w1'].values[0]),
float(data['dered_mag_w2'].values[0])]
    uncs = [float(data['unc_g'].values[0]),
float(data['unc_r'].values[0]),
float(data['unc_z'].values[0]),
float(data['unc_w1'].values[0]),
float(data['unc_w2'].values[0])]
    fluxes_1 = [float(data['dered_flux_g'].values[0]),
float(data['dered_flux_r'].values[0]),
float(data['dered_flux_z'].values[0]),
float(data['dered_flux_w1'].values[0]),
float(data['dered_flux_w2'].values[0])]

    print(f'Redshift: {float(data["z_phot_median"].values[0])}')

    if data["z_spec"].values[0] == -99:
        red_value = data["z_phot_median"].values[0]
    else:
        red_value = data["z_spec"].values[0]
    os.system(f'python /gradient_boosted/photoz_hm_params_short_dpon_on.py --objid={int(data["ls_id"].values[0])} --dynesty --object_redshift={float(red_value)} --outfile=/gradient_boosted/exports/{int(data["ls_id"].values[0])} --mag_in="{mags}" --mag_unc_in="{uncs}"')

    h5_file = f"/gradient_boosted/exports/{int(data['ls_id'].values[0])}.h5"

    def replace_neg(input):
        """Checks if scaled input values are greater than 0. If they are not,
        the minimum scaled input value that is, replaces negative values in the
        final output.

        Parameters:
            input: Some iterable consisting of numbers.
        Returns:
            output: A list of scaled values, negatives replaced w/ minimum.
        """
        output = []
        neg = []
        for x in input:
            if -2.5*np.log(x) > 0:
                output.append(-2.5*np.log(x))
            else:
                neg.append(x)
        min_o = min(output)
        for x in range(len(neg)):
            output.append(min_o)
        return output


    def get_magflux(flux, unc):
        """Divides flux by uncertainty.
        """
        output = []
        for x, y in zip(flux,unc):
            output.append(x/(y))
        return output

    def chi_s(model_file):
        """Compute chi squared statistic on model and observed photometry
        for a particular object

        Paramaters:
            model_file (str): HDF5 file output from Prospector for an object
        Returns:
            final_sum (float): Chi squared statistic
        """
        final_sum = 0
        res, obs, mod = results_from(model_file, dangerous=False)
        for model, observed, unc in zip(res['bestfit']['photometry'],res['obs']['maggies'],res['obs']['maggies_unc']):
            final_sum += ((observed-model)**2)/(unc**2)
        return final_sum


    des_tmp_lensed = []
    des_tmp_ids = []
    des_tmp_values = []

    des_tmp_lensed.append([0])
    h5_file_simple = h5_file.split("/")[3].split('.')[0]
    des_tmp_ids.append(int(h5_file_simple))

    des_tmp_values.append(np.concatenate((replace_neg(h5py.File(h5_file,'r')['obs']['maggies'][:]), get_magflux(fluxes_1,h5py.File(h5_file,'r')['obs']['maggies_unc'][:]),[pickle.loads(h5py.File(h5_file,'r').attrs['run_params'])['object_redshift']],
                                        h5py.File(h5_file,'r')['bestfit']['photometry'][:],
                                        [chi_s(h5_file)])).tolist())

    c = list(zip(des_tmp_values,des_tmp_lensed,des_tmp_ids))
    random.shuffle(c)
    des_tmp_values,des_tmp_lensed,des_tmp_ids = zip(*c)

    dict_2 = {}
    dict_2['values'] = des_tmp_values
    dict_2['ids'] = des_tmp_ids
    dict_2['lensed'] = des_tmp_lensed


    #make into pandas dataframe
    des_final = pd.DataFrame(dict_2)

    #attempt to remove useless rows
    for index, row in des_final.iterrows():
        if np.inf in row['values']:
            final = des_final.drop(index)
        if np.nan in row['values'] :
            final = des_final.drop(index)

    des_final = des_final.reset_index()
    des_final= des_final.drop('index',axis=1)

    des_train_data = des_final.iloc[:]

    #match ids to lensed values for later tracking
    des_y_train = []
    for ls_id, lensed_value in zip(des_train_data['ids'].iloc[:],des_train_data['lensed'].iloc[:]):
        des_y_train.append([ls_id,lensed_value])


    #drop lensed column as it's already in y_train now
    des_train_data = des_train_data.drop(['lensed'],axis=1)
    des_train_data = des_train_data.drop(['ids'],axis=1)

    #trying to get a list of the main list of values
    des_X_train = [x[0] for x in des_train_data.values]

    count = -1
    for x in des_X_train:
        if np.isfinite(x).all():
            count += 1
            continue
        else:
            count +=1
            des_X_train = np.delete(des_X_train,[count],axis=0)
            des_y_train = np.delete(des_y_train,[count],axis=0)
            count -= 1
    xgb_model = XGBClassifier()
    xgb_model.load_model("/gradient_boosted/grad.model")
    predictions = xgb_model.predict(np.array(des_tmp_values))
    data_dict = {'ls_id':data['ls_id'].values[0],'lensed':predictions}
    print(data_dict)
    df = pd.DataFrame(data_dict)
    df.to_csv(f'/gradient_boosted/exports/{data["ls_id"].values[0]}.csv',index=False)
