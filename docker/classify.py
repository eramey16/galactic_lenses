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

import sqlalchemy
import util

bands = ['g', 'r', 'z', 'w1', 'w2']
trac_cols = ['ls_id', 'ra', 'dec', 'type'] \
            + ['dered_mag_'+b for b in bands] \
            + ['dered_flux_'+b for b in bands] \
            + ['snr_'+b for b in bands] \
            + ['flux_ivar_'+b for b in bands] \
            + ['dchisq_'+str(i) for i in range(1,6)] \
            + ['rchisq_'+b for b in bands] \
            + ['sersic', 'sersic_ivar'] \
            + ['psfsize_g', 'psfsize_r', 'psfsize_z'] \
            + ['shape_r', 'shape_e1', 'shape_e2'] \
            + ['shape_r_ivar', 'shape_e1_ivar', 'shape_e2_ivar']
phot_z_cols = ['z_phot_median', 'z_phot_std', 'z_spec']
all_cols = ['trac.'+col for col in trac_cols] + ['phot_z.'+col for col in phot_z_cols]

query_cols = ', '.join(all_cols)

output_dir = '/gradient_boosted/exports/' # Uncomment these two lines before pushing to docker
input_dir = '/gradient_boosted/'
# output_dir = '/global/cscratch1/sd/eramey16/gradient/' # Comment these two lines before pushing to docker
# input_dir='/global/cscratch1/sd/eramey16/gradient/'
prosp_file = 'photoz_hm_params_short_dpon_on.py'

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
    # Set up basic query
    query =[f"""SELECT {query_cols} FROM ls_dr9.tractor AS trac 
    INNER JOIN ls_dr9.photo_z AS phot_z ON trac.ls_id = phot_z.ls_id 
    WHERE (q3c_radial_query(ra,dec,{ra},{dec},{radius})) """,
    f""" ORDER BY q3c_dist({ra}, {dec}, trac.ra, trac.dec) ASC""",
    f""" LIMIT {limit}"""]
    # Add data type, if specified
    if data_type is not None:
        query.insert(1,f""" AND (type = '{data_type}') """)
    # Join full query
    query = ''.join(query)
    
    # Send query to database
    try:
        result10 = qc.query(sql=query) # result as string
        data = convert(result10, "pandas") # result as dataframe
        if data.empty:
            raise ValueError(f"No objects within {radius} of {ra},{dec}")
            
        # Save to file
        ls_id = int(data.ls_id[0])
        basic_file = f"{output_dir}{ls_id}.csv"
        data.to_csv(basic_file, index=False)
        return data
    # Connection error(s) # Still going to fail when it's called and doesn't return data? TODO
    except requests.exceptions.ConnectionError as e:
        return f"ConnectionError: {e} $\nOn RA:{ra},DEC:{dec}"
    except qc.queryClientError as e:
        with open("lensed_ordered_final.csv","a+") as f:
            f.write(f"Error on:{ra},{dec}: {e}\n")
        print(f"Query Client Error: {e} $\nRecorded on {ra},{dec} ")

def run_prospector(ls_id, redshift, mags, mag_uncs, prosp_file=prosp_file):
    """ Runs prospector with provided parameters """
    # Input and output filenames
    pfile = os.path.join(input_dir, prosp_file)
    outfile = os.path.join(output_dir, str(ls_id))
    
    # Run prospector with parameters
    os.system(f'python {pfile} --objid={ls_id} --dynesty --object_redshift={redshift} --outfile={outfile} ' \
              + f' --mag_in="{mags}" --mag_unc_in="{mag_uncs}"')


### MAIN FUNCTION
if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-r","--ra",type=float, help = "RA selection")
    parser.add_argument("-d","--dec",type=float, help="DEC selection")
    parser.add_argument("-rd", "--radius",type=float, default=0.0002777777778, help = "Radius for q3c radial query")
    parser.add_argument("-t","--type",help="Object type to search for",type=str,choices=["DUP","DEV","EXP","REX","COMP","PSF"])
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get galaxy data from NOAO # TODO: will fail with no catch if there's an error
    data = get_galaxy(args.ra,args.dec,args.radius,args.type,1)
    data = data.iloc[0] # Series of values
    
    # Find 
    print(f'Object found: {data.ls_id}')
    
    # magnitudes, uncertainties, and fluxes
    mags = [data['dered_mag_'+b] for b in bands]
    mag_uncs = [ 2.5 / (np.log(10) * data['dered_flux_'+b] * 
                        np.sqrt(data['flux_ivar_'+b])) for b in bands]
    fluxes = [data['dered_flux_'+b] for b in bands]
    
    # Print
    print(f'Redshift: {data.z_phot_median}')
    
    # Data shortcuts
    ls_id = data.ls_id
    red_value = data.z_phot_median
    
    # Run Prospector
    run_prospector(ls_id, red_value, mags, mag_uncs)
    
    # Output file names
    h5_file = os.path.join(output_dir, f'{ls_id}.h5')
    basic_file = os.path.join(output_dir, f'{ls_id}.csv')

#     def replace_neg(input):
#         """Checks if scaled input values are greater than 0. If they are not,
#         the minimum scaled input value that is, replaces negative values in the
#         final output.

#         Parameters:
#             input: Some iterable consisting of numbers.
#         Returns:
#             output: A list of scaled values, negatives replaced w/ minimum.
#         """
#         output = []
#         neg = []
#         for x in input:
#             if -2.5*np.log(x) > 0:
#                 output.append(-2.5*np.log(x))
#             else:
#                 neg.append(x)
#         min_o = min(output)
#         for x in range(len(neg)):
#             output.append(min_o)
#         return output


#     def get_magflux(flux, unc):
#         """Divides flux by uncertainty.
#         """
#         output = []
#         for x, y in zip(flux,unc):
#             output.append(x/(y))
#         return output

#     def chi_s(model_file):
#         """Compute chi squared statistic on model and observed photometry
#         for a particular object

#         Paramaters:
#             model_file (str): HDF5 file output from Prospector for an object
#         Returns:
#             final_sum (float): Chi squared statistic
#         """
#         final_sum = 0
#         res, obs, mod = results_from(model_file, dangerous=False)
#         for model, observed, unc in zip(res['bestfit']['photometry'],res['obs']['maggies'],res['obs']['maggies_unc']):
#             final_sum += ((observed-model)**2)/(unc**2)
#         return final_sum


    des_tmp_lensed = []
    des_tmp_ids = []
    des_tmp_values = []

    des_tmp_lensed.append([0])
    h5_file_simple = h5_file.split("/")[-1].split('.')[0]
    des_tmp_ids.append(int(h5_file_simple))

    des_tmp_values.append(np.concatenate((replace_neg(h5py.File(h5_file,'r')['obs']['maggies'][:]), get_magflux(fluxes,h5py.File(h5_file,'r')['obs']['maggies_unc'][:]),[pickle.loads(h5py.File(h5_file,'r').attrs['run_params'])['object_redshift']],
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
    xgb_model.load_model(f"{input_dir}/grad.model") # this was only in gradient_boosted
    predictions = xgb_model.predict(np.array(des_tmp_values))
    data_dict = {'ls_id':ls_id,'lensed':predictions}
    print(data_dict)
    # df = pd.DataFrame(data_dict)
    
    # Append prediction to file
    df = pd.read_csv(basic_file)
    df['lensed'] = predictions
    df.to_csv(basic_file, index=False)

    
### TODO: Anything under the commented blocs isn't needed at all besides the last few lines.
###       Edit this to update the database instead of making files and get the final lensed value
###         based on the RandomForest classifier