import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.impute import KNNImputer
import xgboost as xgb
import matplotlib.pyplot as plt
import argparse
import sqlalchemy as sa
from sqlalchemy.pool import NullPool

import db_util as util


# Define features - NO MISSING INDICATORS
color_features = ['g_r', 'i_z', 'r_i', 'r_z', 'w1_w2', 'z_w1']
chi_features = ['min_dchisq', 'avg_rchisq']
prosp_features = ['delta_z_frac', 'log_ssfr', 'dust', 
                'log_age', 'mass_unc_rel', 'dust_unc_rel',
                'mass_unc_asym', 'dust_unc_asym'
                ]

feature_cols = color_features + chi_features + prosp_features

train_file = os.path.expandvars('$SCRATCH/data/monocle/lrg_train.parquet')

def flux2mag(flux):
    mags = 22.5 - 2.5*np.log10(flux + 1e-10)
    return mags

class ModelTrainer:
    
    def __init__(self, model_type, tbl_name, n_aug=3, conn_string=util.conn_string):
        self.seed = 42
        self.engine = sa.create_engine(util.conn_string, poolclass=NullPool)
        self.tbl_name = tbl_name
        self.model_type=model_type
        self.n_aug=n_aug

        self.data = None
        if model_type=='XGB':
            self.model = xgb.XGBClassifier
            self.model_params = {
                'objective':'binary:logistic',
                'scale_pos_weight':None,
                'random_state':self.seed,
                'eval_metric':'auc',
                'n_jobs': -1,
                'tree_method': 'hist',
                'device': 'cuda',
            }
        elif model_type=='RF':
            self.model = RandomForestClassifier
            self.model_params = {
                'n_estimators': np.arange(50, 550, 50),
                'max_depth': np.arange(3, 21, 2),
                'max_samples': np.arange(0.7, 1.1, .1),
                'criterion': ('gini', 'entropy'),
                'max_features': ('sqrt', 'log2'),
                'random_state': [self.seed],
                'class_weight': ['balanced'],
            }
    
    def _load_data(self):
        print("Loading data...")
        try:
            with self.engine.connect() as conn:
                gal_tbl = sa.Table(self.tbl_name, sa.MetaData(), autoload_with=self.engine)
                stmt = sa.select(gal_tbl)
                df = pd.DataFrame(conn.execute(stmt))
        except sa.exc.OperationalError:
            print("Database connection failed, loading from file...")
            df = pd.read_parquet(train_file)
        
        # Add False variables since I forgot them in the db
        unlensed_df, lensed_df = df[df.lensed!=True].copy(), df[df.lensed==True].copy()
        unlensed_df['lensed']=False

        df = pd.concat([lensed_df, unlensed_df]).sample(frac=1, 
                                                random_state=self.seed).reset_index(drop=True)


        self.data = df
    
    def _add_prosp(self):
        """
        Add derived features from Prospector outputs
        """
        print("Adding prospector features...")
        df = self.data
        
        # Physical properties (use as-is, already properly scaled)
        df.rename(columns={'mass':'log_mass', 'age':'log_age', 
                                  'met':'log_met'}, inplace=True)
        
        # Redshift residual
        # df['delta_z'] = df['z_prospector'] - df['z_phot_median']
        df['delta_z_frac'] = (df['zred'] - df['z_phot_median']) / (1 + 
                                                            df['z_phot_median'])
        
        # SFR - needs log transform (linear M_sun/yr)
        # df['log_sfr'] = np.log10(df['sfr'] + 1e-10)  # Add small value to avoid log(0)
        
        # Specific SFR (SFR per unit mass)
        df['log_ssfr'] = np.log10(df['sfr'] + 1e-10) - df['log_mass']  # log(SFR/M) in yr^-1
        
        # Tau - log transform for better scaling
        # df['log_tau'] = np.log10(df['tau'] + 1e-10)  # log(Gyr)
        
        # Uncertainty-based features (fit quality indicators)
        # Relative uncertainties
        df['mass_unc_rel'] = (np.abs(df['mass_sig_plus']) + 
                                     np.abs(df['mass_sig_minus'])) / 2.0
        # df['sfr_unc_rel'] = (np.abs(df['sfr_unc_pos']) + np.abs(df['sfr_unc_neg'])) / 2.0
        # df['z_unc_rel'] = (np.abs(df['z_unc_pos']) + np.abs(df['z_unc_neg'])) / (2.0 * (df['z_prospector'] + 1e-10))
        # df['age_unc_rel'] = (np.abs(df['age_unc_pos']) + np.abs(df['age_unc_neg'])) / 2.0
        df['dust_unc_rel'] = (np.abs(df['dust_sig_plus']) + 
                                     np.abs(df['dust_sig_minus'])) / (2.0 * 
                                                                (df['dust'] + 1e-10))
        
        # Asymmetric uncertainty indicators (poor fit quality)
        df['mass_unc_asym'] = (np.abs(df['mass_sig_plus'] - df['mass_sig_minus']) / 
                               (np.abs(df['mass_sig_plus']) + np.abs(df['mass_sig_minus']) + 1e-10))
        # df['age_unc_asym'] = (np.abs(df['age_unc_pos'] - df['age_unc_neg']) / 
        #                       (np.abs(df['age_unc_pos']) + np.abs(df['age_unc_neg']) + 1e-10))
        df['dust_unc_asym'] = (np.abs(df['dust_sig_plus'] - df['dust_sig_minus']) / 
                               (np.abs(df['dust_sig_plus']) + np.abs(df['dust_sig_minus']) + 1e-10))
    
    def _augment_data(self, df):
        print("Augmenting data...")
        bands = ['g', 'r', 'i', 'z', 'w1', 'w2']
        augmented_df = df.copy()
        augmented_df['aug_idx'] = df.index
        augmented_df = pd.concat([augmented_df.copy() for _ in range(self.n_aug)], 
                    ignore_index=True)

        for b in util.bands:
            augmented_df['flux_sigma_'+b] = 1 / np.sqrt(augmented_df['flux_ivar_'+b])
            new_fluxes = np.random.normal(loc=augmented_df['dered_flux_'+b], scale=augmented_df['flux_sigma_'+b])
            augmented_df['dered_mag_'+b] = flux2mag(new_fluxes)
            augmented_df['is_augmented'] = True
        
        augmented_df['g_r'] = augmented_df['dered_mag_g'] - augmented_df['dered_mag_r']
        augmented_df['r_i'] = augmented_df['dered_mag_r'] - augmented_df['dered_mag_i']
        augmented_df['i_z'] = augmented_df['dered_mag_i'] - augmented_df['dered_mag_z']
        augmented_df['r_z'] = augmented_df['dered_mag_r'] - augmented_df['dered_mag_z']
        augmented_df['z_w1'] = augmented_df['dered_mag_z'] - augmented_df['dered_mag_w1']
        augmented_df['w1_w2'] = augmented_df['dered_mag_w1'] - augmented_df['dered_mag_w2']
        
        # Concat with original
        # df_combined = pd.concat([df.copy(), augmented_df], ignore_index=True)
        return augmented_df

    def prepare_data(self):
        print("Preparing data...")
        self._load_data()
        self._add_prosp()
        df = self.data

        # Calculate chi-squared columns
        rchisq = np.array(df[util.rchisq_labels])
        df['avg_rchisq'] = np.nanmean(rchisq, axis=1)
        
        dchisq = np.array(df[util.dchisq_labels])
        df['min_dchisq'] = np.nanmin(dchisq, axis=1)

        # Remove anything with more than 2 bands missing
        df = df[df[color_features].isnull().sum(axis=1)<2].reset_index(drop=True)
        df = df[df[prosp_features].isnull().sum(axis=1)<2].reset_index(drop=True)
        self.data = df

        df.columns = df.columns.astype(str)
        df = df.rename(str, axis="columns")
        X = df[feature_cols]

        if X.isnull().any().any():
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            X_imputed = imputer.fit_transform(X)
            X = pd.DataFrame(X_imputed, columns=feature_cols)

        df.loc[df.index, feature_cols] = X[feature_cols] # Replace feature cols
        self.data = df # just in case

    def train_model(self):
        print("Training model...")
        self.data = self.data.rename(str, axis='columns')

        # Augment data
        if self.n_aug > 1:
            df_lensed = self.data[self.data.lensed.astype(bool)==True]
            df_unlensed = self.data[self.data.lensed.astype(bool)==False]
            augmented_df = self._augment_data(df_lensed)
        
        # Now separate feature and target cols
        X, y = self.data[feature_cols], self.data['lensed']
        y = pd.Series(np.array(y).astype(int), name='lensed') # Formatting

        cv_scores = []

        kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=self.seed)
        for i, (train_index, val_index) in enumerate(kf.split(X, y)):
            # Training-validation split
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            self.model_params['scale_pos_weight'] = (len(y_train[
                y_train.astype(bool)!=True]) / len(y_train))
            
            if self.n_aug > 1:
                aug_train = augmented_df[augmented_df.aug_idx.isin(train_index)]
                X_aug, y_aug = aug_train[feature_cols], aug_train['lensed']

                X_train = pd.concat([X_train, X_aug]).reset_index(drop=True)
                y_train = pd.concat([y_train, y_aug]).reset_index(drop=True)
            
            # df_train = X_train.join(y_train)
            # df_val = X_val.join(y_val)

            # # Augment if required
            # if self.n_aug > 1:
            #     # Break into lensed and unlensed portions
            #     df_train_lensed = df_train[df_train['lensed'].astype(bool)==True]
            #     df_train_unlensed = df_train[df_train['lensed'].astype(bool)!=True]
            #     # Augment only lensed training data
            #     df_train_lensed = self._augment_data(df_train_lensed)
            #     df_train = pd.concat([df_train_lensed, df_train_unlensed], ignore_index=True)
            
            # Now we can finally use only the feature columns
            # X_train, y_train = df_train[feature_cols], df_train['lensed']
            # X_val, y_val = df_val[feature_cols], df_train['lensed']
            y_train, y_val = np.array(y_train).astype(int), np.array(y_val).astype(int)

            self.trained_model = self.model(**self.model_params)
            self.trained_model.fit(X_train, y_train)

            y_pred = self.trained_model.predict_proba(X_val)[:, 1]
            cv_scores.append(roc_auc_score(y_val, y_pred))
        
        print(f"\nMean CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        return cv_scores


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Train XGBoost or RandomForest model and output results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( "--model_type", '-m', help="Type of model to train" )
    parser.add_argument("--tbl_name", '-t', default='lrg_train', 
                        help="DB table name to use for training")
    parser.add_argument('--n_aug', '-n', default=3, type=int,
                        help='Number of augmentations for lensed data')
    args = parser.parse_args()

    model_t = ModelTrainer(**vars(args))
    model_t.prepare_data()
    cv_scores = model_t.train_model()