import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import db_util as util
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.impute import KNNImputer
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
import seaborn as sns
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

seed = 42

def flux2mag(flux):
    mags = 22.5 - 2.5*np.log10(flux)
    return mags

def augment_lensed_from_raw(lensed_df, n_augmentations=3):
    """
    Augment magnitudes based on photometric uncertainties
    Recalculate colors from augmented magnitudes
    
    Parameters:
    -----------
    df : DataFrame
        Must contain flux and flux_ivar columns for each band
    n_augmentations : int
        Number of augmented copies per galaxy
    
    Returns:
    --------
    df_augmented : DataFrame
        Original + augmented samples
    """
    bands = ['g', 'r', 'i', 'z', 'w1', 'w2']
    # lensed = df[df.lensed==True]
    
    # augmented_df = lensed_df.copy()  # Start with original lenses
    augmented_df = pd.concat([lensed_df.copy() for _ in range(n_augmentations)], ignore_index=True)

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
    df_combined = pd.concat([lensed_df.copy(), augmented_df], ignore_index=True)
    
    print(f"Augmentation complete:")
    print(f"  Original samples: {len(df)}")
    print(f"  Augmented samples: {len(df_combined)}")
    print(f"  Augmentation factor: {len(df_combined) / len(df):.1f}x")
    
    return df_combined


def add_prospector_features(df):
    """
    Add derived features from Prospector outputs
    """
    
    # Physical properties (use as-is, already properly scaled)
    df.rename(columns={'mass':'log_mass', 'age':'log_age', 'met':'log_met'}, inplace=True)
    
    # Redshift residual
    # df['delta_z'] = df['z_prospector'] - df['z_phot_median']
    df['delta_z_frac'] = (df['zred'] - df['z_phot_median']) / (1 + df['z_phot_median'])
    
    # SFR - needs log transform (linear M_sun/yr)
    # df['log_sfr'] = np.log10(df['sfr'] + 1e-10)  # Add small value to avoid log(0)
    
    # Specific SFR (SFR per unit mass)
    df['log_ssfr'] = np.log10(df['sfr'] + 1e-10) - df['log_mass']  # log(SFR/M) in yr^-1
    
    # Tau - log transform for better scaling
    # df['log_tau'] = np.log10(df['tau'] + 1e-10)  # log(Gyr)
    
    # Uncertainty-based features (fit quality indicators)
    # Relative uncertainties
    df['mass_unc_rel'] = (np.abs(df['mass_sig_plus']) + np.abs(df['mass_sig_minus'])) / 2.0
    # df['sfr_unc_rel'] = (np.abs(df['sfr_unc_pos']) + np.abs(df['sfr_unc_neg'])) / 2.0
    # df['z_unc_rel'] = (np.abs(df['z_unc_pos']) + np.abs(df['z_unc_neg'])) / (2.0 * (df['z_prospector'] + 1e-10))
    # df['age_unc_rel'] = (np.abs(df['age_unc_pos']) + np.abs(df['age_unc_neg'])) / 2.0
    df['dust_unc_rel'] = (np.abs(df['dust_sig_plus']) + np.abs(df['dust_sig_minus'])) / (2.0 * (df['dust'] + 1e-10))
    
    # Asymmetric uncertainty indicators (poor fit quality)
    # df['mass_unc_asym'] = (np.abs(df['mass_unc_pos'] - df['mass_unc_neg']) / 
    #                        (np.abs(df['mass_unc_pos']) + np.abs(df['mass_unc_neg']) + 1e-10))
    # df['age_unc_asym'] = (np.abs(df['age_unc_pos'] - df['age_unc_neg']) / 
    #                       (np.abs(df['age_unc_pos']) + np.abs(df['age_unc_neg']) + 1e-10))
    # df['dust_unc_asym'] = (np.abs(df['dust_unc_pos'] - df['dust_unc_neg']) / 
    #                        (np.abs(df['dust_unc_pos']) + np.abs(df['dust_unc_neg']) + 1e-10))
    
    return df
    
def prepare_xgboost_model(df, augment_lensed=True, n_augmentations=3):
    """
    Prepare XGBoost model - impute missing data without indicators
    """

    # Add missing features to un-bias sample
    unlensed_df, lensed_df = df[df.lensed!=True].copy(), df[df.lensed==True].copy()
    # unlensed_df = match_iband(unlensed_df, lensed_df)
    unlensed_df['lensed']=False

    # Augment lensed galaxies if requested
    if augment_lensed and len(lensed_df) > 0:
        print("Augmenting lensed galaxies...")
        lensed_df = augment_lensed_magnitudes(lensed_df, n_augmentations)

    # Shuffle
    df = pd.concat([lensed_df, unlensed_df]).sample(frac=1).reset_index(drop=True)
    df = add_prospector_features(df)

    # Calculate chi-squared columns
    rchisq = np.array(df[util.rchisq_labels])
    df['avg_rchisq'] = np.nanmean(rchisq, axis=1)
    
    dchisq = np.array(df[util.dchisq_labels])
    df['min_dchisq'] = np.nanmin(dchisq, axis=1)
    
    # Define features - NO MISSING INDICATORS
    color_features = ['g_r', 'i_z', 'r_i', 'r_z', 'w1_w2', 'z_w1']
    chi_features = ['min_dchisq', 'avg_rchisq']
    prosp_features = ['delta_z_frac', 'log_ssfr', 'dust', 
                      'log_age', 'mass_unc_rel', 'dust_unc_rel']

    # Remove anything with more than 2 bands missing
    df = df[df[color_features].isnull().sum(axis=1)<2].reset_index(drop=True)
    df = df[df[prosp_features].isnull().sum(axis=1)<2].reset_index(drop=True)
    
    # Combine all features
    feature_cols = [str(f) for f in (color_features + chi_features + prosp_features)]
    
    # Prepare data
    X = df[feature_cols].copy()
    y = df['lensed'].copy()
    
    # Fix for column types
    X.columns = X.columns.astype(str)
    
    print(f"Missing values before imputation:")
    print(X.isnull().sum())
    
    # Impute ALL missing values using KNN
    if X.isnull().any().any():
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        
        # Fit on complete cases if possible
        complete_mask = X.notna().all(axis=1)
        if complete_mask.sum() > 10:
            imputer.fit(X.loc[complete_mask])
        else:
            imputer.fit(X)
        
        # Apply imputation
        X_imputed = imputer.transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    
    print(f"Missing values after imputation:")
    print(X.isnull().sum())
    
    # Calculate class weights
    class_counts = y.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]
    print(f"Class distribution: {class_counts}")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    return X, y, scale_pos_weight, feature_cols

def prepare_features_from_df(df, feature_cols):
    """
    Extract and impute features from a dataframe
    Assumes Prospector features and chi-squared already calculated
    """
    df.columns = df.columns.astype(str)
    X = df[feature_cols].copy()
    y = df['lensed'].copy().astype(int)

    # Ensure all column names are strings
    X.columns = X.columns.astype(str)
    
    # Remove rows with too many missing values
    color_features = ['g_r', 'i_z', 'r_i', 'r_z', 'w1_w2', 'z_w1']
    prosp_features = ['delta_z_frac', 'log_ssfr', 'dust', 
                      'log_age', 'mass_unc_rel', 'dust_unc_rel']
    
    valid_mask = (X[color_features].isnull().sum(axis=1) < 2) & \
                 (X[prosp_features].isnull().sum(axis=1) < 2)
    
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    
    # Impute missing values
    X = X.rename(str,axis="columns")
    if X.isnull().any().any():
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=feature_cols)
    
    return X, y

def train_xgboost_model_with_cv(df, n_augmentations=3):
    """
    Train XGBoost with proper CV - augment only training folds from raw data
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset with raw flux columns
    n_augmentations : int
        Number of augmented copies per lensed galaxy in training
    
    Returns:
    --------
    best_model, best_params, cv_scores, feature_importance, X_all, y_all
    """
    
    print("\n1. Preprocessing full dataset...")
    
    df.columns = df.columns.astype(str)

    # Add unlensed as False since I forgot in db
    df.loc[df.lensed!=True, 'lensed'] = False

    # Add Prospector features to full dataset
    df = add_prospector_features(df.copy())
    
    # Calculate chi-squared features
    rchisq = np.array(df[util.rchisq_labels])
    df['avg_rchisq'] = np.nanmean(rchisq, axis=1)
    
    dchisq = np.array(df[util.dchisq_labels])
    df['min_dchisq'] = np.nanmin(dchisq, axis=1)
    
    # Define features
    color_features = ['g_r', 'i_z', 'r_i', 'r_z', 'w1_w2', 'z_w1']
    chi_features = ['min_dchisq', 'avg_rchisq']
    prosp_features = ['delta_z_frac', 'log_ssfr', 'dust', 
                      'log_age', 'mass_unc_rel', 'dust_unc_rel']
    feature_cols = [str(f) for f in (color_features + chi_features + prosp_features)]
    # Remove anything with more than 2 bands missing
    df = df[df[color_features].isnull().sum(axis=1)<2].reset_index(drop=True)
    df = df[df[prosp_features].isnull().sum(axis=1)<2].reset_index(drop=True)
    
    # Get clean base dataset (no augmentation yet)
    X_base, y_base = prepare_features_from_df(df, feature_cols)
    
    # We need to keep track of which rows correspond to which original data
    # So we can augment properly in each fold
    df_clean = df.iloc[X_base.index].reset_index(drop=True)
    X_base = X_base.reset_index(drop=True)
    y_base = y_base.reset_index(drop=True)
    
    print(f"Clean dataset: {len(X_base)} samples")
    print(f"  Lensed: {(y_base==1).sum()}")
    print(f"  Unlensed: {(y_base==0).sum()}")
    
    # Define parameter grid
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.05, 0.1, 0.15],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }
    
    # Stratified CV
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed+1)
    
    cv_scores = []
    all_feature_importances = []
    
    print("\n2. Running 5-fold cross-validation with augmentation...")
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_base, y_base), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/5")
        print(f"{'='*60}")
        
        # Get original dataframes for this fold (with raw flux data)
        df_train_fold = df_clean.iloc[train_idx].copy()
        df_test_fold = df_clean.iloc[test_idx].copy()
        
        # Separate lensed and unlensed in training fold
        df_train_lensed = df_train_fold[df_train_fold.lensed == True].copy()
        df_train_unlensed = df_train_fold[df_train_fold.lensed != True].copy()
        
        print(f"Train: {len(df_train_fold)} ({len(df_train_lensed)} lensed)")
        print(f"Test:  {len(df_test_fold)} ({(df_test_fold.lensed==True).sum()} lensed)")
        
        # Augment lensed training data from raw fluxes
        if len(df_train_lensed) > 0 and n_augmentations > 0:
            print(f"Augmenting {len(df_train_lensed)} lensed samples from raw fluxes...")
            df_train_lensed_aug = augment_lensed_from_raw(df_train_lensed, n_augmentations)
        else:
            df_train_lensed_aug = df_train_lensed
        
        # Combine augmented lensed with unlensed
        df_train_fold_aug = pd.concat([df_train_lensed_aug, df_train_unlensed], ignore_index=True)
        df_train_fold_aug = df_train_fold_aug.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        print(f"Augmented training set: {len(df_train_fold_aug)} samples")
        
        # Extract features from augmented training data
        X_train_aug, y_train_aug = prepare_features_from_df(df_train_fold_aug, feature_cols)
        
        # Extract features from test data (no augmentation)
        X_test_fold, y_test_fold = prepare_features_from_df(df_test_fold, feature_cols)
        
        # Calculate scale_pos_weight for this fold
        scale_pos_weight_fold = (y_train_aug == 0).sum() / (y_train_aug == 1).sum()
        
        # Hyperparameter tuning on this fold's training data
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight_fold,
            random_state=seed,
            eval_metric='auc'
        )
        
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=inner_cv,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_aug, y_train_aug)
        best_model_fold = grid_search.best_estimator_
        
        # Evaluate on ORIGINAL (non-augmented) test set
        y_test_pred_proba = best_model_fold.predict_proba(X_test_fold)[:, 1]
        fold_auc = roc_auc_score(y_test_fold, y_test_pred_proba)
        cv_scores.append(fold_auc)
        
        print(f"Fold {fold} AUC: {fold_auc:.4f}")
        
        # Store feature importance
        fold_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model_fold.feature_importances_,
            'fold': fold
        })
        all_feature_importances.append(fold_importance)
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"Fold AUCs: {[f'{s:.4f}' for s in cv_scores]}")
    
    # Average feature importance across folds
    all_importance_df = pd.concat(all_feature_importances)
    avg_feature_importance = all_importance_df.groupby('feature')['importance'].mean().reset_index()
    avg_feature_importance = avg_feature_importance.sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Features (averaged across folds):")
    print(avg_feature_importance.head(10))
    
    # Train final model on ALL data (with augmentation)
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL ON ALL DATA")
    print(f"{'='*60}")
    
    # Separate lensed and unlensed from full clean dataset
    df_lensed_all = df_clean[df_clean.lensed == True].copy()
    df_unlensed_all = df_clean[df_clean.lensed != True].copy()
    
    # Augment all lensed data
    if len(df_lensed_all) > 0 and n_augmentations > 0:
        print(f"Augmenting {len(df_lensed_all)} lensed samples for final model...")
        df_lensed_all_aug = augment_lensed_from_raw(df_lensed_all, n_augmentations)
    else:
        df_lensed_all_aug = df_lensed_all
    
    # Combine
    df_all_aug = pd.concat([df_lensed_all_aug, df_unlensed_all], ignore_index=True)
    df_all_aug = df_all_aug.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Extract features
    X_all, y_all = prepare_features_from_df(df_all_aug, feature_cols)
    
    print(f"Final training set: {len(X_all)} samples")
    
    scale_pos_weight_all = (y_all == 0).sum() / (y_all == 1).sum()
    
    # Train final model with grid search
    xgb_model_final = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight_all,
        random_state=seed,
        eval_metric='auc'
    )
    
    grid_search_final = GridSearchCV(
        estimator=xgb_model_final,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        n_jobs=-1,
        verbose=1
    )
    
    grid_search_final.fit(X_all, y_all)
    best_model_final = grid_search_final.best_estimator_
    best_params = grid_search_final.best_params_
    
    print(f"\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Final feature importance
    final_feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model_final.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return best_model_final, best_params, cv_scores, final_feature_importance, X_all, y_all


def evaluate_model(model, X, y, feature_cols):
    """
    Evaluate model performance and feature importance
    """
    
    # Convert y to numpy array of integers to avoid type issues
    y = np.array(y).astype(int)
    
    # Cross-validation evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]  # Changed from .iloc to direct indexing
        
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        cv_scores.append(roc_auc_score(y_val, y_pred_proba))
    
    print(f"Cross-validation AUC: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return cv_scores, feature_importance


def save_model_and_metrics(best_model, best_params, cv_scores, feature_importance, 
                          X, y, output_dir='./model_outputs'):
    """Save model, parameters, and all evaluation metrics"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    y = np.array(y).astype(int)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Save model
    model_path = f'{output_dir}/xgboost_lens_classifier_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nModel saved to: {model_path}")
    
    # 2. Save hyperparameters
    params_path = f'{output_dir}/best_params_{timestamp}.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Parameters saved to: {params_path}")
    
    # 3. Save feature importance
    feature_importance_path = f'{output_dir}/feature_importance_{timestamp}.csv'
    feature_importance.to_csv(feature_importance_path, index=False)
    print(f"Feature importance saved to: {feature_importance_path}")
    
    # 4. Save CV scores
    cv_metrics = {
        'cv_auc_scores': [float(score) for score in cv_scores],
        'mean_auc': float(np.mean(cv_scores)),
        'std_auc': float(np.std(cv_scores)),
        'timestamp': timestamp
    }
    cv_path = f'{output_dir}/cv_metrics_{timestamp}.json'
    with open(cv_path, 'w') as f:
        json.dump(cv_metrics, f, indent=4)
    print(f"CV metrics saved to: {cv_path}")
    
    # 5. ROC curve
    y_pred_proba = best_model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {np.mean(cv_scores):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/roc_curve_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/precision_recall_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Confusion matrix
    y_pred = best_model.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    plt.figure(figsize=(8, 6))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot_labels = np.array([[f'{count}\n({pct:.1f}%)' 
                             for count, pct in zip(row_counts, row_pcts)]
                            for row_counts, row_pcts in zip(cm, cm_percent)])
    
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', 
                xticklabels=['Unlensed', 'Lensed'],
                yticklabels=['Unlensed', 'Lensed'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(f'{output_dir}/confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Feature importance plot
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 15 Features')
    plt.gca().invert_yaxis()
    plt.savefig(f'{output_dir}/feature_importance_plot_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All outputs saved with timestamp: {timestamp}")
    
    return timestamp

def load_model(model_path):
    """
    Load a saved model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def main(df, n_augmentations=3):
    """
    Main training pipeline with proper cross-validation
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset with raw flux columns
    n_augmentations : int
        Number of augmented copies per lensed galaxy
    """
    print("="*80)
    print("GRAVITATIONAL LENS CLASSIFICATION")
    print("="*80)
    
    # Train with proper CV (augments only training folds from raw fluxes)
    best_model, best_params, cv_scores, feature_importance, X_all, y_all = \
        train_xgboost_model_with_cv(df, n_augmentations)
    
    # Save everything
    print("\nSaving outputs...")
    timestamp = save_model_and_metrics(
        best_model, best_params, cv_scores, feature_importance, X_all, y_all
    )
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE!")
    print(f"CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"Timestamp: {timestamp}")
    print(f"{'='*80}")
    
    return best_model, best_params, cv_scores, feature_importance, timestamp

if __name__ == "__main__":
    # Load your data
    engine = sa.create_engine(util.conn_string, poolclass=NullPool)
    with engine.connect() as conn:
        gal_tbl = sa.Table('lrg_train', sa.MetaData(), autoload_with=engine)
        stmt = sa.select(gal_tbl)
        df = pd.DataFrame(conn.execute(stmt))
    
    # Run training
    model, params, scores, importance, timestamp = main(df, n_augmentations=3)

# Usage example:
# X, y, scale_pos_weight, feature_cols = prepare_xgboost_model(your_df)
# best_model, best_params = train_xgboost_model(X, y, scale_pos_weight)
# cv_scores, feature_importance = evaluate_model(best_model, X, y, feature_cols)