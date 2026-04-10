import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
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
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

seed = 42

def flux2mag(flux):
    mags = 22.5 - 2.5*np.log10(flux)
    return mags

def augment_lensed_magnitudes(lensed_df, n_augmentations=3):
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
    # lensed_df = lensed_df[lensed_df.lensed==True]
    
    augmented_df = lensed_df.copy()  # Start with original lenses
    # augmented_df['original_idx'] = lensed_df.index
    augmented_df = pd.concat([augmented_df.copy() for _ in range(n_augmentations)])

    for b in util.bands:
        augmented_df['flux_sigma_'+b] = 1 / np.sqrt(augmented_df['flux_ivar_'+b])
        new_fluxes = np.random.normal(loc=augmented_df['dered_flux_'+b], scale=augmented_df['flux_sigma_'+b])
        augmented_df['dered_mag_'+b] = flux2mag(new_fluxes)
    
    augmented_df['g_r'] = augmented_df['dered_mag_g'] - augmented_df['dered_mag_r']
    augmented_df['r_i'] = augmented_df['dered_mag_r'] - augmented_df['dered_mag_i']
    augmented_df['i_z'] = augmented_df['dered_mag_i'] - augmented_df['dered_mag_z']
    augmented_df['r_z'] = augmented_df['dered_mag_r'] - augmented_df['dered_mag_z']
    augmented_df['z_w1'] = augmented_df['dered_mag_z'] - augmented_df['dered_mag_w1']
    augmented_df['w1_w2'] = augmented_df['dered_mag_w1'] - augmented_df['dered_mag_w2']
    
    # Concat with original
    # df_combined = pd.concat([lensed.copy(), augmented_df], ignore_index=True)
    
    # print(f"Augmentation complete:")
    # print(f"  Original samples: {len(df)}")
    # print(f"  Augmented samples: {len(df_combined)}")
    # print(f"  Augmentation factor: {len(df_combined) / len(df):.1f}x")
    
    return augmented_df


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
    
def prepare_model(df, augment=True, n_augmentations=3):
    """
    Prepare XGBoost model - impute missing data without indicators
    """

    # Add missing features to un-bias sample
    df['original_idx'] = df.index
    df = add_prospector_features(df)
    unlensed_df, lensed_df = df[df.lensed!=True].copy(), df[df.lensed==True].copy()
    # unlensed_df = match_iband(unlensed_df, lensed_df)
    unlensed_df['lensed']=False

    if augment:
        augmented_df = augment_lensed_magnitudes(lensed_df, n_augmentations)

    df = pd.concat([lensed_df, unlensed_df]).sample(frac=1).reset_index(drop=True)
    
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
    feature_cols = color_features + chi_features + prosp_features
    
    # Prepare data
    X = df[feature_cols].copy()
    y = df['lensed'].copy()
    
    # Fix for column types
    X.columns = X.columns.astype(str)
    
    print(f"Missing values before imputation:")
    print(X.isnull().sum())
    X = X.rename(str, axis='columns')
    
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

    # Augment, if needed
    if augment:
        df[feature_cols] = X[feature_cols]
        lensed_df = df[df.lensed==True].copy()
        augmented_df = augment_lensed_magnitudes(lensed_df, n_augmentations)
        # augmented_X = augmented_df[feature_cols+['original_idx']]
    else:
        augmented_df = None

    return X, y, augmented_df, scale_pos_weight, feature_cols

# --- Keep all your existing functions unchanged ---
# flux2mag, augment_lensed_magnitudes, add_prospector_features, 
# prepare_xgboost_model, evaluate_model, save_model_and_metrics,
# load_model are all unchanged

def train_model_gridsearch(X, y, scale_pos_weight, model_type='XGB'):
    """
    Train XGBoost or RandomForest with grid search hyperparameter tuning.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target labels
    scale_pos_weight : float
        Class imbalance weight (unlensed/lensed ratio)
    model_type : str
        'XGB' or 'RF'
    
    Returns:
    --------
    best_estimator : trained model
    best_params : dict of best hyperparameters
    """
    
    y = np.array(y.astype(int))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    if model_type == 'XGB':
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            eval_metric='auc',
            n_jobs=-1,
        )
        param_grid = {
            'max_depth': [3, 4, 5, 6, None],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [100, 200, 300, 500, 1000],
            'subsample': [0.6, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 0.9],
            'reg_lambda': [.5, 1, 1.5, 2, 3],
        }
        
    elif model_type == 'RF':
        base_model = RandomForestClassifier(
            random_state=seed,
            class_weight={0: 1, 1: scale_pos_weight},  # Equivalent to scale_pos_weight
            n_jobs=-1,
        )
        param_grid = {
            'n_estimators': [100, 200, 300, 500, 800, 1000],
            'max_depth': [3, 5, 8, 10, None],       # None = fully grown trees
            'min_samples_split': [3, 5, 10],         # Min samples to split a node
            'min_samples_leaf': [2, 4, 6],           # Min samples at leaf node
            'max_features': ['sqrt', 'log2', 0.5],   # Features to consider per split
            'max_samples': [0.6, 0.7, 0.8, 0.9],    # Bootstrap sample size
            'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: scale_pos_weight}, None]
        }
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'XGB' or 'RF'.")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,  # Refit best model on full dataset
    )
    
    grid_search.fit(X, y)
    
    print(f"\nBest {model_type} parameters:")
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def train_with_best_params(X, y, scale_pos_weight, best_params, model_type='XGB'):
    """
    Train XGBoost or RandomForest with known best parameters, no grid search.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target labels
    scale_pos_weight : float
        Class imbalance weight
    best_params : dict
        Known best hyperparameters
    model_type : str
        'XGB' or 'RF'
    
    Returns:
    --------
    model : trained model
    best_params : dict
    """
    
    y = np.array(y.astype(int))
    
    if model_type == 'XGB':
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            eval_metric='auc',
            n_jobs=-1,
            **best_params
        )
    elif model_type == 'RF':
        model = RandomForestClassifier(
            random_state=seed,
            class_weight={0: 1, 1: scale_pos_weight},
            n_jobs=-1,
            **best_params
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'XGB' or 'RF'.")
    
    model.fit(X, y)
    
    return model, best_params


def evaluate_model(model, X, y, feature_cols, augmented_df=None):
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

        if augmented_df is not None:
            augmented_df = pd.concat([augmented_df[augmented_df.original_idx==i] 
                                  for i in train_idx])
            # augmented_df.drop(labels=['original_idx'], inplace=True)
            augmented_X = augmented_df[X_train.columns]
            augmented_y = np.array([True]*len(augmented_X)).reshape(-1)
            X_train = pd.concat([X_train, augmented_X]).reset_index(drop=True)
            y_train = np.concatenate([y_train, augmented_y])
        
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
                          X, y, output_dir='./model_outputs', title='RF'):
    """
    Save trained model, parameters, and performance metrics
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert y to integer numpy array to avoid type issues
    y = np.array(y).astype(int)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Save the model
    model_path = f'{output_dir}/lens_classifier_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Model saved to: {model_path}")
    
    # 2. Save hyperparameters
    params_path = f'{output_dir}/best_params_{timestamp}.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Parameters saved to: {params_path}")
    
    # 3. Save feature importance
    feature_importance_path = f'{output_dir}/feature_importance_{timestamp}.csv'
    feature_importance.to_csv(feature_importance_path, index=False)
    print(f"Feature importance saved to: {feature_importance_path}")
    
    # 4. Save cross-validation scores
    cv_metrics = {
        'cv_auc_scores': [float(score) for score in cv_scores],  # Convert to float for JSON
        'mean_auc': float(np.mean(cv_scores)),
        'std_auc': float(np.std(cv_scores)),
        'timestamp': timestamp
    }
    cv_path = f'{output_dir}/cv_metrics_{timestamp}.json'
    with open(cv_path, 'w') as f:
        json.dump(cv_metrics, f, indent=4)
    print(f"CV metrics saved to: {cv_path}")
    
    # 5. Generate and save ROC curve
    y_pred_proba = best_model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {np.mean(cv_scores):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title} Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    roc_path = f'{output_dir}/roc_curve_{timestamp}.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {roc_path}")
    
    # 6. Generate and save Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {title} Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    pr_path = f'{output_dir}/precision_recall_{timestamp}.png'
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curve saved to: {pr_path}")
    
    # 7. Generate and save confusion matrix - FIXED LABELS
    y_pred = best_model.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    # Print the confusion matrix values for debugging
    print(f"\n{title} Confusion Matrix:")
    print(f"                Predicted Unlensed  Predicted Lensed")
    print(f"Actual Unlensed:      {cm[0,0]:6d}           {cm[0,1]:6d}")
    print(f"Actual Lensed:        {cm[1,0]:6d}           {cm[1,1]:6d}")
    
    plt.figure(figsize=(8, 6))
    
    # Add percentage annotations
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot_labels = np.array([[f'{count}\n({pct:.1f}%)' 
                             for count, pct in zip(row_counts, row_pcts)]
                            for row_counts, row_pcts in zip(cm, cm_percent)])
    
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', 
                xticklabels=['Predicted: Unlensed', 'Predicted: Lensed'],
                yticklabels=['Actual: Unlensed', 'Actual: Lensed'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Confusion Matrix - Lens Classification')
    
    # Add text explanation
    # plt.text(0.5, -0.15, 
    #          'Top-left: True Negatives | Top-right: False Positives\n' +
    #          'Bottom-left: False Negatives | Bottom-right: True Positives',
    #          ha='center', transform=plt.gca().transAxes, fontsize=9, style='italic')
    
    cm_path = f'{output_dir}/confusion_matrix_{timestamp}.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Calculate and print key metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives (correctly identified unlensed):  {tn}")
    print(f"  False Positives (unlensed predicted as lensed):  {fp}")
    print(f"  False Negatives (lensed predicted as unlensed):  {fn}")
    print(f"  True Positives (correctly identified lensed):    {tp}")
    print(f"  Accuracy: {(tp + tn) / (tp + tn + fp + fn):.3f}")
    print(f"  Precision: {tp / (tp + fp):.3f}")
    print(f"  Recall: {tp / (tp + fn):.3f}")
    
    # 8. Generate and save classification report
    report = classification_report(y, y_pred, target_names=['Unlensed', 'Lensed'], 
                                   output_dict=True)
    report_path = f'{output_dir}/classification_report_{timestamp}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Classification report saved to: {report_path}")
    
    # 9. Plot and save feature importance
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top 15 Feature Importances - {title}')
    plt.gca().invert_yaxis()
    feat_imp_path = f'{output_dir}/feature_importance_plot_{timestamp}.png'
    plt.savefig(feat_imp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to: {feat_imp_path}")
    
    # 10. Save summary metadata
    metadata = {
        'timestamp': timestamp,
        'n_samples': len(X),
        'n_features': len(X.columns),
        'feature_names': list(X.columns),
        'class_distribution': {str(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()},
        'best_params': best_params,
        'cv_auc_mean': float(np.mean(cv_scores)),
        'cv_auc_std': float(np.std(cv_scores))
    }
    metadata_path = f'{output_dir}/model_metadata_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to: {metadata_path}")
    
    return timestamp

def load_model(output_dir, timestamp=None):
    """
    Load all saved files needed for evaluate_model.
    If timestamp is None, loads the most recent files in output_dir.
    
    Parameters:
    -----------
    output_dir : str
        Directory where model files are saved
    timestamp : str, optional
        Specific timestamp to load, e.g. '20240101_120000'
        If None, loads the most recent files
    
    Returns:
    --------
    model : trained model
    best_params : dict
    cv_scores : list
    feature_importance : DataFrame
    feature_cols : list
    metadata : dict
    """
    import os
    import glob
    
    if timestamp is None:
        # Find most recent model file
        model_files = glob.glob(os.path.join(output_dir, 'lens_classifier_*.pkl'))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {output_dir}")
        
        # Sort by timestamp in filename and take most recent
        model_path = sorted(model_files)[-1]
        timestamp = model_path.split('lens_classifier_')[1].replace('.pkl', '')
        print(f"Loading most recent model with timestamp: {timestamp}")
    
    # Build paths from timestamp
    model_path      = os.path.join(output_dir, f'lens_classifier_{timestamp}.pkl')
    params_path     = os.path.join(output_dir, f'best_params_{timestamp}.json')
    cv_path         = os.path.join(output_dir, f'cv_metrics_{timestamp}.json')
    feat_imp_path   = os.path.join(output_dir, f'feature_importance_{timestamp}.csv')
    metadata_path   = os.path.join(output_dir, f'model_metadata_{timestamp}.json')
    
    # Check all files exist before loading
    missing = [p for p in [model_path, params_path, cv_path, feat_imp_path, metadata_path]
               if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing files for timestamp {timestamp}:\n" + 
                                '\n'.join(missing))
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Loaded model from: {model_path}")
    
    # Load hyperparameters
    with open(params_path, 'r') as f:
        best_params = json.load(f)
    print(f"Loaded parameters from: {params_path}")
    
    # Load CV metrics
    with open(cv_path, 'r') as f:
        cv_metrics = json.load(f)
    cv_scores = cv_metrics['cv_auc_scores']
    print(f"Loaded CV metrics from: {cv_path}")
    
    # Load feature importance
    feature_importance = pd.read_csv(feat_imp_path)
    print(f"Loaded feature importance from: {feat_imp_path}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    feature_cols = metadata['feature_names']
    print(f"Loaded metadata from: {metadata_path}")
    
    print(f"\nModel summary:")
    print(f"  Timestamp: {timestamp}")
    print(f"  Features: {feature_cols}")
    print(f"  CV AUC: {cv_metrics['mean_auc']:.4f} ± {cv_metrics['std_auc']:.4f}")
    
    return model, best_params, cv_scores, feature_importance, feature_cols, metadata

# Best params for each model type
xgb_best_params = {
    "colsample_bytree": 1.0,
    "learning_rate": 0.05,
    "max_depth": 6,
    "n_estimators": 200,
    "reg_alpha": 0.1,
    "reg_lambda": 2,
    "subsample": 0.9
}

rf_best_params = {}  # Fill in after running grid search

# Usage example:
# X, y, scale_pos_weight, feature_cols = prepare_xgboost_model(your_df)
# best_model, best_params = train_xgboost_model(X, y, scale_pos_weight)
# cv_scores, feature_importance = evaluate_model(best_model, X, y, feature_cols)