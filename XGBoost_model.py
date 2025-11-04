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

seed = 42

def flux2mag(flux):
    mags = 22.5 - 2.5*np.log10(flux)
    return mags

def augment_lensed_magnitudes(df, n_augmentations=3):
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
    lensed = df[df.lensed==True]
    
    augmented_df = lensed.copy()  # Start with original lenses
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
    df_combined = pd.concat([lensed.copy(), augmented_df], ignore_index=True)
    
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
    
def prepare_xgboost_model(df, target_col='lensed'):
    """
    Prepare XGBoost model - impute missing data without indicators
    """

    # Add missing features to un-bias sample
    unlensed_df, lensed_df = df[df.lensed!=True].copy(), df[df.lensed==True].copy()
    # unlensed_df = match_iband(unlensed_df, lensed_df)
    unlensed_df['lensed']=False

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
    feature_cols = color_features + chi_features + prosp_features
    
    # Prepare data
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
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

def train_xgboost_model(X, y, scale_pos_weight):
    """
    Train XGBoost with hyperparameter tuning
    """
    
    # Define parameter grid
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.05, 0.1, 0.15],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],  # L1 regularization
        'reg_lambda': [1, 1.5, 2]   # L2 regularization
    }
    
    # Base XGBoost model - REMOVED early_stopping_rounds
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc'
        # early_stopping_rounds removed - incompatible with GridSearchCV
    )
    
    # Stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit model
    y = np.array(y.astype(int))
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_

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
    """
    Save trained model, parameters, and performance metrics
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert y to integer numpy array to avoid type issues
    y = np.array(y).astype(int)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Save the model
    model_path = f'{output_dir}/xgboost_lens_classifier_{timestamp}.pkl'
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
    plt.title('ROC Curve - Lens Classification')
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
    plt.title('Precision-Recall Curve - Lens Classification')
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
    print(f"\nConfusion Matrix:")
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
    plt.text(0.5, -0.15, 
             'Top-left: True Negatives | Top-right: False Positives\n' +
             'Bottom-left: False Negatives | Bottom-right: True Positives',
             ha='center', transform=plt.gca().transAxes, fontsize=9, style='italic')
    
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
    plt.title('Top 15 Feature Importances')
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

def load_model(model_path):
    """
    Load a saved model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Usage example:
# X, y, scale_pos_weight, feature_cols = prepare_xgboost_model(your_df)
# best_model, best_params = train_xgboost_model(X, y, scale_pos_weight)
# cv_scores, feature_importance = evaluate_model(best_model, X, y, feature_cols)