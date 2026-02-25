### RandomForest.py : A file for running random forest on our data samples
### Author : Emily Ramey
### Date : 5/6/22

### Imports
import pandas as pd
import numpy as np
from util import clean_and_calc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
# from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import pickle
import util
import sys
from datetime import date, datetime

### Files and variables
lens_file = "../data/dr9_training/dr9_lensed.csv"
unlens_file = "../data/dr9_training/dr9_unlensed.csv"
save_file = f"gridsearch_models/gridsearch_{date.today().isoformat()}.sav"

bands = ['g', 'r', 'z', 'w1', 'w2']
theta_labels = ['massmet_1', 'massmet_2', 'dust2', 'tau', 'logtmax']
use_cols = util.use_cols
seed = 42

### Parameters for Grid Search
parameters = {
    'n_estimators': np.arange(50, 550, 50),
    'max_depth': np.arange(3, 21, 2),
    'max_samples': np.arange(0.7, 1.1, .1),
    'criterion': ('gini', 'entropy'),
    'max_features': ('sqrt', 'log2'),
    'random_state': [seed],
    'class_weight': ['balanced', 'balanced_subsample'],
}

### Main function
if __name__ == '__main__':
    start = datetime.now()
    # Read in data
    lensed = util.read_table("lensed_augmented")
    unlensed = util.read_table("unlensed")
    print(f"Data set:\n Lensed: {len(lensed)}\n Unlensed: {len(unlensed)}")
    
    # Set lensed and unlensed status (just in case)
    lensed['lensed'] = True
    unlensed['lensed'] = False
    
    # Filter
    lensed = clean_and_calc(lensed, filter_cols=use_cols)
    unlensed = clean_and_calc(unlensed, filter_cols=use_cols)
    
    print(f"After filtering:\n Lensed: {len(lensed)}\n Unlensed: {len(unlensed)}")
    
    # Concatenate lensed and unlensed data, mix up
    all_data = pd.concat([lensed, unlensed])
    all_data = all_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Split into training and testing
    X, y = all_data[use_cols], all_data['lensed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed)
    
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    
    # Grid search
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, parameters, scoring='roc_auc', cv=cv, n_jobs=10)
    clf.fit(X_train, y_train)
    
    end = datetime.now()
    
    # Predict
    # y_pr = clf.decision_function(X_test)
    preds = clf.predict(X_test)
    
    # Save model to file
    with open(save_file, 'wb') as file:
        pickle.dump(clf, file)
    
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    
    with open(f"gridsearch_models/gridsearch_{date.today().isoformat()}.out", 'w+') as file:
        file.write(f"Confusion matrix (test set): {tn} (TN), {fp} (FP)\n{'\t'*7} {fn} (FN), {tp} (TP)\n")
        file.write(f"Accuracy (test set): {accuracy_score(y_test, preds)}\n")
        file.write(f"Time to train: {str(end-start)}\n")
        file.write(f"Created file {save_file}")