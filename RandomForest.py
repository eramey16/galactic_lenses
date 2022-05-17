### RandomForest.py : A file for running random forest on our data samples
### Author : Emily Ramey
### Date : 5/6/22

### Imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
# from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import pickle

### Files and variables
lens_file = "../data/dr9_training/dr9_lensed.csv"
unlens_file = "../data/dr9_training/dr9_unlensed.csv"
save_file = "gridsearch_model.sav"
bands = ['g', 'r', 'z', 'w1', 'w2']
use_cols = ['g-r', 'r-z', 'r-w1', 'r-w2', 'abs_mag_r', 'z_phot_median']
use_cols.extend([f"rchisq_{band}" for band in bands])
seed = 42

### Parameters for Grid Search
parameters = {
    'n_estimators': np.arange(50, 550, 50),
    'max_depth': np.arange(3, 21, 2),
    'max_samples': np.arange(0.7, 1.1, .1),
    'criterion': ('gini', 'entropy'),
    'max_features': ('auto', None),
    'random_state': [seed],
    'class_weight': ['balanced'],
}

### Main function
if __name__ == '__main__':
    # Read in data
    lensed = pd.read_csv(lens_file)
    unlensed = pd.read_csv(unlens_file)
    
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
    clf = GridSearchCV(rf, parameters, scoring='roc_auc', cv=cv)
    clf.fit(X_train, y_train)
    
    # Predict
    # y_pr = clf.decision_function(X_test)
    preds = clf.predict(X_test)
    
    # # Summarize performance
    print("Confusion matrix (test set):")
    confusion_matrix(y_test, preds)
    print(f"Accuracy (test set): {accuracy_score(y_test, preds)}")
    
    # Save model to file
    with open(save_file, 'wb') as file:
        pickle.dump(clf, file)
    print(f"Created file: {save_file}")