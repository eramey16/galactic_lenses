sweep_config_XGB = {
    'method': 'bayes',  # or 'random', 'grid'
    'metric': {
        'name': 'cv_mean_roc_auc',  # or 'cv_mean_roc_auc'
        'goal': 'maximize'
    },
    'parameters': {
        'tree_method': {'values': ['hist']},
        'device': {'values':['cuda']},
        'objective': {'values':['binary:logistic']},
        # 'scale_pos_weight': {
        #     'distribution': 'log_uniform_values',
        #     'min': 10,
        #     'max': 1000,
        # },
        'n_jobs': {'values':[-1]},
        'seed': {'values':[42]},
        'eval_metric': {'values':['auc']},
        'n_aug': {'values': [1, 3, 5]},
        'max_depth': {'values': [3, 4, 5, 6, 7]},
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 0.01,
            'max': 0.3
        },
        'n_estimators': {'values': [100, 200, 300, 400]},
        'subsample': {
            'distribution': 'uniform',
            'min': 0.6,
            'max': 1.0
        },
        'colsample_bytree': {
            'distribution': 'uniform',
            'min': 0.6,
            'max': 1.0
        },
        'reg_alpha': {'values': [0, 0.1, 0.5, 1.0]},
        'reg_lambda': {'values': [1, 1.5, 2, 3]},
    }
}

sweep_config_RF = {
    'method': 'bayes',  # or 'random', 'grid'
    'metric': {
        'name': 'cv_mean_roc_auc',  # or 'cv_mean_roc_auc'
        'goal': 'maximize',
    },
    'parameters': {
        'n_aug': {'values': [1, 3, 5]},
        'n_estimators': {
            'distribution': 'int_uniform',
            'min': 50,
            'max': 700,
        },
        'max_depth': {
            'distribution': 'int_uniform',
            'min': 3,
            'max': 21,
        },
        'max_samples': {
            'distribution': 'uniform',
            'min': 0.6,
            'max': 1.0,
        },
        'n_jobs': {'value': -1},
        'criterion': {'values': ['gini', 'entropy']},
        'max_features': {'values': ['sqrt', 'log2']},
        'random_state': {'value': 42},
        'class_weight': {'value': 'balanced'}
    }
}

best_config_XGB = {}
best_config_RF = {}