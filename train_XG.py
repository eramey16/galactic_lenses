import pandas as pd
import numpy as np
import db_util as util
import argparse

import sqlalchemy as sa
from sqlalchemy.pool import NullPool
from XGBoost_model_gridsearch import (
    xgb_best_params, rf_best_params,
    prepare_xgboost_model,
    train_model_gridsearch,
    train_with_best_params,
    evaluate_model,
    save_model_and_metrics
)

def main(model_type='XGB', use_grid_search=True):
    
    # Load data
    engine = sa.create_engine(util.conn_string, poolclass=NullPool)
    with engine.connect() as conn:
        gal_tbl = sa.Table('lrg_train', sa.MetaData(), autoload_with=engine)
        stmt = sa.select(gal_tbl)
        gal_data = pd.DataFrame(conn.execute(stmt))
    
    # Prepare features
    X, y, scale_pos_weight, feature_cols = prepare_model(gal_data)
    
    # Train model
    if use_grid_search:
        print(f"\nRunning grid search for {model_type}...")
        best_model, best_params = train_model_gridsearch(
            X, y, scale_pos_weight, model_type=model_type
        )
    else:
        print(f"\nTraining {model_type} with best known params...")
        best_params = xgb_best_params if model_type == 'XGB' else rf_best_params
        best_model, best_params = train_with_best_params(
            X, y, scale_pos_weight, best_params, model_type=model_type
        )
    
    # Evaluate
    cv_scores, feature_importance = evaluate_model(best_model, X, y, feature_cols)
    
    # Save
    output_dir = f'./model_outputs/{model_type.lower()}'
    timestamp = save_model_and_metrics(
        best_model, best_params, cv_scores, feature_importance, X, y,
        output_dir=output_dir
    )
    
    print(f"\nDone! Results saved to {output_dir}")
    return best_model, best_params, cv_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', '-m', default='XGB', 
                        choices=['XGB', 'RF'],
                        help='Model type to train')
    parser.add_argument('--grid_search', '-g', action='store_true',
                        help='Run grid search instead of using best known params')
    args = parser.parse_args()
    
    main(model_type=args.model_type, use_grid_search=args.grid_search)