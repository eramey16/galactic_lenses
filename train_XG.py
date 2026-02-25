import pandas as pd
import numpy as np
import db_util as util

import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from docker import db_util as util
from XGBoost_model_gridsearch import best_params, prepare_xgboost_model, \
    train_with_best_params, evaluate_model, save_model_and_metrics

engine = sa.create_engine(util.conn_string, poolclass=NullPool)
with engine.connect() as conn:
    gal_tbl = sa.Table('lrg_train', sa.MetaData(), autoload_with=engine)
    stmt = sa.select(gal_tbl)
    gal_data = pd.DataFrame(conn.execute(stmt))

X, y, scale_pos_weight, feature_cols = prepare_xgboost_model(gal_data)
best_model, best_params = train_with_best_params(X, y, scale_pos_weight, best_params)
cv_scores, feature_importance = evaluate_model(best_model, X, y, feature_cols)
timestamp = save_model_and_metrics(best_model, best_params, cv_scores, 
                                   feature_importance, X, y)