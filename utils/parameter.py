xgb_params_from_utils = {
    'max_depth': [1, 20, 'int'],
    'learning_rate': [0.001, 1.0, 'float'],
    'n_estimators': [500, 1000, 'int'],
    'min_child_weight': [1, 10, 'int'],
    'gamma': [0.01, 1.0, 'float'],
    'subsample': [0.01, 1.0, 'float'],
    'colsample_bytree': [0.01, 1.0, 'float'],
    'reg_alpha': [0.01, 1.0, 'float'],
    'reg_lambda': [0.01, 1.0, 'float'],
}

lgbm_params_from_utils = {
    'verbosity': [-1, -1, 'int'],
    'boosting_type': ['gbdt', 'categorical'],
    'max_depth': [3, 10, 'int'],
    'num_leaves': [10, 500, 'int'],
    'min_child_samples': [5, 100, 'int'],
}

# params = {
#         "verbosity": -1,
#         "boosting_type": "gbdt",
#         "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.5),
#         "max_depth": trial.suggest_int("max_depth", 3, 7),
#         "num_leaves": trial.suggest_int("num_leaves", 10, 500),
#         "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
#         "subsample": trial.suggest_discrete_uniform("subsample", 0.1, 1, 0.01),
#         "colsample_bytree": trial.suggest_discrete_uniform("colsample_bytree", 0.1, 1, 0.01),
#         "n_estimators": trial.suggest_int("n_estimators", 50, 5000),
#         "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-6, 1e-1),
#         "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
#         "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
#     }