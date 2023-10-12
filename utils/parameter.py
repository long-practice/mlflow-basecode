xgb_params_from_utils = {
    "max_depth": [1, 20, "int"],
    "learning_rate": [0.001, 1.0, "float"],
    "n_estimators": [500, 1000, "int"],
    "min_child_weight": [1, 10, "int"],
    "gamma": [0.01, 1.0, "float"],
    "subsample": [0.01, 1.0, "float"],
    "colsample_bytree": [0.01, 1.0, "float"],
    "reg_alpha": [0.01, 1.0, "float"],
    "reg_lambda": [0.01, 1.0, "float"],
}

lgbm_params_from_utils = {
    "verbosity": [-1, -1, "int"],
    # "boosting_type": ["gbdt", "categorical"],
    "max_depth": [3, 10, "int"],
    "num_leaves": [10, 500, "int"],
    "min_child_samples": [5, 100, "int"],
}