import os
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from preprocessing.preprocess import Preprocess
from utils.logger import set_logger
from utils.model_tuning import Objective


class Pipeline:
    def __init__(self, train_data, test_data, n_trial, test_size, use_mlflow):
        self.train = train_data
        self.test = test_data

        self.n_trial = n_trial
        self.test_size = test_size

        self.model = None
        self.logger = set_logger("Model_Tuning")
        self.use_mlflow = use_mlflow

    def preprocess(self):
        self.preprocess = Preprocess(self.train, self.test)
        self.preprocess.do_preprocess()

    def model_tuning(self, model_name):
        if model_name == "XGBoost":
            self.logger.info("Start XGBoost Model Hyperparameter Tuning")
            xgb_mdl = XGBRegressor()
            X, y = self.train.drop("Strength", axis=1), self.train["Strength"]
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=self.test_size, random_state=42
            )
            err_func = lambda y_true, y_pred: mean_squared_error(
                y_true, y_pred, squared=False
            )
            xgb_obj = Objective(
                xgb_mdl,
                model_name,
                X_train,
                X_valid,
                y_train,
                y_valid,
                err_func,
                "minimize",
                self.n_trial,
                self.logger,
                self.use_mlflow
            )
            xgb_obj.study()

        if model_name == "LightGBM":
            self.logger.info("Start XGBoost Model Hyperparameter Tuning")
            lgbm_mdl = LGBMRegressor()
            X, y = self.train.drop("Strength", axis=1), self.train["Strength"]
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=self.test_size, random_state=42
            )
            err_func = lambda y_true, y_pred: mean_squared_error(
                y_true, y_pred, squared=False
            )
            lgbm_obj = Objective(
                lgbm_mdl,
                model_name,
                X_train,
                X_valid,
                y_train,
                y_valid,
                err_func,
                "minimize",
                self.n_trial,
                self.logger,
                self.use_mlflow
            )
            lgbm_obj.study()

    def run(self):
        self.preprocess()
        self.model_tuning("XGBoost")
        self.model_tuning("LightGBM")

