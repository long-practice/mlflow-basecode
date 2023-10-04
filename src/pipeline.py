import os
import pickle

import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from preprocessing.preprocess import Preprocess
from utils.model_tuning import Objective
from utils.logger import set_logger


class Pipeline():
    def __init__(self, train_data, test_data):
        self.train = train_data
        self.test = test_data
        self.model = None
        self.logger = set_logger('Model_Tuning')

    def preprocess(self):
        self.preprocess = Preprocess(self.train, self.test)
        self.preprocess.do_preprocess()

    def get_model(self):
        mdl = None
        with open('.....', 'rb') as f:
            mdl = pickle.load(f)
        return mdl

    def model_tuning(self, model_name):
        # if model_name not in self.valid_regressor_models:
        #     print(f'{model_name} is not included in {self.valid_models}')
        #     print('Check the working category (Regression or Classification)')
        #     return None

        if model_name == 'XGBoost':
            self.logger.info('Start XGBoost Model Hyperparameter Tuning')
            xgb_mdl = XGBRegressor()
            X, y = self.train.drop('Strength', axis=1), self.train['Strength']
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
            print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
            err_func = lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)
            xgb_obj = Objective(xgb_mdl, model_name, X_train, X_valid, y_train, y_valid, err_func, 'minimize', 10, self.logger)
            xgb_obj.study()


    def run(self):
        self.preprocess()
        self.model_tuning('XGBoost')

        # self.model = self.get_model()