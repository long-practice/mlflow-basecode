import os
import pickle

import numpy as np
import pandas as pd

from preprocessing.preprocess import Preprocess
from model_tuning.XGBoost_tuning import

class Pipeline():
    def __init__(self, train_data, test_data):
        self.train = train_data
        self.test = test_data
        self.model = None

        self.valid_regressor_models = ['XGBoost', 'LightGBM', 'CatBoost', 'Linear Regression', 'Random Forest']
        self.valid_classifier_models = ['XGBoost', 'LightGBM', 'CatBoost', 'Logistic Regression', 'Random Forest']

    def preprocess(self):
        self.preprocess = Preprocess(self.train, self.test)
        self.preprocess.do_preprocess()

    def get_model(self):
        mdl = None
        with open('.....', 'rb') as f:
            mdl = pickle.load(f)
        return mdl

    def model_tuning(self, model_name):
        if model_name not in self.valid_models:
            print(f'{model_name} is not included in {self.valid_models}')
            print('Check the working category (Regression or Classification)')
            return None

        if model_name == 'XGBoost':
            ## optuna tuning


    def run(self):
        self.preprocess()
        self.model_tuning('XGBoost')


        self.model = self.get_model()
