import os
import pickle
import logging

import optuna
from utils.logger import get_logger_path
from utils.parameter import xgb_params_from_utils, lgbm_params_from_utils


class Objective():
    def __init__(self, model, model_name, X_train, X_test, y_train, y_test, error_function, direction, n_trials, logger):
        self.model = model
        self.model_name = model_name
        self.params = {}

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.error = error_function
        self.direction = direction
        self.n_trials = n_trials

        self.logger = logger
        self.logger.setLevel(logging.INFO)

        log_path = get_logger_path(self.logger)

        optuna.logging.enable_default_handler()
        optuna_log_file_handler = logging.FileHandler(log_path)
        optuna_log_file_handler.setFormatter(optuna.logging.create_default_formatter())
        optuna.logging.get_logger('optuna').addHandler(optuna_log_file_handler)
        optuna.logging.set_verbosity(optuna.logging.INFO)

    def get_params(self, trial):
        util_params = {}
        if self.model_name == 'XGBoost':
            util_params = xgb_params_from_utils.copy()
        elif self.model_name == 'LightGBM':
            util_params = lgbm_params_from_utils.copy()
        else:
            print('Implementing')
            pass

        params = {}
        for parameter, val_list in util_params.items():
            _type = val_list[-1]
            if _type == 'categorical':
                params[parameter] = trial.suggest_categorical(parameter, val_list[:-1])
            elif _type == 'int':
                params[parameter] = trial.suggest_int(parameter, val_list[0], val_list[1])
            elif _type == 'float':
                params[parameter] = trial.suggest_float(parameter, val_list[0], val_list[1])
            else:
                print('No params from utils.parameter.py')

        return params

    def __call__(self, trial):
        self.params = self.get_params(trial)

        self.model.set_params(**self.params)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)

        err = self.error(self.y_test, y_pred)
        return err

    def study(self):
        self.logger.info('Start Optimization')
        study = optuna.create_study(direction=self.direction)
        study.optimize(self, n_trials=self.n_trials)

        best_params = study.best_params
        self.model.set_params(**best_params)

        self.logger.info('Save Model')
        with open(f'./artifact/{self.model_name}.pkl', 'wb') as f:
            pickle.dump(self.model, f)
