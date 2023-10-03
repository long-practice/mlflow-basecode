import os
import pickle
import logging

import optuna


class Objective():
    def __init__(self, model, params, X_train, X_test, y_train, y_test, error_function, direction, n_trials, logger):
        self.model = model

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.error = error_function
        self.direction = direction
        self.n_trials = n_trials

        self.logger = logger
        optuna.logging.enable_default_handler()
        optuna.logging.set_verbosity(optuna.logging.INFO)
        optuna.logging.enable_propagation()

    # def set_optuna_logger(self):
    #     optuna.logging.enable_propagation()
    #     optuna.logging.disable_default_handler()

    #     log_file_path = get_logger_path(self.logger)
    #     optuna_log_file_handler = logging.FileHandler(log_file_path)
    #     optuna_log_file_handler.setFormatter(optuna.logging.create_default_formatter())
    #     optuna.logging.get_logger('Optuna').addHandler(optuna_log_file_handler)
    #     optuna.logging.set_verbosity(optuna.logging.DEBUG)

    #     print(log_file_path)

    def __call__(self, trial):
        self.params =  {
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 500, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        }
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
        model_name = 'XGB_model'
        with open(f'./artifact/{model_name}.pkl', 'wb') as f:
            pickle.dump(self.model, f)
