import optuna
from xgboost import XGBRegressor, XGBClassifier

# from sklearn.preprocessing import train_test_split
from sklearn.metrics import mean_squared_error

# from utils.error_collection import Collection


class objective():
    def __init__(self, model, params, X_train, X_test, y_train, y_test, task, error_function, direction, n_trials):
        self.model = model
        self.params = params

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.task = task
        self.error = error_function
        self.direction = direction
        self.n_trials = n_trials

    def __call__(self, trial):
        model = None
        if self.task == 'Regression':
            model = XGBRegressor(**self.params)
        elif self.task == 'Classification':
            model = XGBClassifier(**self.params)
        else:
            print('Wrong Task')

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        err = self.error(self.y_test, y_pred)
        return err

    def study(self):
        study = optuna.create_study(direction=self.direction)
        study.optimize(self.objective, n_trials=100)
