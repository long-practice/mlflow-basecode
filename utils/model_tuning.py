import optuna


class Objective():
    def __init__(self, model, params, X_train, X_test, y_train, y_test, error_function, direction, n_trials):
        self.model = model
        self.params = params

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.error = error_function
        self.direction = direction
        self.n_trials = n_trials

    def __call__(self, trial):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)

        err = self.error(self.y_test, y_pred)
        return err

    def study(self):
        study = optuna.create_study(direction=self.direction)
        study.optimize(self, n_trials=100)
