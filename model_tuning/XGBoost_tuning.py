import optuna
from xgboost import XGBRegressor

from sklearn.preprocessing import train_test_split
from sklearn.metrics import mean_squared_error

def objective():
    params = {
        'scale_weight_pos': optuna.suggest.float(),
        'gamma': optuna.sugget.float()
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred)
    return rmse