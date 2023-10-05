import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
    precision_score,
    recall_score,
)


def get_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def get_rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def get_roc_auc_score(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def get_precision_score(y_true, y_pred):
    return precision_score(y_true, y_pred)


def get_recall_score(y_true, y_pred):
    return recall_score(y_true, y_pred)


def get_custom_error(y_true, y_pred):
    pass

    ### For Example
    ### eps = 1e-6
    ### numerator = (np.log(y_true * y_pred) / np.log(1.2)) ** 2
    ### denominator = np.sum(y_pred) + eps
    ### return numerator / denominator
