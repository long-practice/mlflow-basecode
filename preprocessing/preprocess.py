import datetime

import numpy as np
import pandas as pd

from utils.logger import set_logger


class Preprocess:
    def __init__(self, train_df, test_df):
        self.train = train_df
        self.test = test_df

        self.logger = set_logger("preprocess")
        self.eps = 1e-6

    def set_missing_value(self, df):
        self.logger.info("Preprocessing Set Missing Value")
        pass

    def add_feature(self, df, category):
        self.logger.info("Preprocessing Add Feature")
        if (category == "train") or (category == "test"):
            # lnAgeInDays : AgeInDays 로그
            new_col, col = "lnAgeInDays", "AgeInDays"
            df[new_col] = np.log(df[col] + 1)
            if np.sum(df[new_col].isna()):
                self.logger.info(f"Invalid LogTransform {new_col} in train")

            # CementToWaterRatio : CementComponent / WaterComponent
            new_col, col1, col2 = (
                "CementToWaterRatio",
                "CementComponent",
                "WaterComponent",
            )
            df[new_col] = df[col1] / (df[col2] + self.eps)

            # FlyAshComponent : FlyAshComponent가 0, 0 아니면 1 (정수형)
            new_col, col = "FlyAshComponent_YN", "FlyAshComponent"
            df[new_col] = np.where(df[col] == 0.0, 0, 1).astype("int64")

    def remove_outlier(self, df):
        self.logger.info("Preprocessing Remove Outlier")
        cols = df.select_dtypes([np.int64, np.float64]).columns.values.tolist()
        factor = 1.5
        for col in cols:
            self.logger.info(f"column name is : {col}")
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (factor * IQR)
            upper_bound = Q3 + (factor * IQR)
            self.logger.info(f"lower_bound is : {lower_bound}")
            self.logger.info(f"upper_bound is : {upper_bound}")
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            self.logger.info("-" * 40)

    def do_preprocess(self):
        self.set_missing_value(self.train)
        self.set_missing_value(self.test)

        self.remove_outlier(self.train)
        self.remove_outlier(self.test)

        self.add_feature(self.train, "train")
        self.add_feature(self.test, "test")
