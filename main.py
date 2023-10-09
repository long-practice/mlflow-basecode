import os

import pandas as pd
from src.pipeline import Pipeline

if __name__ == '__main__':
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')
    n_trial = 10
    test_size = 0.2

    main_Pipeline = Pipeline(
        train_data=train_data,
        test_data=test_data,
        n_trial=n_trial,
        test_size=test_size
    )

    main_Pipeline.run()