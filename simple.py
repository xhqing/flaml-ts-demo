import numpy as np
from flaml import AutoML

X_train = np.arange('2014-01', '2022-01', dtype='datetime64[M]')
y_train = np.random.random(size=84)
automl = AutoML()
automl.fit(X_train=X_train[:84],  # a single column of timestamp
           y_train=y_train,  # value for each timestamp
           period=12,  # time horizon to forecast, e.g., 12 months
           task='ts_forecast', time_budget=15,  # time budget in seconds
           log_file_name="ts_forecast.log",
           eval_method="holdout",
          )
print(automl.predict(X_train[84:]))
