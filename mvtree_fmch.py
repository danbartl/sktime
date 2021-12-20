# -*- coding: utf-8 -*-
"""Example application of global forecasting."""

import pandas as pd
from category_encoders.ordinal import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

from helpers import MVTimeSeriesSplit
from mvtree import MVTreeFeatureExtractor, find_maxlag
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
)

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)

# %%
# data= pd.read_csv("daten.jay")
data = pd.read_pickle("daten.jay")
# data= pd.read_pickle("m5_forecasting/daten.csv")
data.rename(columns={"value": "y", "ds": "Period"}, inplace=True)

data["Period"] = pd.to_datetime(data["Period"], format="%Y-%m-%d")
data.sort_values(["ts_id", "Period"], inplace=True)


# %%
encode_cols = [
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap",
    "no_stock_days",
]

id_encoder = OrdinalEncoder()
for i in encode_cols:
    id_encoder.fit(data.loc[:, [i]])
    data[i] = id_encoder.transform(data.loc[:, [i]])

# %%

model_params1 = {
    "max_bin": 127,
    "bin_construct_sample_cnt": 20000000,
    "num_leaves": 2 ** 10 - 1,
    "min_data_in_leaf": 2 ** 10 - 1,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "boost_from_average": False,
}

model_params2 = {
    "max_bin": 127,
    "bin_construct_sample_cnt": 20000000,
    "num_leaves": 60,
    "min_data_in_leaf": 20,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "boost_from_average": False,
}

# %%


def mvts_cv(X, y):
    """Split dataset based on tsids column."""
    for train_index, test_index in tscv.split(
        X, y, groups=X.index.get_level_values("ts_id")
    ):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        return X_train, X_test, y_train, y_test


# %%
data = data[data["ts_id"].isin(data["ts_id"].unique()[:5])]
data.sort_values(["ts_id", "Period"], inplace=True)
data = data.groupby("ts_id").tail(40)
# %%

dateline = pd.to_datetime(data["Period"].unique())
dateline.freq = "D"

tsids = data["ts_id"].unique()


# %%
mi = pd.MultiIndex.from_product([tsids, dateline], names=["ts_id", "Period"])


y = pd.DataFrame(data["y"].values, index=mi, columns=["y"])
X = pd.DataFrame(
    data.drop(["y", "ts_id", "Period"], axis=1).values,
    index=mi,
    columns=data.drop(["y", "ts_id", "Period"], axis=1).columns,
)

# %%

tscv = MVTimeSeriesSplit(n_splits=3, test_size=5)

X_train, X_test, y_train, y_test = mvts_cv(X, y)


# %%
kwargs = {
    "functions": {
        "lag": {"func": "lag", "window": [[1, 1], [5, 1], [7, 1]]},
        "mean": {"func": "mean", "window": [[1, 2], [2, 2], [4, 2]]},
        "median": {"func": "median", "window": [[1, 2], [2, 2], [4, 2]]},
        "std": {"func": "std", "window": [[1, 3], [2, 3]]},
    }
}


window_length = find_maxlag(kwargs)

# %%

regressor = make_pipeline(
    RandomForestRegressor(),
)

forecaster = make_reduction(
    regressor,
    scitype="tabular-regressor",
    transformers=[MVTreeFeatureExtractor(**kwargs)],
    window_length=window_length,
)

forecaster.fit(X=X_train, y=y_train, fh=2)

y_pred = forecaster.predict(X=X_test, fh=2)

# %%
# Window length is actually not relevant here
# Need to figure out how to change dictionary

kwargs_cv = {
    "functions": {
        "lag": {"func": "lag", "window": [[1, 1], [3, 1], [5, 1]]},
        "mean": {"func": "mean", "window": [[1, 2], [2, 2], [4, 2]]},
        "median": {"func": "median", "window": [[1, 2], [2, 2], [4, 2]]},
        "std": {"func": "std", "window": [[1, 3], [2, 3]]},
    }
}

forecaster_param_grid = {
    "window_length": [8],
    "transformers": [
        MVTreeFeatureExtractor(**kwargs),
        MVTreeFeatureExtractor(**kwargs_cv),
    ],
}

y_length = len(
    y_train.xs(y_train.index.get_level_values("ts_id")[0], level="ts_id").index
)
cv = SlidingWindowSplitter(
    initial_window=int(y_length * 0.8), window_length=window_length
)
gscv = ForecastingGridSearchCV(forecaster, cv=cv, param_grid=forecaster_param_grid)

# %%

gscv.fit(X=X_train, y=y_train, fh=2)
y_pred_cv = gscv.predict(X=X_test, fh=2)

# mean_absolute_percentage_error(y_pred, y_test)

gscv.best_params_
# print(gscv.best_params_)
# print(y_pred_cv)
