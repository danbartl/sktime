#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements functions to be used in evaluating forecasting models."""

__author__ = ["Martin Walter", "Markus Löning"]
__all__ = ["evaluate"]

import time

import numpy as np
import pandas as pd
from sklearn.base import clone

from sktime.datatypes._panel._convert import _get_time_index
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.validation.forecasting import (
    check_cv,
    check_fh,
    check_scoring,
    check_X,
)
from sktime.utils.validation.series import check_series


def evaluate(
    forecaster,
    cv,
    y,
    X=None,
    strategy="refit",
    scoring=None,
    fit_params=None,
    return_data=False,
):
    """Evaluate forecaster using timeseries cross-validation.

    Parameters
    ----------
    forecaster : sktime.forecaster
        Any forecaster
    cv : Temporal cross-validation splitter
        Splitter of how to split the data into test data and train data
    y : pd.Series
        Target time series to which to fit the forecaster.
    X : pd.DataFrame, default=None
        Exogenous variables
    strategy : {"refit", "update"}
        Must be "refit" or "update". The strategy defines whether the `forecaster` is
        only fitted on the first train window data and then updated, or always refitted.
    scoring : subclass of sktime.performance_metrics.BaseMetric, default=None.
        Used to get a score function that takes y_pred and y_test arguments
        and accept y_train as keyword argument.
        If None, then uses scoring = MeanAbsolutePercentageError(symmetric=True).
    fit_params : dict, default=None
        Parameters passed to the `fit` call of the forecaster.
    return_data : bool, default=False
        Returns three additional columns in the DataFrame, by default False.
        The cells of the columns contain each a pd.Series for y_train,
        y_pred, y_test.

    Returns
    -------
    pd.DataFrame
        DataFrame that contains several columns with information regarding each
        refit/update and prediction of the forecaster.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.model_evaluation import evaluate
    >>> from sktime.forecasting.model_selection import ExpandingWindowSplitter
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster(strategy="mean", sp=12)
    >>> cv = ExpandingWindowSplitter(
    ...     initial_window=24,
    ...     step_length=12,
    ...     fh=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> results = evaluate(forecaster=forecaster, y=y, cv=cv)
    """
    _check_strategy(strategy)
    cv = check_cv(cv, enforce_start_with_window=True)
    scoring = check_scoring(scoring)

    # danbartl: Checks are passed, but losing index col names
    if not isinstance(y.index, pd.MultiIndex):
        y = check_series(
            y,
            enforce_univariate=forecaster.get_tag("scitype:y") == "univariate",
            enforce_multivariate=forecaster.get_tag("scitype:y") == "multivariate",
        )
        X = check_X(X)

    fit_params = {} if fit_params is None else fit_params

    # Define score name.
    score_name = "test_" + scoring.name

    # Initialize dataframe.
    results = pd.DataFrame()

    # danbartl ts_index
    if isinstance(y.index, pd.MultiIndex):
        y_to_split = y.xs(y.index.get_level_values("ts_id")[0], level="ts_id")
    else:
        y_to_split = y
    # Run temporal cross-validation.

    for i, (train, test) in enumerate(cv.split(y_to_split)):
        # split data
        y_train, y_test, X_train, X_test = _split(y, X, train, test, cv.fh)

        # create forecasting horizon
        # danbartl:
        fh = ForecastingHorizon(_get_time_index(y_test), is_relative=False)

        # fit/update
        start_fit = time.perf_counter()
        if i == 0 or strategy == "refit":
            forecaster = clone(forecaster)
            forecaster.fit(y_train, X_train, fh=fh, **fit_params)

        else:  # if strategy == "update":
            forecaster.update(y_train, X_train)
        fit_time = time.perf_counter() - start_fit

        # predict
        start_pred = time.perf_counter()
        y_pred = forecaster.predict(fh, X=X_test)
        pred_time = time.perf_counter() - start_pred

        # score
        score = scoring(y_test, y_pred, y_train=y_train)

        # save results
        results = results.append(
            {
                score_name: score,
                "fit_time": fit_time,
                "pred_time": pred_time,
                "len_train_window": len(y_train),
                "cutoff": forecaster.cutoff,
                "y_train": y_train if return_data else np.nan,
                "y_test": y_test if return_data else np.nan,
                "y_pred": y_pred if return_data else np.nan,
            },
            ignore_index=True,
        )

    # post-processing of results
    if not return_data:
        results = results.drop(columns=["y_train", "y_test", "y_pred"])
    results["len_train_window"] = results["len_train_window"].astype(int)

    return results


def _split(y, X, train, test, fh):
    """Split y and X for given train and test set indices."""
    # danbartl insertion
    if isinstance(y.index, pd.MultiIndex):
        # y_train = y.groupby(level=0).nth(list(train))
        # y_test = y.groupby(level=0).nth(list(test))

        y_train = y.groupby(level=0).head(max(train))
        y_train = y_train.groupby(level=0).tail(max(train) - min(train) + 1)

        y_test = y.groupby(level=0).head(max(test))
        y_test = y_test.groupby(level=0).tail(1)

        X_train = X.groupby(level=0).head(max(train))
        X_train = X_train.groupby(level=0).tail(max(train) - min(train) + 1)

        X_test = X.groupby(level=0).head(max(test))
        X_test = X_test.groupby(level=0).tail(1)
    else:
        y_train = y.iloc[train]
        y_test = y.iloc[test]

        cutoff = y_train.index[-1]
        fh = check_fh(fh)
        fh = fh.to_relative(cutoff)
        if X is not None:
            X_train = X.iloc[train, :]

            # We need to expand test indices to a full range, since some forecasters
            # require the full range of exogenous values.
            test = np.arange(test[0] - fh.min(), test[-1]) + 1
            X_test = X.iloc[test, :]
        else:
            X_train = None
            X_test = None

    return y_train, y_test, X_train, X_test


def _check_strategy(strategy):
    """Assert strategy value.

    Parameters
    ----------
    strategy : str
        strategy of how to evaluate a forecaster

    Raises
    ------
    ValueError
        If strategy value is not in expected values, raise error.
    """
    valid_strategies = ("refit", "update")
    if strategy not in valid_strategies:
        raise ValueError(f"`strategy` must be one of {valid_strategies}")
