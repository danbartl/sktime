#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Composition functionality for reduction approaches to forecasting."""

__author__ = [
    "Ayushmaan Seth",
    "Kavin Anand",
    "Luis Zugasti",
    "Lovkush Agarwal",
    "Markus Löning",
]

__all__ = [
    "make_reduction",
    "DirectTimeSeriesRegressionForecaster",
    "RecursiveTimeSeriesRegressionForecaster",
    "MultioutputTimeSeriesRegressionForecaster",
    "DirectTabularRegressionForecaster",
    "RecursiveTabularRegressionForecaster",
    "MultioutputTabularRegressionForecaster",
    "DirRecTabularRegressionForecaster",
    "DirRecTimeSeriesRegressionForecaster",
]

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone

from sktime.datatypes._panel._convert import _get_time_index
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _BaseWindowForecaster
from sktime.regression.base import BaseRegressor
from sktime.utils.validation import check_window_length
from sktime.utils.datetime import _shift

#danbartl: for Base Class checks that are in the interim handled here
from warnings import warn
from sktime.datatypes import convert_to, mtype

def _concat_y_X(y, X):
    """Concatenate y and X prior to sliding-window transform."""
    z = y.to_numpy()
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if X is not None:
        z = np.column_stack([z, X.to_numpy()])
    return z


def _check_fh(fh):
    """Check fh prior to sliding-window transform."""
    assert fh.is_relative
    assert fh.is_all_out_of_sample()
    return fh.to_indexer().to_numpy()


def _sliding_window_transform(
    y, window_length, transformers, fh, X=None, scitype="tabular-regressor"
):
    """Transform time series data using sliding window.

    See `test_sliding_window_transform_explicit` in test_reduce.py for explicit
    example.

    Parameters
    ----------
    y : pd.Series
        Endogenous time series
    window_length : int
        Window length for transformed feature variables
    fh : ForecastingHorizon
        Forecasting horizon for transformed target variable
    X : pd.DataFrame, optional (default=None)
        Exogenous series.
    scitype : str {"tabular-regressor", "time-series-regressor"}, optional
        Scitype of estimator to use with transformed data.
        - If "tabular-regressor", returns X as tabular 2d array
        - If "time-series-regressor", returns X as panel 3d array

    Returns
    -------
    yt : np.ndarray, shape = (n_timepoints - window_length, 1)
        Transformed target variable.
    Xt : np.ndarray, shape = (n_timepoints - window_length, n_variables,
    window_length)
        Transformed lagged values of target variable and exogenous variables,
        excluding contemporaneous values.
    """
    # There are different ways to implement this transform. Pre-allocating an
    # array and filling it by iterating over the window length seems to be the most
    # efficient one.

    ts_index = _get_time_index(y)

    n_timepoints = ts_index.shape[0]
    window_length = check_window_length(window_length, n_timepoints)

    if isinstance(y.index, pd.MultiIndex):
        # danbartl: how to implement iteration over all transformers?
        if isinstance(transformers, list):
            tf_fit = transformers[0].fit()
            X_from_y = tf_fit.transform(y)
        else:
            tf_fit = transformers.fit()
            X_from_y = tf_fit.transform(y)

        X_from_y_cut = X_from_y.groupby(level=0).tail(
            n_timepoints - tf_fit._truncate_start + 1
        )
        #    X_from_y = LaggedWindowSummarizer(**model_kwargs,X)
        # fix maxlag to take lag into account
        X_cut = X.groupby(level=0).tail(n_timepoints - tf_fit._truncate_start + 1)

        z = pd.concat([X_from_y_cut, X_cut], axis=1)
        yt = z[["y"]]
        Xt = z.drop("y", axis=1)
    else:
        z = _concat_y_X(y, X)
        n_timepoints, n_variables = z.shape

        fh = _check_fh(fh)
        fh_max = fh[-1]

        if window_length + fh_max >= n_timepoints:
            raise ValueError(
                "The `window_length` and `fh` are incompatible with the length of `y`"
            )

        # Get the effective window length accounting for the forecasting horizon.
        effective_window_length = window_length + fh_max
        Zt = np.zeros(
            (
                n_timepoints + effective_window_length,
                n_variables,
                effective_window_length + 1,
            )
        )

        # Transform data.
        for k in range(effective_window_length + 1):
            i = effective_window_length - k
            j = n_timepoints + effective_window_length - k
            Zt[i:j, :, k] = z

        # Truncate data, selecting only full windows, discarding incomplete ones.
        Zt = Zt[effective_window_length:-effective_window_length]

        # Return transformed feature and target variables separately. This
        # excludes contemporaneous values of the exogenous variables. Including them
        # would lead to unequal-length data, with more time points for
        # exogenous series than the target series, which is currently not supported.
        yt = Zt[:, 0, window_length + fh]
        Xt = Zt[:, :, :window_length]
    # Pre-allocate array for sliding windows.
    # If the scitype is tabular regression, we have to convert X into a 2d array.
    if scitype == "tabular-regressor":
        if isinstance(y.index, pd.MultiIndex):
            return yt, Xt
        else:
            return yt, Xt.reshape(Xt.shape[0], -1)
    else:
        return yt, Xt


class _Reducer(_BaseWindowForecaster):
    """Base class for reducing forecasting to regression."""

    _required_parameters = ["estimator"]

    def __init__(self, estimator, window_length=10, transformers=None):
        super(_Reducer, self).__init__(
            window_length=window_length
        )
        self.transformers = transformers
        self.transformers_ = None
        self.estimator = estimator
        self._cv = None

    def fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets self._is_fitted flag to True.
            Writes self._y and self._X with `y` and `X`, respectively.
            Sets self.cutoff and self._cutoff to last index seen in `y`.
            Sets fitted model attributes ending in "_".
            Stores fh to self.fh if fh is passed.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                must have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                must have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
            if self.get_tag("requires-fh-in-fit"), must be passed, not optional
        X : pd.DataFrame, or 2D np.array, optional (default=None)
            Exogeneous time series to fit to
            if self.get_tag("X-y-must-have-same-index"), X.index must contain y.index

        Returns
        -------
        self : Reference to self.
        """
        # if fit is called, fitted state is re-set
        self._is_fitted = False

        self._set_fh(fh)

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # set internal X/y to the new X/y
        # this also updates cutoff from y
        self._update_y_X(y_inner, X_inner)

        # checks and conversions complete, pass to inner fit
        #####################################################

        self._fit(y=y_inner, X=X_inner, fh=fh)

        # this should happen last
        self._is_fitted = True

        return self

    def _get_last_window(self):
        """Select last window."""
        # Get the start and end points of the last window.
        cutoff = self.cutoff
        start = _shift(cutoff, by=-self.window_length_ + 1)

        if isinstance(self._y.index, pd.MultiIndex):

            # Get the last window of the endogenous variable.
            y = self._y.query("Period >= @start & Period <= @cutoff")

            # If X is given, also get the last window of the exogenous variables.
            X = (
                self._X.query("Period >= @start & Period <= @cutoff")
                if self._X is not None
                else None
            )

            X_from_y = self.transformers.fit().transform(y)

            X_from_y_cut = X_from_y.groupby(level=0).tail(1)
            #    X_from_y = LaggedWindowSummarizer(**model_kwargs,X)
            # fix maxlag to take lag into account
            X_cut = X.groupby(level=0).tail(1)

            z = pd.concat([X_from_y_cut, X_cut], axis=1)

            # X_cut = X.groupby(level=0).tail(n_timepoints-maxlag)
            X = z.drop("y", axis=1)
            y = y.groupby(level=0).tail(1)
        else:
            # Get the last window of the endogenous variable.
            y = self._y.loc[start:cutoff].to_numpy()

            # If X is given, also get the last window of the exogenous variables.
            X = self._X.loc[start:cutoff].to_numpy() if self._X is not None else None

        return y, X

    def _set_cutoff_from_y(self, y):
        """Set and update cutoff from series y.

        Parameters
        ----------
        y: pd.Series, pd.DataFrame, or np.array
            Time series from which to infer the cutoff.

        Notes
        -----
        Set self._cutoff to last index seen in `y`.
        """
        # y_mtype = mtype(y, as_scitype="Series")

        # danbartl: not sure why mtype inference does not work
        # y_mtype = mtype(y, as_scitype="Series")
        # danbartl: manual override
        y_mtype = "pd.DataFrame"

        ts_index = _get_time_index(y)

        if len(ts_index) > 0:
            if y_mtype in ["pd.Series", "pd.DataFrame"]:
                self._cutoff = ts_index[-1]
            elif y_mtype == "np.ndarray":
                self._cutoff = len(ts_index)
            else:
                raise TypeError("y does not have a supported type")

    def _update_y_X(self, y, X=None, enforce_index_type=None):
        """Update internal memory of seen training data.

        Accesses in self:
        _y : only if exists, then assumed same type as y and same cols
        _X : only if exists, then assumed same type as X and same cols
            these assumptions should be guaranteed by calls

        Writes to self:
        _y : same type as y - new rows from y are added to current _y
            if _y does not exist, stores y as _y
        _X : same type as X - new rows from X are added to current _X
            if _X does not exist, stores X as _X
            this is only done if X is not None
        cutoff : is set to latest index seen in y

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or nd.nparray (1D or 2D)
            Endogenous time series
        X : pd.DataFrame or 2D np.ndarray, optional (default=None)
            Exogeneous time series
        """
        # we only need to modify _y if y is not None
        if y is not None:
            # if _y does not exist yet, initialize it with y
            if not hasattr(self, "_y") or self._y is None or not self.is_fitted:
                self._y = y
            # otherwise, update _y with the new rows in y
            #  if y is np.ndarray, we assume all rows are new
            elif isinstance(y, np.ndarray):
                self._y = np.concatenate(self._y, y)
            #  if y is pandas, we use combine_first to update
            elif isinstance(y, (pd.Series, pd.DataFrame)) and len(y) > 0:
                self._y = y.combine_first(self._y)

            # set cutoff to the end of the observation horizon
            self._set_cutoff_from_y(y)

        # we only need to modify _X if X is not None
        if X is not None:
            # if _X does not exist yet, initialize it with X
            if not hasattr(self, "_X") or self._X is None or not self.is_fitted:
                self._X = X
            # otherwise, update _X with the new rows in X
            #  if X is np.ndarray, we assume all rows are new
            elif isinstance(X, np.ndarray):
                self._X = np.concatenate(self._X, X)
            #  if X is pandas, we use combine_first to update
            elif isinstance(X, (pd.Series, pd.DataFrame)) and len(X) > 0:
                self._X = X.combine_first(self._X)


    def _is_predictable(self, last_window):
        """Check if we can make predictions from last window."""
        return (
            len(last_window) == self.window_length_
            and np.sum(np.isnan(last_window)) == 0
            and np.sum(np.isinf(last_window)) == 0
        )

    def _predict_in_sample(self, fh, X=None, return_pred_int=False, alpha=None):
        # Note that we currently only support out-of-sample predictions. For the
        # direct and multioutput strategy, we need to check this already during fit,
        # as the fh is required for fitting.
        raise NotImplementedError(
            f"Generating in-sample predictions is not yet "
            f"implemented for {self.__class__.__name__}."
        )

    def predict(
        self,
        fh=None,
        X=None,
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
        keep_old_return_type=True,
    ):
        """Forecast time series at future horizon.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self.cutoff, self._is_fitted

        Writes to self:
            Stores fh to self.fh if fh is passed and has not been passed previously.

        Parameters
        ----------
        fh : int, list, np.ndarray or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, or 2D np.ndarray, optional (default=None)
            Exogeneous time series to predict from
            if self.get_tag("X-y-must-have-same-index"), X.index must contain fh.index
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Point forecasts at fh, with same index as fh
            y_pred has same type as y passed in fit (most recently)
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            in this case, return is 2-tuple (otherwise a single y_pred)
            Prediction intervals
        """
        # handle inputs

        self.check_is_fitted()
        self._set_fh(fh)

        # todo deprecate NotImplementedError in v 10.0.1
        if return_pred_int and not self.get_tag("capability:pred_int"):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have the capability to return "
                "prediction intervals. Please set return_pred_int=False. If you "
                "think this estimator should have the capability, please open "
                "an issue on sktime."
            )

        # input check and conversion for X
        X_inner = self._check_X(X=X)

        # this is how it is supposed to be after the refactor is complete and effective
        if not return_pred_int:
            y_pred = self._predict(
                self.fh,
                X=X_inner,
            )

            # convert to output mtype, identical with last y mtype seen
            if isinstance(y_pred.index, pd.MultiIndex):
                y_out = y_pred
            else:
                y_out = convert_to(
                    y_pred,
                    self._y_mtype_last_seen,
                    as_scitype="Series",
                    store=self._converter_store_y,
                )

            return y_out

        # keep following code for downward compatibility,
        # todo: can be deleted once refactor is completed and effective,
        # todo: deprecate in v 10
        else:
            warn(
                "return_pred_int in predict() will be deprecated;"
                "please use predict_interval() instead to generate "
                "prediction intervals.",
                FutureWarning,
            )

            if not self._has_predict_quantiles_been_refactored():
                # this means the method is not refactored
                y_pred = self._predict(
                    self.fh,
                    X=X_inner,
                    return_pred_int=return_pred_int,
                    alpha=alpha,
                )

                # returns old return type anyways
                pred_int = y_pred[1]
                y_pred = y_pred[0]

            else:
                # it's already refactored
                # opposite definition previously vs. now
                coverage = [1 - a for a in alpha]
                pred_int = self.predict_interval(fh=fh, X=X_inner, coverage=coverage)

                if keep_old_return_type:
                    pred_int = _convert_new_to_old_pred_int(pred_int, alpha)

            # convert to output mtype, identical with last y mtype seen
            y_out = convert_to(
                y_pred,
                self._y_mtype_last_seen,
                as_scitype="Series",
                store=self._converter_store_y,
            )

            return (y_out, pred_int)


    def _check_X_y(self, X=None, y=None):
        """Check and coerce X/y for fit/predict/update functions.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D), optional (default=None)
            Time series to check.
        X : pd.DataFrame, or 2D np.array, optional (default=None)
            Exogeneous time series.

        Returns
        -------
        y_inner : Series compatible with self.get_tag("y_inner_mtype") format
            converted/coerced version of y, mtype determined by "y_inner_mtype" tag
            None if y was None
        X_inner : Series compatible with self.get_tag("X_inner_mtype") format
            converted/coerced version of y, mtype determined by "X_inner_mtype" tag
            None if X was None

        Raises
        ------
        TypeError if y or X is not one of the permissible Series mtypes
        TypeError if y is not compatible with self.get_tag("scitype:y")
            if tag value is "univariate", y must be univariate
            if tag value is "multivariate", y must be bi- or higher-variate
            if tag vaule is "both", y can be either
        TypeError if self.get_tag("X-y-must-have-same-index") is True
            and the index set of X is not a super-set of the index set of y

        Writes to self
        --------------
        _y_mtype_last_seen : str, mtype of y
        _converter_store_y : dict, metadata from conversion for back-conversion
        """
        # input checks and minor coercions on X, y
        ###########################################

        enforce_univariate = self.get_tag("scitype:y") == "univariate"
        enforce_multivariate = self.get_tag("scitype:y") == "multivariate"
        enforce_index_type = self.get_tag("enforce_index_type")

        # checking y

        # danbartl: checks disabled

        if y is not None:
            if not isinstance(y.index, pd.MultiIndex):
                check_y_args = {
                    "enforce_univariate": enforce_univariate,
                    "enforce_multivariate": enforce_multivariate,
                    "enforce_index_type": enforce_index_type,
                    "allow_None": False,
                    "allow_empty": True,
                }

                y = check_series(y, **check_y_args, var_name="y")

                self._y_mtype_last_seen = mtype(y, as_scitype="Series")
        # end checking y

        # checking X
        if X is not None:
            if not isinstance(X.index, pd.MultiIndex):
                X = check_series(X, enforce_index_type=enforce_index_type, var_name="X")
                if self.get_tag("X-y-must-have-same-index"):
                    check_equal_time_index(X, y)
        # end checking X

        # convert X & y to supported inner type, if necessary
        #####################################################

        # retrieve supported mtypes

        # convert X and y to a supported internal mtype
        #  it X/y mtype is already supported, no conversion takes place
        #  if X/y is None, then no conversion takes place (returns None)
        if X is not None:
            if not isinstance(X.index, pd.MultiIndex):
                y_inner_mtype = self.get_tag("y_inner_mtype")
                y_inner = convert_to(
                    y,
                    to_type=y_inner_mtype,
                    as_scitype="Series",  # we are dealing with series
                    store=self._converter_store_y,
                )

                X_inner_mtype = self.get_tag("X_inner_mtype")
                X_inner = convert_to(
                    X,
                    to_type=X_inner_mtype,
                    as_scitype="Series",  # we are dealing with series
                )
            else:
                y_inner = y
                X_inner = X

        return X_inner, y_inner



class _DirectReducer(_Reducer):
    strategy = "direct"
    _tags = {
        "requires-fh-in-fit": True,  # is the forecasting horizon required in fit?
    }

    def _transform(self, y, X=None):
        fh = self.fh.to_relative(self.cutoff)
        return _sliding_window_transform(
            y,
            window_length=self.window_length,
            fh=fh,
            X=X,
            scitype=self._estimator_scitype,
        )

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        fh : int, list or np.array, optional (default=None)
             The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : Estimator
            An fitted instance of self.
        """
        # We currently only support out-of-sample predictions. For the direct
        # strategy, we need to check this at the beginning of fit, as the fh is
        # required for fitting.
        if not self.fh.is_all_out_of_sample(self.cutoff):
            raise NotImplementedError("In-sample predictions are not implemented.")

        self.window_length_ = check_window_length(
            self.window_length, n_timepoints=len(y)
        )

        yt, Xt = self._transform(y, X)

        # Iterate over forecasting horizon, fitting a separate estimator for each step.
        self.estimators_ = []
        for i in range(len(self.fh)):
            estimator = clone(self.estimator)
            estimator.fit(Xt, yt[:, i])
            self.estimators_.append(estimator)
        return self

    def fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets self._is_fitted flag to True.
            Writes self._y and self._X with `y` and `X`, respectively.
            Sets self.cutoff and self._cutoff to last index seen in `y`.
            Sets fitted model attributes ending in "_".
            Stores fh to self.fh if fh is passed.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                must have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                must have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
            if self.get_tag("requires-fh-in-fit"), must be passed, not optional
        X : pd.DataFrame, or 2D np.array, optional (default=None)
            Exogeneous time series to fit to
            if self.get_tag("X-y-must-have-same-index"), X.index must contain y.index

        Returns
        -------
        self : Reference to self.
        """
        # if fit is called, fitted state is re-set
        self._is_fitted = False

        self._set_fh(fh)

        # check and convert X/y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # set internal X/y to the new X/y
        # this also updates cutoff from y
        self._update_y_X(y_inner, X_inner)

        # checks and conversions complete, pass to inner fit
        #####################################################

        self._fit(y=y_inner, X=X_inner, fh=fh)

        # this should happen last
        self._is_fitted = True

        return self


    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        # Get last window of available data.
        y_last, X_last = self._get_last_window()

        # If we cannot generate a prediction from the available data, return nan.
        # danbartl- check removed
        # if not self._is_predictable(y_last):
        #     return self._predict_nan(fh)

        if self._X is None:
            n_columns = 1
        else:
            # X is ignored here, since we currently only look at lagged values for
            # exogenous variables and not contemporaneous ones.
            n_columns = self._X.shape[1] + 1

        # Pre-allocate arrays.
        window_length = self.window_length_
        X_pred = np.zeros((1, n_columns, window_length))

        # Fill pre-allocated arrays with available data.
        X_pred[:, 0, :] = y_last
        if self._X is not None:
            X_pred[:, 1:, :] = X_last.T

        # We need to make sure that X has the same order as used in fit.
        if self._estimator_scitype == "tabular-regressor":
            X_pred = X_pred.reshape(1, -1)

        # Allocate array for predictions.
        y_pred = np.zeros(len(fh))

        # Iterate over estimators/forecast horizon
        for i, estimator in enumerate(self.estimators_):
            y_pred[i] = estimator.predict(X_pred)

        return y_pred


class _MultioutputReducer(_Reducer):
    strategy = "multioutput"
    _tags = {
        "requires-fh-in-fit": True,  # is the forecasting horizon required in fit?
    }

    def _transform(self, y, X=None):
        fh = self.fh.to_relative(self.cutoff)
        return _sliding_window_transform(
            y,
            window_length=self.window_length,
            fh=fh,
            X=X,
            scitype=self._estimator_scitype,
        )

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        fh : int, list or np.array, optional (default=None)
             The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        # We currently only support out-of-sample predictions. For the direct
        # strategy, we need to check this at the beginning of fit, as the fh is
        # required for fitting.
        if not self.fh.is_all_out_of_sample(self.cutoff):
            raise NotImplementedError("In-sample predictions are not implemented.")

        self.window_length_ = check_window_length(
            self.window_length, n_timepoints=len(y)
        )

        yt, Xt = self._transform(y, X)

        # Fit a multi-output estimator to the transformed data.
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(Xt, yt)
        return self

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        # Get last window of available data.
        y_last, X_last = self._get_last_window()

        # If we cannot generate a prediction from the available data, return nan.
        if not self._is_predictable(y_last):
            return self._predict_nan(fh)

        if self._X is None:
            n_columns = 1
        else:
            # X is ignored here, since we currently only look at lagged values for
            # exogenous variables and not contemporaneous ones.
            n_columns = self._X.shape[1] + 1

        # Pre-allocate arrays.
        window_length = self.window_length_
        X_pred = np.zeros((1, n_columns, window_length))

        # Fill pre-allocated arrays with available data.
        X_pred[:, 0, :] = y_last
        if self._X is not None:
            X_pred[:, 1:, :] = X_last.T

        # We need to make sure that X has the same order as used in fit.
        if self._estimator_scitype == "tabular-regressor":
            X_pred = X_pred.reshape(1, -1)

        # Iterate over estimators/forecast horizon
        y_pred = self.estimator_.predict(X_pred)
        return y_pred.ravel()


class _RecursiveReducer(_Reducer):
    strategy = "recursive"
    _tags = {
        "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
    }

    def _transform(self, y, X=None):
        # For the recursive strategy, the forecasting horizon for the sliding-window
        # transform is simply a one-step ahead horizon, regardless of the horizon
        # used during prediction.
        fh = ForecastingHorizon([1])
        return _sliding_window_transform(
            y,
            self.window_length_,
            self.transformers_,
            fh,
            X,
            scitype=self._estimator_scitype,
        )

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        fh : int, list or np.array, optional (default=None)
             The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        self.window_length_ = check_window_length(
            self.window_length, n_timepoints=len(y)
        )

        self.transformers_ = self.transformers
        if self.window_length is None:
            if isinstance(self.transformers, list):
                truncate_start = self.transformers[0].fit()._truncate_start
                self.window_length_ = truncate_start
                self.window_length = truncate_start
            else:
                truncate_start = self.transformers.fit()._truncate_start
                self.window_length_ = truncate_start
                self.window_length = truncate_start

        yt, Xt = self._transform(y, X)

        # Make sure yt is 1d array to avoid DataConversion warning from scikit-learn.
        if isinstance(y.index, pd.MultiIndex):
            yt = yt["y"].ravel()
        else:
            yt = yt.ravel()

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(Xt, yt)

        return self

    def _get_shifted_window(self, y_update, X_update, shift):
        """Select last window."""
        # Get the start and end points of the last window.
        cutoff = _shift(self.cutoff, by=shift)
        start = _shift(cutoff, by=-self.window_length_ + 1)

        # danbartl: need to transform

        if isinstance(self._y.index, pd.MultiIndex):
            # Get the last window of the endogenous variable.

            # If X is given, also get the last window of the exogenous variables.
            dateline = pd.date_range(start=start, end=cutoff, freq=start.freq)
            tsids = X_update.index.get_level_values("ts_id").unique()

            mi = pd.MultiIndex.from_product(
                [tsids, dateline], names=["ts_id", "Period"]
            )

            # Create new y with old values and new forecasts
            y = pd.DataFrame(index=mi, columns=["y"])
            y.update(self._y)
            y.update(y_update)

            # Create new X with old values and new features derived from forecasts
            X = pd.DataFrame(index=mi, columns=self._X.columns)
            X.update(self._X)
            X.update(X_update)

            y = y.query("Period >= @start & Period <= @cutoff")
            X_cut = X_update.query("Period == @cutoff")

            ts_index = _get_time_index(y)
            n_timepoints = ts_index.shape[0]
            X_from_y = self.transformers.fit().transform(y)

            X_from_y_cut = X_from_y.groupby(level=0).tail(
                n_timepoints - self.window_length + 1
            )
            # X_from_y = LaggedWindowSummarizer(**model_kwargs,X)
            X_cut = X.groupby(level=0).tail(n_timepoints - self.window_length + 1)

            X = pd.concat([X_from_y_cut, X_cut], axis=1)

            # X_cut = X.groupby(level=0).tail(n_timepoints-maxlag)
            X = X.drop("y", axis=1)
            y = y.groupby(level=0).tail(1)
        else:
            # Get the last window of the endogenous variable.
            y = self._y.loc[start:cutoff].to_numpy()
            # If X is given, also get the last window of the exogenous variables.
            X = self._X.loc[start:cutoff].to_numpy() if self._X is not None else None

        return X

    def _predict_fixed_cutoff(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Make single-step or multi-step fixed cutoff predictions.

        Parameters
        ----------
        fh : np.array
            all positive (> 0)
        X : pd.DataFrame
        return_pred_int : bool
        alpha : float or array-like

        Returns
        -------
        y_pred = pd.Series
        """
        # assert all(fh > 0)
        y_pred = self._predict_last_window(
            fh, X, return_pred_int=return_pred_int, alpha=alpha
        )
        index = fh.to_absolute(self.cutoff)

        if isinstance(y_pred.index, pd.MultiIndex):
            y_return = y_pred
        else:
            y_return = pd.Series(y_pred, index=index)

        return y_return

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        if self._X is not None and X is None:
            raise ValueError(
                "`X` must be passed to `predict` if `X` is given in `fit`."
            )

        # Get last window of available data.
        y_last, X_last = self._get_last_window()

        # If we cannot generate a prediction from the available data, return nan.
        # danbartl: remove check
        # if not self._is_predictable(y_last):
        #     return self._predict_nan(fh)
        if isinstance(self._y.index, pd.MultiIndex):
            fh_max = fh.to_relative(self.cutoff)[-1]

            dateline = pd.date_range(
                end=fh.to_absolute(self.cutoff)[-1], periods=fh_max
            )
            #            fh.to_absolute(self.cutoff).to_pandas()
            tsids = y_last.index.get_level_values("ts_id")

            mi = pd.MultiIndex.from_product(
                [tsids, dateline], names=["ts_id", "Period"]
            )

            y_pred = pd.DataFrame(index=mi, columns=["y"])

            for i in range(fh_max):
                # Slice prediction window.
                date_curr = pd.date_range(end=dateline[i], periods=1)

                mi = pd.MultiIndex.from_product(
                    [tsids, date_curr], names=["ts_id", "Period"]
                )

                # Generate predictions.
                y_pred_vector = self.estimator_.predict(X_last)
                y_pred_curr = pd.DataFrame(y_pred_vector, index=mi, columns=["y"])
                # fh.to_absolute(self.cutoff).to_pandas()
                # tsids = y_last.index.get_level_values("ts_id")

                y_pred.update(y_pred_curr)
                # danbartl: check for horizon larger than one
                # danbartl should not take y_pred_curr but y_pred as input for fh > 2

                X_last = self._get_shifted_window(
                    y_update=y_pred, X_update=X, shift=i + 1
                )

                # y_pred[i] = self.estimator_.predict(X_pred)
                # # Update last window with previous prediction.
                # last[:, 0, window_length + i] = y_pred[i]
        else:
            # Pre-allocate arrays.
            if X is None:
                n_columns = 1
            else:
                n_columns = X.shape[1] + 1
            window_length = self.window_length_
            fh_max = fh.to_relative(self.cutoff)[-1]

            y_pred = np.zeros(fh_max)
            last = np.zeros((1, n_columns, window_length + fh_max))

            # Fill pre-allocated arrays with available data.
            last[:, 0, :window_length] = y_last
            if X is not None:
                last[:, 1:, :window_length] = X_last.T
                last[:, 1:, window_length:] = X.T

            # Recursively generate predictions by iterating over forecasting horizon.
            for i in range(fh_max):
                # Slice prediction window.
                X_pred = last[:, :, i : window_length + i]

                # Reshape data into tabular array.
                if self._estimator_scitype == "tabular-regressor":
                    X_pred = X_pred.reshape(1, -1)

                # Generate predictions.
                y_pred[i] = self.estimator_.predict(X_pred)

                # Update last window with previous prediction.
                last[:, 0, window_length + i] = y_pred[i]

        # While the recursive strategy requires to generate predictions for all steps
        # until the furthest step in the forecasting horizon, we only return the
        # requested ones.
        fh_idx = fh.to_indexer(self.cutoff)

        if isinstance(self._y.index, pd.MultiIndex):
            y_return = y_pred
        else:
            y_return = y_pred[fh_idx]

        return y_return


class _DirRecReducer(_Reducer):
    strategy = "dirrec"
    _tags = {
        "requires-fh-in-fit": True,  # is the forecasting horizon required in fit?
    }

    def _transform(self, y, X=None):
        # Note that the transform for dirrec is the same as in the direct
        # strategy.
        fh = self.fh.to_relative(self.cutoff)
        return _sliding_window_transform(
            y,
            window_length=self.window_length,
            fh=fh,
            X=X,
            scitype=self._estimator_scitype,
        )

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        fh : int, list or np.array, optional (default=None)
             The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : Estimator
            An fitted instance of self.
        """
        # Exogenous variables are not yet supported for the dirrec strategy.
        if X is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not yet support exogenous "
                f"variables `X`."
            )

        if len(self.fh.to_in_sample(self.cutoff)) > 0:
            raise NotImplementedError("In-sample predictions are not implemented")

        self.window_length_ = check_window_length(
            self.window_length, n_timepoints=len(y)
        )

        # Transform the data using sliding-window.
        yt, Xt = self._transform(y, X)

        # We cast the 2d tabular array into a 3d panel array to handle the data
        # consistently for the reduction to tabular and time-series regression.
        if self._estimator_scitype == "tabular-regressor":
            Xt = np.expand_dims(Xt, axis=1)

        # This only works without exogenous variables. To support exogenous
        # variables, we need additional values for X to fill the array
        # appropriately.
        X_full = np.concatenate([Xt, np.expand_dims(yt, axis=1)], axis=2)

        self.estimators_ = []
        n_timepoints = Xt.shape[2]

        for i in range(len(self.fh)):
            estimator = clone(self.estimator)

            # Slice data using expanding window.
            X_fit = X_full[:, :, : n_timepoints + i]

            # Convert to 2d tabular array for reduction to tabular regression.
            if self._estimator_scitype == "tabular-regressor":
                X_fit = X_fit.reshape(X_fit.shape[0], -1)

            estimator.fit(X_fit, yt[:, i])
            self.estimators_.append(estimator)

        self._is_fitted = True
        return self

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        # Exogenous variables are not yet support for the dirrec strategy.
        if X is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not yet support exogenous "
                f"variables `X`."
            )

        # Get last window of available data.
        y_last, X_last = self._get_last_window()

        if not self._is_predictable(y_last):
            return self._predict_nan(fh)

        window_length = self.window_length_

        # Pre-allocated arrays.
        # We set `n_columns` here to 1, because exogenous variables
        # are not yet supported.
        n_columns = 1
        X_full = np.zeros((1, n_columns, window_length + len(self.fh)))
        X_full[:, 0, :window_length] = y_last

        y_pred = np.zeros(len(fh))

        for i in range(len(self.fh)):

            # Slice data using expanding window.
            X_pred = X_full[:, :, : window_length + i]

            if self._estimator_scitype == "tabular-regressor":
                X_pred = X_pred.reshape(1, -1)

            y_pred[i] = self.estimators_[i].predict(X_pred)

            # Update the last window with previously predicted value.
            X_full[:, :, window_length + i] = y_pred[i]

        return y_pred


class DirectTabularRegressionForecaster(_DirectReducer):
    """Direct reduction from forecasting to tabular regression.

    For the direct reduction strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon.

    Parameters
    ----------
    estimator : Estimator
        A tabular regression estimator as provided by scikit-learn.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    """

    _estimator_scitype = "tabular-regressor"


class MultioutputTabularRegressionForecaster(_MultioutputReducer):
    """Multioutput reduction from forecasting to tabular regression.

    For the multioutput strategy, a single estimator capable of handling multioutput
    targets is fitted to all the future steps in the forecasting horizon.

    Parameters
    ----------
    estimator : Estimator
        A tabular regression estimator as provided by scikit-learn.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    """

    _estimator_scitype = "tabular-regressor"


class RecursiveTabularRegressionForecaster(_RecursiveReducer):
    """Recursive reduction from forecasting to tabular regression.

    For the recursive strategy, a single estimator is fit for a one-step-ahead
    forecasting horizon and then called iteratively to predict multiple steps ahead.

    Parameters
    ----------
    estimator : Estimator
        A tabular regression estimator as provided by scikit-learn.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    """

    _estimator_scitype = "tabular-regressor"


class DirRecTabularRegressionForecaster(_DirRecReducer):
    """Dir-rec reduction from forecasting to tabular regression.

    For the hybrid dir-rec strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon and then
    the previous forecasting horizon is added as an input
    for training the next forecaster, following the recusrive
    strategy.

    Parameters
    ----------
    estimator : sklearn estimator object
        Tabular regressor.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    """

    _estimator_scitype = "tabular-regressor"


class DirectTimeSeriesRegressionForecaster(_DirectReducer):
    """Direct reduction from forecasting to time-series regression.

    For the direct reduction strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon.

    Parameters
    ----------
    estimator : Estimator
        A time-series regression estimator as provided by sktime.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    """

    _estimator_scitype = "time-series-regressor"


class MultioutputTimeSeriesRegressionForecaster(_MultioutputReducer):
    """Multioutput reduction from forecasting to time series regression.

    For the multioutput strategy, a single estimator capable of handling multioutput
    targets is fitted to all the future steps in the forecasting horizon.

    Parameters
    ----------
    estimator : Estimator
        A time-series regression estimator as provided by sktime.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    """

    _estimator_scitype = "time-series-regressor"


class RecursiveTimeSeriesRegressionForecaster(_RecursiveReducer):
    """Recursive reduction from forecasting to time series regression.

    For the recursive strategy, a single estimator is fit for a one-step-ahead
    forecasting horizon and then called iteratively to predict multiple steps ahead.

    Parameters
    ----------
    estimator : Estimator
        A time-series regression estimator as provided by sktime.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    """

    _estimator_scitype = "time-series-regressor"


class DirRecTimeSeriesRegressionForecaster(_DirRecReducer):
    """Dir-rec reduction from forecasting to time-series regression.

    For the hybrid dir-rec strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon and then
    the previous forecasting horizon is added as an input
    for training the next forecaster, following the recusrive
    strategy.

    Parameters
    ----------
    estimator : sktime estimator object
        Time-series regressor.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    """

    _estimator_scitype = "time-series-regressor"


def make_reduction(
    estimator,
    strategy="recursive",
    window_length=10,
    scitype="infer",
    transformers=None,
):
    """Make forecaster based on reduction to tabular or time-series regression.

    During fitting, a sliding-window approach is used to first transform the
    time series into tabular or panel data, which is then used to fit a tabular or
    time-series regression estimator. During prediction, the last available data is
    used as input to the fitted regression estimator to generate forecasts.

    Parameters
    ----------
    estimator : an estimator instance
        Either a tabular regressor from scikit-learn or a time series regressor from
        sktime.
    strategy : str, optional (default="recursive")
        The strategy to generate forecasts. Must be one of "direct", "recursive" or
        "multioutput".
    window_length : int, optional (default=10)
        Window length used in sliding window transformation.
    scitype : str, optional (default="infer")
        Must be one of "infer", "tabular-regressor" or "time-series-regressor". If
        the scitype cannot be inferred, please specify it explicitly.
        See :term:`scitype`.

    Returns
    -------
    estimator : an Estimator instance
        A reduction forecaster

    References
    ----------
    ..[1] Bontempi, Gianluca & Ben Taieb, Souhaib & Le Borgne, Yann-Aël. (2013).
      Machine Learning Strategies for Time Series Forecasting.
    """
    # We provide this function as a factory method for user convenience.
    strategy = _check_strategy(strategy)
    scitype = _check_scitype(scitype)

    if scitype == "infer":
        scitype = _infer_scitype(estimator)

    Forecaster = _get_forecaster(scitype, strategy)
    return Forecaster(
        estimator=estimator, window_length=window_length, transformers=transformers
    )


def _check_scitype(scitype):
    valid_scitypes = ("infer", "tabular-regressor", "time-series-regressor")
    if scitype not in valid_scitypes:
        raise ValueError(
            f"Invalid `scitype`. `scitype` must be one of:"
            f" {valid_scitypes}, but found: {scitype}."
        )
    return scitype


def _infer_scitype(estimator):
    # We can check if estimator is an instance of scikit-learn's RegressorMixin or
    # of sktime's BaseRegressor, otherwise we raise an error. Note that some time-series
    # regressor also inherit from scikit-learn classes, hence the order in which we
    # check matters and we first need to check for BaseRegressor.
    if isinstance(estimator, BaseRegressor):
        return "time-series-regressor"
    elif isinstance(estimator, RegressorMixin):
        return "tabular-regressor"
    else:
        raise ValueError(
            "The `scitype` of the given `estimator` cannot be inferred. "
            "Please specify the `scitype` explicitly."
        )


def _check_strategy(strategy):
    valid_strategies = ("direct", "recursive", "multioutput", "dirrec")
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid `strategy`. `strategy` must be one of :"
            f" {valid_strategies}, but found: {strategy}."
        )
    return strategy


def _get_forecaster(scitype, strategy):
    """Select forecaster for a given scientific type and reduction strategy."""
    registry = {
        "tabular-regressor": {
            "direct": DirectTabularRegressionForecaster,
            "recursive": RecursiveTabularRegressionForecaster,
            "multioutput": MultioutputTabularRegressionForecaster,
            "dirrec": DirRecTabularRegressionForecaster,
        },
        "time-series-regressor": {
            "direct": DirectTimeSeriesRegressionForecaster,
            "recursive": RecursiveTimeSeriesRegressionForecaster,
            "multioutput": MultioutputTimeSeriesRegressionForecaster,
            "dirrec": DirRecTimeSeriesRegressionForecaster,
        },
    }
    return registry[scitype][strategy]


def _convert_new_to_old_pred_int(pred_int_new, alpha):
    name = pred_int_new.columns.get_level_values(0).unique()[0]
    alpha = check_alpha(alpha)
    pred_int_old_format = [
        pd.DataFrame(
            {
                "lower": pred_int_new[name, a / 2],
                "upper": pred_int_new[name, 1 - (a / 2)],
            }
        )
        for a in alpha
    ]

    # for a single alpha, return single pd.DataFrame
    if len(alpha) == 1:
        return pred_int_old_format[0]

    # otherwise return list of pd.DataFrames
    return pred_int_old_format