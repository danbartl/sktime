import numpy as np
import pandas as pd
import timeit

from sktime.datatypes._series._check import check_pddataframe_series
from sktime.utils.validation.series import is_integer_index
from sktime.datatypes._series._check import _index_equally_spaced
from sktime.datatypes._series._check import is_in_valid_index_types
from sktime.datatypes._utilities import get_time_index
from sktime.datatypes._series._check import check_pddataframe_series
from joblib import Parallel, delayed
import multiprocessing

def tmpFunc(df):
    df['c'] = df.a + df.b
    return df

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=4)(delayed(func)(group) for name, group in dfGrouped)
    return retLst


def np_all_old_df(X):
    diffs = np.diff(X)
    all_equal = np.all(diffs == diffs[0])
    return all_equal

obj_raw = pd.read_pickle("data.pickle")

obj_1 = obj_raw.reset_index()

obj_list = [obj_1,obj_1,obj_1,obj_1,obj_1,obj_1,obj_1,obj_1,obj_1]

for index,df in enumerate(obj_list):
    df = obj_list[index].copy()
    df["time_series"] = df["time_series"] + "flag" + str(index)
    obj_list[index] = df

obj = pd.concat(obj_list)
obj["date"] = obj["date"].dt.to_timestamp()
obj_indexed = obj.set_index(["time_series", "date"])

#obj_indexed.reset_index().groupby("time_series", as_index=False,group_keys=False).apply(lambda x: [check_pddataframe_series(x.set_index("date"), return_metadata = True)])

inst_inds = obj_indexed.index.get_level_values(0).unique()
inst_inds = np.unique(obj_indexed.index.get_level_values(0))

period_groupby =timeit.timeit(lambda: obj.groupby("time_series", as_index=False)["date"].apply(lambda x: np_all_old_df(x)), number =1)
datetime_parallel = timeit.timeit(lambda: applyParallel(obj.groupby("time_series", as_index=False)["date"], np_all_old_df), number =1)
datetime_list = timeit.timeit(lambda: [np_all_old_df(obj_indexed.loc[i].index) for i in inst_inds], number =1)

datetime_groupby_1 = timeit.timeit(lambda: obj_indexed.groupby(level="time_series",group_keys=False, as_index=True).apply(lambda df: check_pddataframe_series(df.droplevel(0), return_metadata=True)), number =1)
datetime_groupby_2 = timeit.timeit(lambda: obj_indexed.reset_index(-1).groupby(level="time_series").apply(lambda df: pd.DataFrame(check_pddataframe_series(df.set_index(obj_indexed.index.names[-1]), return_metadata=True))), number =1)
datetime_list = timeit.timeit(lambda: [check_pddataframe_series(obj_indexed.loc[i], return_metadata=True) for i in inst_inds], number =1)


import polars as pl
objpl = pl.DataFrame(obj)
polars_test = timeit.timeit(lambda: objpl.groupby("time_series").agg(pl.col("date").diff(null_behavior="drop").unique().len()).filter(pl.col("date")!=1), number = 1)

a=0

#timeit.timeit(lambda: obj_indexed.groupby(level="time_series", as_index=False).apply(lambda df: np_all_new(df.index.get_level_values(-1))), number =1)

obj = pd.concat(obj_list)
obj_indexed = obj.set_index(["time_series", "date"])

#obj_indexed.reset_index().groupby("time_series", as_index=False,group_keys=False).apply(lambda x: [check_pddataframe_series(x.set_index("date"), return_metadata = True)])

inst_inds = obj_indexed.index.get_level_values(0).unique()
inst_inds = np.unique(obj_indexed.index.get_level_values(0))

period_groupby =timeit.timeit(lambda: obj.groupby("time_series", as_index=False)["date"].apply(lambda x: np_all_old_df(x)), number =1)
period_parallel = timeit.timeit(lambda: applyParallel(obj.groupby("time_series", as_index=False)["date"], np_all_old_df), number =1)
period_list = timeit.timeit(lambda: [np_all_old_df(obj_indexed.loc[i].index) for i in inst_inds], number =1)


period_groupby_1 = timeit.timeit(lambda: [check_pddataframe_series(obj_indexed.loc[i], return_metadata=True) for i in inst_inds], number =1)
period_groupby_2 = timeit.timeit(lambda: obj_indexed.reset_index(-1).groupby(level="time_series").apply(lambda df: pd.DataFrame(check_pddataframe_series(df.set_index(obj_indexed.index.names[-1]), return_metadata=True))), number =1)
period_list = timeit.timeit(lambda: obj_indexed.groupby(level="time_series",group_keys=False, as_index=True).apply(lambda df: check_pddataframe_series(df.droplevel(0), return_metadata=True)), number =1)


