import numpy as np
import pandas as pd
import timeit

from sktime.datatypes._series._check import check_pddataframe_series
from sktime.utils.validation.series import is_integer_index
from sktime.datatypes._series._check import _index_equally_spaced
from sktime.datatypes._series._check import is_in_valid_index_types
from sktime.datatypes._utilities import get_time_index

from joblib import Parallel, delayed
import multiprocessing

def tmpFunc(df):
    df['c'] = df.a + df.b
    return df

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=4)(delayed(func)(group) for name, group in dfGrouped)
    return retLst
    
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

inst_inds = obj_indexed.index.get_level_values(0).unique()
inst_inds = np.unique(obj_indexed.index.get_level_values(0))

def np_all_old_df(X):
    diffs = np.diff(X)
    all_equal = np.all(diffs == diffs[0])
    return all_equal


ace1 =timeit.timeit(lambda: obj.groupby("time_series", as_index=False)["date"].apply(lambda x: np_all_old_df(x)), number =1)
ace2 = timeit.timeit(lambda: applyParallel(obj.groupby("time_series", as_index=False)["date"], np_all_old_df), number =1)
ace3 = timeit.timeit(lambda: [np_all_old_df(obj_indexed.loc[i].index) for i in inst_inds], number =1)

#timeit.timeit(lambda: obj_indexed.groupby(level="time_series", as_index=False).apply(lambda df: np_all_new(df.index.get_level_values(-1))), number =1)

import polars as pl

objpl = pl.DataFrame(obj)
ace4 = timeit.timeit(lambda: objpl.groupby("time_series").agg(pl.col("date").diff(null_behavior="drop").unique().len()).filter(pl.col("date")!=1), number = 1)

a=0