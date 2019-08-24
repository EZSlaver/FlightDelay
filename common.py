import datetime
import re
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
import sklearn


def replace_fucking_header_format(col):
    a = 97
    A = 65
    for i in range(26):
        col = re.sub(r'([a-z])%s' % chr(A + i), r"\1_" + chr(A + i), col)
        col = re.sub(r'([A-Z]+)%s([a-z])' % chr(A + i), r"\1_" + chr(A + i) + r"\2", col)

    return str.upper(col)


def standardize_df(df: pd.DataFrame):
    standard_cols = []
    for col in df.columns.values:
        col = col.replace('Dayof', 'DayOf')  # BASTARDS!
        col = replace_fucking_header_format(col)
        standard_cols.append(col)

    df.columns = standard_cols


def datetime_from_row(row, CRS_time_col):
    return datetime.datetime(
        year=row['YEAR'],
        month=row['MONTH'],
        day=row['DAY_OF_MONTH'],
        hour=int(row[CRS_time_col] / 100),
        minute=row[CRS_time_col] % 100
    )