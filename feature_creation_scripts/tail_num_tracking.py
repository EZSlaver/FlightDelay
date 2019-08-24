import pandas as pd
import itertools as it
import numpy as np


global CACHE
CACHE = {}

COLS = ['FlightDate', 'CRSDepTime', 'Tail_Number', 'ArrDelay', 'DepDelay', 'AirTime', 'DistanceGroup', 'Distance', 'Origin', 'Dest']

def get_df(strng, source_path):
    global CACHE
    if strng not in CACHE:
        df = pd.read_csv(source_path + strng + '.csv', usecols=COLS)
        df['DateTime'] = df['FlightDate'] + '-' + df['CRSDepTime'].map('{:04}'.format)
        df = df.sort_values(by=['DateTime'], ascending=False)
        CACHE[strng] = df
    if len(CACHE) > 3:
        to_del = sorted(CACHE.keys())[0]
        CACHE.pop(to_del)
    return CACHE[strng]


def get_n_prior_flights(n, tail_num, year, month, day, time, source_path, last=False):
    month_str = '{:02}'.format(month)
    day_str = '{:02}'.format(day)
    date_time = '-'.join(['20' + str(year), month_str, day_str, '{:04}'.format(time)])
    df = get_df(str(year) + '_' + month_str, source_path)
    flights = df[df['Tail_Number'] == tail_num]
    flights = flights[flights['DateTime'] < date_time]
    flights = flights.iloc[: n]
    if flights.shape[0] < n and not last:
        year = year - 1 if month == 1 else year
        month = month - 1 if month != 1 else 12
        day = 31
        time = 2400
        flights2 = get_n_prior_flights(n - flights.shape[0], tail_num, year, month, day, time, source_path, last=True)
        flights = pd.concat([flights, flights2])
    return flights


# if __name__ == '__main__':
#     new_cols = list(it.chain.from_iterable([['ARR_DELAY_{}'.format(i), 'DEP_DELAY_{}'.format(i + 1)] for i in range(n)]))
#     for col in new_cols:
#         dep_df[col] = np.nan
#     for i in range(dep_df.shape[0]):
#         print(i)
#         tail_num = dep_df.at[i, 'TAIL_NUM']
#         month = dep_df.at[i, 'MONTH']
#         year = dep_df.at[i, 'YEAR'] % 2000
#         day = dep_df.at[i, 'DAY_OF_MONTH']
#         time = dep_df.at[i, 'CRS_DEP_TIME']
#         flights = get_n_prior_flights(n, tail_num, year, month, day, time)
#         for j, (index, series) in enumerate(flights.iterrows()):
#             dep_df.at[i, 'ARR_DELAY_{}'.format(j)] = flights.at[index, 'ARR_DELAY']
#             if j < n:
#                 dep_df.at[i, 'DEP_DELAY_{}'.format(j + 1)] = flights.at[index, 'DEP_DELAY']
#     dep_df.to_csv(out_file)
#
