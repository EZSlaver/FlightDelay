import pandas as pd
import itertools as it
import numpy as np

all_path = '/home/dbeiski/Project/data/national_org/'
dep_df = pd.read_csv('/home/dbeiski/Project/FlightDelay/Data/new_york/year_lga_dep.csv')
out_file = '/home/dbeiski/Project/FlightDelay/Data/new_york/year_lga_dep_w_prev_flights.csv'
n = 3

global CACHE
CACHE = {}

def get_df(strng):
    global CACHE
    if strng not in CACHE:
        CACHE[strng] = pd.read_csv(str(all_path) + strng + '.csv')
    if len(CACHE) > 3:
        to_del = sorted(CACHE.keys())[0]
        CACHE.pop(to_del)
    return CACHE[strng]


def get_n_prior_flights(n, tail_num, year, month, day, time, last=False):
    month_str = '{:02}'.format(month)
    df = get_df(str(year) + '_' + month_str)
    flights = df[df['TAIL_NUM'] == tail_num]
    flights = flights[flights['DAY_OF_MONTH'] <= day]
    flights = flights[flights['CRS_DEP_TIME'] < time]
    flights = flights.sort_values(by=['DAY_OF_MONTH', 'CRS_DEP_TIME'], ascending=False)
    flights = flights.iloc[: n]
    if flights.shape[0] < n and not last:
        year = year - 1 if month == 1 else year
        month = month - 1 if month != 1 else 12
        day = 31
        time = 2359
        flights2 = get_n_prior_flights(n - flights.shape[0], tail_num, year, month, day, time, last=True)
        flights = pd.concat([flights, flights2])
    return flights


if __name__ == '__main__':
    new_cols = list(it.chain.from_iterable([['ARR_DELAY_{}'.format(i), 'DEP_DELAY_{}'.format(i + 1)] for i in range(n)]))
    for col in new_cols:
        dep_df[col] = np.nan
    for i in range(dep_df.shape[0]):
        print(i)
        tail_num = dep_df.at[i, 'TAIL_NUM']
        month = dep_df.at[i, 'MONTH']
        year = dep_df.at[i, 'YEAR'] % 2000
        day = dep_df.at[i, 'DAY_OF_MONTH']
        time = dep_df.at[i, 'CRS_DEP_TIME']
        flights = get_n_prior_flights(n, tail_num, year, month, day, time)
        for j, (index, series) in enumerate(flights.iterrows()):
            dep_df.at[i, 'ARR_DELAY_{}'.format(j)] = flights.at[index, 'ARR_DELAY']
            if j < n:
                dep_df.at[i, 'DEP_DELAY_{}'.format(j + 1)] = flights.at[index, 'DEP_DELAY']
    dep_df.to_csv(out_file)

