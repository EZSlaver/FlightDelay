import pandas as pd
import itertools as it
import numpy as np
import os
import sys

from feature_creation_scripts.tail_num_tracking import get_n_prior_flights
from feature_creation_scripts.flight_load import LoadDFClass
from feature_creation_scripts.holidays import get_holiday_features_dict
from feature_creation_scripts.weather import get_weather_features_dict


source_path = '/home/dbeiski/Project/data/lga_dep_yearly/'  # Folder containing the lga departures files by year
load_source_path = '/home/dbeiski/Project/data/lga_load_data/'  # Folder containing lga load information
raw_monthly_source_path = '/home/dbeiski/Project/data/five_year/'  # Folder containing raw monthly data from internet
output_path = "/home/dbeiski/Project/data/lga_dep_yearly_enhanced/"  # Folder to output files

dep_prefix = 'lga_dep_'  # Prefix for departure files
load_prefix = 'lga_load'  # Prefix for load files

file_name_list = sorted(os.listdir(source_path))

n = 3  # Number of previous plane flights to track

ldfc = LoadDFClass(load_source_path, load_prefix)



def special_print(index, total):
    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write('Row {}/{}'.format(index, total - 1))
    sys.stdout.flush()


def add_new_columns(dep_df):
    tracking_cols1 = list(it.chain.from_iterable([['ArrDelay{}'.format(i), 'DepDelay{}'.format(i + 1)] for i in range(n)]))
    tracking_cols2 = ['IncomingAirTime', 'IncomingDistance', 'IncomingDistanceGroup']
    load_cols = ['ArrTimeBlkLoad', 'ArrDailyLoad', 'DepTimeBlkLoad', 'DepDailyLoad', 'TotalTimeBlkLoad', 'TotalDailyLoad']
    weather_cols = []
    holiday_cols = []
    new_cols = tracking_cols1 + tracking_cols2 + load_cols + weather_cols + holiday_cols
    for col in new_cols:
        dep_df[col] = np.nan
    return dep_df


def get_plane_tracking_features(dep_df, n, index):
    tail_num = dep_df.at[index, 'Tail_Number']
    month = dep_df.at[index, 'Month']
    year = dep_df.at[index, 'Year'] % 2000
    day = dep_df.at[index, 'DayofMonth']
    time = dep_df.at[index, 'CRSDepTime']
    flights = get_n_prior_flights(n, tail_num, year, month, day, time, raw_monthly_source_path)
    rtn_dict = {}
    for j, (i, series) in enumerate(flights.iterrows()):
        rtn_dict['ArrDelay{}'.format(j)] = flights.at[i, 'ArrDelay']
        if j < n:
            rtn_dict['DepDelay{}'.format(j + 1)] = flights.at[i, 'DepDelay']
        if j == 0:
            rtn_dict['IncomingAirTime'] = flights.at[i, 'AirTime']
            rtn_dict['IncomingDistance'] = flights.at[i, 'Distance']
            rtn_dict['IncomingDistanceGroup'] = flights.at[i, 'DistanceGroup']
    return rtn_dict


def get_load_features(dep_df, index):
    month = dep_df.at[index, 'Month']
    year = dep_df.at[index, 'Year'] % 2000
    day = dep_df.at[index, 'DayofMonth']
    time_blk = dep_df.at[index, 'DepTimeBlk']
    rtn_dict = {}
    for mode in ['Dep', 'Arr']:
        load_df = ldfc.get_load_df(year, month, mode.lower())
        try:
          time_blk_load = load_df.at[day - 1, time_blk]
        except KeyError as e:
            time_blk_load = 0
        rtn_dict.update({mode + 'TimeBlkLoad': time_blk_load,
                         mode + 'DailyLoad' : load_df.at[day - 1, 'daily']})
    rtn_dict.update({'TotalTimeBlkLoad': rtn_dict['ArrTimeBlkLoad'] + rtn_dict['DepTimeBlkLoad'],
                     'TotalDailyLoad' : rtn_dict['ArrDailyLoad'] + rtn_dict['DepDailyLoad']})
    return rtn_dict


def get_weather_features(dep_df, index):
    flight_date = dep_df.at[index, 'FlightDate']
    time = dep_df.at[index, 'CRSDepTime']
    return get_weather_features_dict(flight_date, time)


def get_holiday_features(dep_df, index):
    flight_date = dep_df.at[index, 'FlightDate']
    time = dep_df.at[index, 'CRSDepTime']
    return get_holiday_features_dict(flight_date)


if __name__ == '__main__':
    for file_name in file_name_list:
        print('Working on departures file ', file_name)
        dep_df = pd.read_csv(source_path + file_name)
        dep_df = add_new_columns(dep_df)
        for index in range(dep_df.shape[0]):
            special_print(index, dep_df.shape[0])
            new_features_dict = {}
            new_features_dict.update(get_plane_tracking_features(dep_df, n, index))
            new_features_dict.update(get_load_features(dep_df, index))
            new_features_dict.update(get_weather_features(dep_df, index))
            new_features_dict.update(get_holiday_features(dep_df, index))
            for feature, value in new_features_dict.items():
                dep_df.at[index, feature] = value
        dep_df.to_csv(output_path + 'enhanced_' + file_name)

