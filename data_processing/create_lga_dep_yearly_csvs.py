import numpy as np
import pandas as pd
import os


source_path = '/home/dbeiski/Project/data/five_year/'  # Folder containing raw monthly data from the site
target_path = '/home/dbeiski/Project/data/lga_dep_yearly/'  # Folder to save departures from lga in

load_target_path = '/home/dbeiski/Project/data/lga_load_data/'  # Folder to save monthly load data of lga

cols_to_keep = ['Year',
                'Month',
                'DayofMonth',
                'DayOfWeek',
                'FlightDate',
                'Reporting_Airline',
                'Tail_Number',
                'Origin',
                'OriginCityMarketID',
                'Dest',
                'DestCityMarketID',
                'CRSDepTime',
                'DepTime',
                'DepDelay',
                'DepDelayMinutes',
                'DepDel15',
                'DepartureDelayGroups',
                'DepTimeBlk',
                'TaxiOut',
                'WheelsOff',
                'WheelsOn',
                'TaxiIn',
                'CRSArrTime',
                'ArrTime',
                'ArrDelay',
                'ArrDelayMinutes',
                'ArrDel15',
                'ArrivalDelayGroups',
                'ArrTimeBlk',
                'CRSElapsedTime',
                'ActualElapsedTime',
                'AirTime',
                'Distance',
                'DistanceGroup',
                'CarrierDelay',
                'WeatherDelay',
                'NASDelay',
                'SecurityDelay',
                'LateAircraftDelay',
                'Cancelled',
                'Diverted']


def create_load_df(df, mode):
    df_raw = df.groupby(by=[mode + 'TimeBlk', 'DayofMonth']).size()
    rtn_df = pd.DataFrame({col: df_raw[col] for col in df_raw.index.levels[0]})
    rtn_df['daily'] = rtn_df.apply(lambda row: np.sum(row), axis=1)
    rtn_df = rtn_df.fillna(0)
    return rtn_df


def create_and_save_load_db_and_return_dep_df(df, file_name):
    df_dep = df[df['Origin'] == 'LGA']
    df_arr = df[df['Dest'] == 'LGA']
    dep_load_df = create_load_df(df_dep, 'Dep')
    dep_load_df.to_csv(load_target_path + 'lga_load_dep_' + file_name)
    arr_load_df = create_load_df(df_arr, 'Arr')
    arr_load_df.to_csv(load_target_path + 'lga_load_arr_' + file_name)
    return df_dep


file_name_list = sorted(os.listdir(source_path))[1: ]
year_prev = '0'
for file_name in file_name_list:
    print(file_name)
    year = file_name[: 2]
    df = pd.read_csv(source_path + file_name, usecols=cols_to_keep)
    df = create_and_save_load_db_and_return_dep_df(df, file_name)
    df = df[(df['Cancelled'] == 0) & (df['Diverted'] == 0)]
    df = df.drop(columns=['Cancelled', 'Diverted'], axis=1)
    df['DayOfYear'] = df["Month"].map('{:02}'.format) + '-' + df["DayofMonth"].map('{:02}'.format)
    df = df.sort_values(by=['FlightDate', 'CRSDepTime'])
    if year == year_prev:
        with open(target_path + 'lga_dep_' + year + '.csv', mode='a') as f:
            df.to_csv(f, header=False)
    else:
        year_prev = year
        df.to_csv(target_path + 'lga_dep_' + year + '.csv')



