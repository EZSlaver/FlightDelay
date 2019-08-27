import pandas as pd
import itertools as it
import numpy as np
import os
import sys
import pickle

from feature_creation_scripts.tail_num_tracking import get_n_prior_flights
from feature_creation_scripts.flight_load import LoadDFClass
from feature_creation_scripts.holidays import get_holiday_features_dict
from feature_creation_scripts.weather import get_weather_features_dict


source_path = '/home/dbeiski/Project/data/lga_dep_yearly_enhanced/'  # Folder containing the lga departures files by year
output_path = "/home/dbeiski/Project/data/lga_dep_yearly_enhanced/"  # Folder to output files
plane_data_path = '/home/dbeiski/Project/data/tail_num_data/'

dep_prefix = 'lga_dep_'  # Prefix for departure files
load_prefix = 'lga_load'  # Prefix for load files

file_name_list = sorted(os.listdir(source_path))

with open(plane_data_path + 'file.bin', 'rb') as f:
    tail_seats_dict = pickle.load(f)


def special_print(index, total):
    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write('Row {}/{}'.format(index, total - 1))
    sys.stdout.flush()


def add_new_columns(dep_df):
    plane_cols = ['SeatsNum', 'PlaneModel']
    new_cols = plane_cols
    for col in new_cols:
        dep_df[col] = np.nan
    return dep_df


def get_holiday_features(date_string):
    return get_holiday_features_dict(date_string)



if __name__ == '__main__':
    for file_name in file_name_list:
        year = 2000 + int(file_name[17: 19])
        print('\nWorking on departures file ', file_name)
        dep_df = pd.read_csv(source_path + file_name)
        dep_df = add_new_columns(dep_df)
        for tail_num in tail_seats_dict:
            bool_mask = dep_df['Tail_Number'] == ('N' + tail_num)
            if not np.any(bool_mask):
                continue
            seat_num = tail_seats_dict[tail_num]['seat_number']
            seat_num = int(seat_num) if seat_num is not None else None
            dep_df.loc[bool_mask, ['SeatsNum', 'PlaneModel']] = \
                seat_num, tail_seats_dict[tail_num]['aircraft_model']
        print('Doint Holidays')
        if year >= 2017:
            dep_df.to_csv(output_path + file_name)
            continue
        for month in range(1, 13):
            for day in range(1, 32):
                date_string = '-'.join([str(year), '{:02}'.format(month), '{:02}'.format(day)])
                holiday_dict = get_holiday_features(date_string)
                bool_mask = dep_df['FlightDate'] == date_string
                if not np.any(bool_mask):
                    continue
                dep_df.loc[bool_mask, ['NATIONAL_HOLIDAY', 'CHRISTIAN']] = \
                    holiday_dict['NATIONAL_HOLIDAY'], holiday_dict['CHRISTIAN']

        dep_df.to_csv(output_path + file_name)


