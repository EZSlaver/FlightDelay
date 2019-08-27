import requests
import pandas as pd
import os
import pickle
import sys
import time


url = 'https://he.flightaware.com/resources/registration/'
source_path = '/home/dbeiski/Project/data/lga_dep_yearly/'
target_path = '/home/dbeiski/Project/data/tail_num_data/'


def special_print(index, total):
    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write('Tail_num {}/{}'.format(index, total - 1))
    sys.stdout.flush()



# if __name__ == '__main__':
#     pickles_list = os.listdir(target_path)
#     if pickles_list:
#         with open(target_path + sorted(pickles_list)[-1], 'rb') as in_f:
#             data_dict = pickle.load(in_f)
#     else:
#         data_dict = {}
#     file_name_list = sorted(os.listdir(source_path))
#     counter = 1
#     for file_name in file_name_list:
#         print('File: ', file_name)
#         tail_num_set = set(pd.read_csv(source_path + file_name, usecols=['Tail_Number']).to_numpy(str).ravel())
#         for i, tail_num in enumerate(tail_num_set):
#             if tail_num in data_dict:
#                 continue
#             special_print(i, len(tail_num_set))
#             try:
#                 response = requests.get(url + tail_num)
#                 if response.status_code != 200:
#                     time.sleep(5)
#                 else:
#                     time.sleep(0.05)
#                 raw_html = response.text
#                 index = raw_html.index('seats')
#             except ValueError as e:
#                 print('\n', e)
#                 print('Tail_number= ', tail_num, '\n')
#                 data_dict[tail_num] = None
#                 continue
#             seat_num_string = raw_html[index-5: index-1]
#             seat_num = int(''.join((ch for ch in seat_num_string if ch.isnumeric())))
#             data_dict[tail_num] = seat_num
#         with open(target_path + 'seats_final.pickle'.format(counter), 'wb') as out_f:
#             pickle.dump(data_dict, out_f)
#             print('Saved pickle number ', counter)
file_name_list = sorted(os.listdir(source_path))
tail_num_set = set([])
for file_name in file_name_list:
    tail_num_set.update(set(pd.read_csv(source_path + file_name, usecols=['Tail_Number']).values.ravel()))
    print(len(tail_num_set))
tail_num_list = list(tail_num_set)
with open(target_path + 'tail_num_list.pickle', 'wb') as f:
    pickle.dump(tail_num_list, f)
