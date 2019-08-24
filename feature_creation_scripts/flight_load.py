import numpy as np
import pandas as pd
import os

source_path = '/home/dbeiski/Project/data/national_org/'
target_path = '/home/dbeiski/Project/data/lga_load_data/'

def create_load_df(df, mode):
    df_raw = df.groupby(by=[mode + '_TIME_BLK', 'DAY_OF_MONTH']).size()
    rtn_df = pd.DataFrame({col: df_raw[col] for col in df_raw.index.levels[0]})
    rtn_df['daily'] = rtn_df.apply(lambda row: np.sum(row), axis=1)
    return rtn_df



if __name__ == "__main__":
    for file_name in sorted(os.listdir(source_path)):
        if file_name != '18_12.csv':
            continue
        df = pd.read_csv(source_path + file_name)
        # df["DAY_OF_YEAR"] = df["MONTH"].map(str) + '_' + df["DAY_OF_MONTH"].map('{:02}'.format)
        df_dep = df[df['ORIGIN'] == 'LGA']
        df_arr = df[df['DEST'] == 'LGA']
        dep_load_df = create_load_df(df_dep, 'DEP')
        # dep_load_df.to_csv(target_path + 'lga_load_dep_' + file_name)
        arr_load_df = create_load_df(df_dep, 'ARR')
        # arr_load_df.to_csv(target_path + 'lga_load_arr_' + file_name)
        pass


