import pandas as pd

class LoadDFClass(object):

    def __init__(self, source_path, prefix):
        self.source_path = source_path
        self.prefix = prefix
        self.year = 0
        self.month = 0
        self.dep_df = None
        self.arr_df = None

    def get_load_df(self, year, month, mode):
        if year != self.year or month != self.month:
            self.year = year
            self.month = month
            self.update_dfs(year, month)
        return self.__getattribute__(mode + '_df')

    def update_dfs(self, year, month):
        for mode in ['arr', 'dep']:
            file_name = '_'.join([self.prefix, mode, '{:02}'.format(year), '{:02}'.format(month)]) + '.csv'
            self.__setattr__(mode + '_df', pd.read_csv(self.source_path + file_name))
