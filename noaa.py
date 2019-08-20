from typing import Iterable, List

import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
import sklearn
from noaa_sdk import noaa
from datetime import datetime, timedelta
import requests


class WeatherData:
    API_OUT_DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S%z'

    def __init__(self, data_dict):
        self.timestep = datetime.strptime(data_dict['format'], self.API_OUT_DATETIME_FORMAT)
        self.temperature_C = data_dict['temperature']['value']
        self.dewpoint = data_dict['dewpoint']['value']
        self.wind_angle = data_dict['dewpoint']['value']


class WeatherAPIWrapper:
    API_IN_DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S%z'
    # API_IN_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
    NEW_YORK_STATION_ID = 'KNYC'

    def __init__(self):
        self._n = noaa.NOAA()

    def get_observations_by_station(self, station_id, start_time: datetime = None, end_time: datetime = None,
                                    limit=None) -> List[WeatherData]:

        if start_time is None:
            start_time = datetime.now()

        uri = 'https://api.weather.gov/stations/{}/observations?start={}' \
            .format(station_id, start_time.strftime(self.API_IN_DATETIME_FORMAT))

        if end_time is not None:
            uri += '&end=' + end_time.strftime(self.API_IN_DATETIME_FORMAT)

        if limit is not None:
            uri += '&limit=' + str(limit)

        response = requests.get(uri, {
            'accept': 'application/geo+json',
            'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:68.0) Gecko/20100101 Firefox/68.0"
        })

        json = response.json()
        return []
        #
        # for obs in self._n.get_observations('11371', 'US',
        #                                     start=start.strftime(self.API_IN_DATETIME_FORMAT),
        #                                     end=end.strftime(self.API_IN_DATETIME_FORMAT)):
        #     # This is the latest
        #     return WeatherData(obs)


if __name__ == "__main__":
    WeatherAPIWrapper().get_observation_at_time(datetime.now() - timedelta(days=365))
