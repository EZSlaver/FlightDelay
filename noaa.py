from enum import Enum
from typing import Iterable, List, Union

import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
import sklearn
from noaa_sdk import noaa
from datetime import datetime, timedelta
import datetime as dt
import requests
import pickle

LAGURADIA_LAT = 40.7769  # ° N,
LAGURADIA_LON = -73.8740  # ° E


class WeatherLabel(Enum):
    ClearDay = 'clear-day'
    ClearNight = 'clear-night'
    Rain = 'rain'
    Snow = 'snow'
    Sleet = 'sleet'
    Wind = 'wind'
    Fog = 'fog'
    Cloudy = 'cloudy'
    PartlyCloudyDay = 'partly-cloudy-day'
    PartlyCloudyNight = 'partly-cloudy-night'
    # Unused but reserved
    Hail = 'hail'
    Thunderstorm = 'thunderstorm'
    Tornado = 'tornado'


class WeatherReportType(Enum):
    Monthly = 'monthly'
    Weekly = 'weekly'
    Daily = 'daily'
    Hourly = 'hourly'
    Minutely = 'minutely'


class WeatherData:
    # API_OUT_DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S%z'

    def __init__(self, time_step, weather_label, ):
        self.timestep = dt.datetime.fromisoformat(data_dict['format'])
        self.temperature_C = data_dict['temperature']['value']
        self.dewpoint = data_dict['dewpoint']['value']
        self.wind_angle = data_dict['dewpoint']['value']


class WeatherAPIWrapper:
    DARK_CITY_KEY = '0f88ed49ffa188f993bf53b490c5ff5d'
    DARK_CITY_URL = 'https://api.darksky.net/forecast/{key}/{lat},{lon},{time}?units=si&extend=hourly'

    @staticmethod
    def date_time_to_format(date: datetime):
        return (date - datetime(1970, 1, 1)).total_seconds()

    @staticmethod
    def date_time_from_format(form: str):
        return datetime.utcfromtimestamp(int(form))

    #
    # def __init__(self):
    #     self._n = noaa.NOAA()

    def get_observations_by_lat_lon(self, lat, lon, time: Union[None, datetime, Iterable[datetime]] = None):

        if time is None:
            time = [datetime.now()]

        if isinstance(time, datetime):
            time = [time]

        base_uri = self.DARK_CITY_URL.format(key=self.DARK_CITY_KEY, lat=lat, lon=lon)

        for t in time:
            uri = base_uri.format(time=t)

            response = requests.get(uri, {
                'User-Agent': "This is me: erezinman.ai@gmail.com"
            })
            return response.json()

        with open('pickle.pckl', 'w') as f:
            pickle.dump([json])
        return []
        #
        # for obs in self._n.get_observations('11371', 'US',
        #                                     start=start.strftime(self.API_IN_DATETIME_FORMAT),
        #                                     end=end.strftime(self.API_IN_DATETIME_FORMAT)):
        #     # This is the latest
        #     return WeatherData(obs)


if __name__ == "__main__":
    WeatherAPIWrapper().get_observations_by_station(
        WeatherAPIWrapper.NEW_YORK_STATION_ID,
        start_time=datetime.now() - timedelta(days=365),
        limit=3)
