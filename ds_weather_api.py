import pickle
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Iterable, Union

import requests
from requests import RequestException

LAGURADIA_LAT = 40.7769  # ° N,
LAGURADIA_LON = -73.8740  # ° E


class WeatherLabel(Enum):
    Unknown = '<UNKNOWN>'
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
    Currently = 'currently'


class WeatherReport:

    def __init__(self, type: WeatherReportType, label: WeatherLabel):
        self.weather_label = label
        self.report_type = type

        self.time: Union[None, datetime] = None
        self.summary: Union[None, str] = None
        self.sunrise_time: Union[None, datetime] = None
        self.sunset_time: Union[None, datetime] = None
        self.moon_phase: Union[None, float] = None
        self.precipitation_intensity: Union[None, float] = None
        self.precipitation_intensity_max: Union[None, float] = None
        self.precipitation_intensity_max_time: Union[None, datetime] = None
        self.precipitation_probability: Union[None, float] = None
        self.precipitation_type: Union[None, WeatherLabel] = None
        self.temperature: Union[None, float] = None
        self.apparent_temperature: Union[None, float] = None
        self.temperature_high: Union[None, float] = None
        self.temperature_high_time: Union[None, datetime] = None
        self.temperature_low: Union[None, float] = None
        self.temperature_low_time: Union[None, datetime] = None
        self.apparent_temperature_high: Union[None, float] = None
        self.apparent_temperature_high_time: Union[None, datetime] = None
        self.apparent_temperature_low: Union[None, float] = None
        self.apparent_temperature_low_time: Union[None, datetime] = None
        self.dew_point: Union[None, float] = None
        self.humidity: Union[None, float] = None
        self.pressure: Union[None, float] = None
        self.wind_speed: Union[None, float] = None
        self.wind_gust: Union[None, float] = None
        self.wind_gust_time: Union[None, float] = None
        self.wind_bearing: Union[None, float] = None
        self.cloud_cover: Union[None, float] = None
        self.uv_index: Union[None, float] = None
        self.uv_index_time: Union[None, float] = None
        self.visibility: Union[None, float] = None
        self.ozone: Union[None, float] = None
        self.temperature_min: Union[None, float] = None
        self.temperature_min_time: Union[None, datetime] = None
        self.temperature_max: Union[None, float] = None
        self.temperature_max_time: Union[None, datetime] = None
        self.apparent_temperature_min: Union[None, float] = None
        self.apparent_temperature_min_time: Union[None, datetime] = None
        self.apparent_temperature_max: Union[None, float] = None
        self.apparent_temperature_max_time: Union[None, datetime] = None

    def __repr__(self):
        first = True
        ret = "("
        for attr in self.__dict__.keys():
            if attr.startswith("_") or self.__dict__[attr] is None:
                continue
            if not first:
                ret += ", "
            first = False
            ret += "{}: {}".format(attr, self.__dict__[attr])

        return ret + ")"

    def extend_with(self, another):
        new = WeatherReport(self.report_type, self.weather_label)

        for attr in self.__dict__.keys():
            if attr.startswith("_"):
                continue

            if self.__dict__[attr] is not None:
                new.__dict__[attr] = self.__dict__[attr]
            else:
                new.__dict__[attr] = another.__dict__[attr]

        return new


class BaseAPIWrapper:
    def __init__(self, api_keys):
        self._api_keys = api_keys

    def should_change_key(self, response: requests.Response):
        return response.status_code == 400

    def make_api_calls(self, urls, callback):

        usable_keys_list = self._api_keys.copy()

        tot = len(urls)
        observations = []

        for i, url in enumerate(urls):
            rerun = True
            while rerun:
                if len(usable_keys_list) == 0:
                    raise RequestException('Reached daily maximum.')

                response = requests.get(url.format(key=usable_keys_list[-1]), {
                    'User-Agent': "This is me: erezinman.ai@gmail.com"
                })

                json = response.json()

                rerun = self.should_change_key(response) or callback(json) is False

                if rerun:
                    usable_keys_list.pop()

            print('\rFinished %03d out of %03d (%5.2f%%).' % (i, tot, (i * 100 / tot)), end='')

        return observations


class WeatherDarkSkyAPIWrapper(BaseAPIWrapper):
    DARK_SKY_URL = 'https://api.darksky.net/forecast/{key}/{lat},{lon},{time}?units=si&extend=hourly'
    DARK_SKY_KEYS = ['0f88ed49ffa188f993bf53b490c5ff5d',
                     'c1b2cc4a6d9004dfff2d150cc343f3b7',
                     '2e42518033f5d4d7e12f8c696ed20eee',
                     '98d0865199dc5f7c861b0a126b07d1fe',
                     'a44ef0240ab97b9a3af6f932740bbcec']

    def __init__(self, custom_keys=None):
        super().__init__(custom_keys or self.DARK_SKY_KEYS)

    @staticmethod
    def date_time_to_format(date: datetime):
        return int((date - datetime(1970, 1, 1)).total_seconds())

    @staticmethod
    def date_time_from_format(form: str):
        return datetime.utcfromtimestamp(int(form))

    def _get_weather_data(self, report_type: WeatherReportType, report: dict) -> WeatherReport:

        label = WeatherLabel.Unknown

        if 'icon' in report:
            label = WeatherLabel(report['icon'])

        data = WeatherReport(report_type, label)

        if 'report_type' in report:
            data.report_type = float(report['report_type'])
        if 'time' in report:
            data.time = self.date_time_from_format(report['time'])
        if 'summary' in report:
            data.summary = report['summary']
        if 'sunriseTime' in report:
            data.sunrise_time = self.date_time_from_format(report['sunriseTime'])
        if 'sunsetTime' in report:
            data.sunset_time = self.date_time_from_format(report['sunsetTime'])
        if 'moonPhase' in report:
            data.moon_phase = float(report['moonPhase'])
        if 'precipIntensity' in report:
            data.precipitation_intensity = float(report['precipIntensity'])
        if 'precipIntensityMax' in report:
            data.precipitation_intensity_max = float(report['precipIntensityMax'])
        if 'precipIntensityMaxTime' in report:
            data.precipitation_intensity_max_time = self.date_time_from_format(report['precipIntensityMaxTime'])
        if 'precipProbability' in report:
            data.precipitation_probability = float(report['precipProbability'])
        if 'precipType' in report:
            data.precipitation_type = WeatherLabel(report['precipType'])
        if 'temperature' in report:
            data.temperature = float(report['temperature'])
        if 'temperatureHigh' in report:
            data.temperature_high = float(report['temperatureHigh'])
        if 'temperatureHighTime' in report:
            data.temperature_high_time = self.date_time_from_format(report['temperatureHighTime'])
        if 'temperatureLow' in report:
            data.temperature_low = float(report['temperatureLow'])
        if 'temperatureLowTime' in report:
            data.temperature_low_time = self.date_time_from_format(report['temperatureLowTime'])
        if 'apparentTemperature' in report:
            data.apparent_temperature = float(report['apparentTemperature'])
        if 'apparentTemperatureHigh' in report:
            data.apparent_temperature_high = float(report['apparentTemperatureHigh'])
        if 'apparentTemperatureHighTime' in report:
            data.apparent_temperature_high_time = self.date_time_from_format(report['apparentTemperatureHighTime'])
        if 'apparentTemperatureLow' in report:
            data.apparent_temperature_low = float(report['apparentTemperatureLow'])
        if 'apparentTemperatureLowTime' in report:
            data.apparent_temperature_low_time = self.date_time_from_format(report['apparentTemperatureLowTime'])
        if 'dewPoint' in report:
            data.dew_point = float(report['dewPoint'])
        if 'humidity' in report:
            data.humidity = float(report['humidity'])
        if 'pressure' in report:
            data.pressure = float(report['pressure'])
        if 'windSpeed' in report:
            data.wind_speed = float(report['windSpeed'])
        if 'windGust' in report:
            data.wind_gust = float(report['windGust'])
        if 'windGustTime' in report:
            data.wind_gust_time = float(report['windGustTime'])
        if 'windBearing' in report:
            data.wind_bearing = float(report['windBearing'])
        if 'cloudCover' in report:
            data.cloud_cover = float(report['cloudCover'])
        if 'uvIndex' in report:
            data.uv_index = float(report['uvIndex'])
        if 'uvIndexTime' in report:
            data.uv_index_time = float(report['uvIndexTime'])
        if 'visibility' in report:
            data.visibility = float(report['visibility'])
        if 'ozone' in report:
            data.ozone = float(report['ozone'])
        if 'temperatureMin' in report:
            data.temperature_min = float(report['temperatureMin'])
        if 'temperatureMinTime' in report:
            data.temperature_min_time = self.date_time_from_format(report['temperatureMinTime'])
        if 'temperatureMax' in report:
            data.temperature_max = float(report['temperatureMax'])
        if 'temperatureMaxTime' in report:
            data.temperature_max_time = float(report['temperatureMaxTime'])
        if 'apparentTemperatureMin' in report:
            data.apparent_temperature_min = float(report['apparentTemperatureMin'])
        if 'apparentTemperatureMinTime' in report:
            data.apparent_temperature_min_time = self.date_time_from_format(report['apparentTemperatureMinTime'])
        if 'apparentTemperatureMax' in report:
            data.apparent_temperature_max = self.date_time_from_format(report['apparentTemperatureMax'])
        if 'apparentTemperatureMaxTime' in report:
            data.apparent_temperature_max_time = float(report['apparentTemperatureMaxTime'])

        return data

    def _get_responses_by_type_from_json_dict(self, type, json_dict, output_dict):
        output_dict[type] = {}

        if 'data' not in json_dict:
            json_dict = {'data': [json_dict]}

        for rep in json_dict['data']:
            data = self._get_weather_data(type, rep)
            output_dict[type][data.time] = data

    def get_observations_by_lat_lon(self, lat, lon, time: Union[None, datetime, Iterable[datetime]] = None):

        if time is None:
            time = [datetime.now()]

        if isinstance(time, datetime):
            time = [time]

        observations = []
        urls = []
        for i, t in enumerate(time):
            urls.append(
                self.DARK_SKY_URL.format(key='{key}', lat=lat, lon=lon, time=self.date_time_to_format(t))
            )

        def callback(json):
            obs = {}
            for type in WeatherReportType:
                if type.value in json:
                    self._get_responses_by_type_from_json_dict(type, json[type.value], obs)
            if len(obs) == 0:
                return False
            observations.append(obs)
            return True

        self.make_api_calls(urls, callback)

        return observations

    def should_change_key(self, response: requests.Response):
        return super(WeatherDarkSkyAPIWrapper, self).should_change_key(response) \
               or int(response.headers['X-Forecast-API-Calls']) >= 1000


if __name__ == "__main__":

    start = datetime(year=2014, month=5, day=1) - timedelta(days=1)
    end = datetime(year=2019, month=5, day=31) + timedelta(days=1)
    times = []
    while start.timestamp() <= end.timestamp():
        times.append(start)
        start += timedelta(days=1)

    ret = WeatherDarkSkyAPIWrapper() \
        .get_observations_by_lat_lon(LAGURADIA_LAT, LAGURADIA_LON, times)

    with open(os.path.join('Data', 'WeatherData_May2018-May2019.bin'), 'wb') as f:
        pickle.dump(ret, f)

    pass
