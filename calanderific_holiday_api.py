import pickle
import os
from datetime import datetime, timedelta, date
from enum import Enum
from typing import Iterable, Union

from ds_weather_api import BaseAPIWrapper

UNITED_STATES_API_ID = 'us'
UNITED_STATES_NY_API_ID = 'us-ny'


class HolidayType(Enum):
    LocalHoliday='Local holiday'
    Christian='Christian'
    SportingEvent='Sporting event'
    Hebrew='Hebrew'
    Season='Season'
    WorldwideObservance='Worldwide observance'
    NationalHoliday='National holiday'
    Hinduism='Hinduism'
    LocalObservance='Local observance'
    ClockChange='Clock change/Daylight Saving Time'
    Orthodox='Orthodox'
    Observance='Observance'
    Muslim='Muslim'
    UNObservance='United Nations observance'


class Holiday:

    def __init__(self):
        self.name = None
        self.description = None
        self.locations = None
        self.date  = None
        self.types  = None

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


class HolidayCalanderificAPIWrapper(BaseAPIWrapper):
    API_KEYS = ['ab70b09e9c65c4c892cd042adbbf714452b4563e']
    API_URL = 'https://calendarific.com/api/v2/holidays?&api_key={key}&country={country}&year={year}'

    @staticmethod
    def date_from_format(iso_str: str):
        return date.fromisoformat(iso_str[:10])

    def __init__(self, custom_keys=None):
        super().__init__(custom_keys or self.API_KEYS)

    def _get_holiday_data(self, report: dict) -> Holiday:

        data = Holiday()

        data.name = report['name']
        data.date = self.date_from_format(report['date']['iso'])
        data.description = report['description']
        data.types = []
        if not isinstance(report['type'], list):
            report['type'] = [report['type']]
        for type in report['type']:
            data.types.append(HolidayType(type))
        data.locations = []
        if not isinstance(report['locations'], list):
            report['locations'] = [report['locations']]
        for loc in report['locations']:
            data.locations.append(loc)

        return data

    def get_holiday_calander_by_place_and_year(self, country, years: Union[None, int, Iterable[int]] = None):

        if years is None:
            years = [datetime.now().year]

        if isinstance(years, int):
            years = [years]

        urls = []
        for year in years:
            urls.append(self.API_URL.format(key='{key}', country=country, year=year))

        observations_by_date_and_name = {}

        def callback(json):
            if json["meta"]['code'] != 200:
                return False

            for h_dict in json['response']['holidays']:
                holiday = self._get_holiday_data(h_dict)

                if holiday.date not in observations_by_date_and_name:
                    observations_by_date_and_name[holiday.date] = {}

                if holiday.name in observations_by_date_and_name[holiday.date]:
                    for loc in holiday.locations:
                        if loc not in observations_by_date_and_name[holiday.date][holiday.name].locations:
                            observations_by_date_and_name[holiday.date][holiday.name].locations.append(loc)
                    for t in holiday.types:
                        if t not in observations_by_date_and_name[holiday.date][holiday.name].types:
                            observations_by_date_and_name[holiday.date][holiday.name].types.append(t)
                else:
                    observations_by_date_and_name[holiday.date][holiday.name] = holiday

        self.make_api_calls(urls, callback)

        return observations_by_date_and_name


if __name__ == "__main__":

    years = list(range(2014,2020))

    ret = HolidayCalanderificAPIWrapper().get_holiday_calander_by_place_and_year(UNITED_STATES_NY_API_ID, years)

    with open(os.path.join('Data', 'HolidayData_NY_2014-2019.bin'), 'wb') as f:
        pickle.dump(ret, f)

    pass
